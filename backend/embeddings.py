import os
import logging
from typing import List, Optional, Set
import numpy as np

from tokenizers import Tokenizer
import onnxruntime as ort
from langchain_core.embeddings import Embeddings
from huggingface_hub import hf_hub_download

logger = logging.getLogger("asktube.embeddings")

DEFAULT_MODEL_REPO = "Xenova/multilingual-e5-small"
DEFAULT_ONNX_FILENAME = "onnx/model_int8.onnx"
DEFAULT_TOKENIZER_ID = "intfloat/multilingual-e5-small"


class ONNXE5Embeddings(Embeddings):
    """
    Lightweight, PyTorch-free multilingual embedding implementation using
    ONNX Runtime INT8 and HuggingFace Rust Tokenizers.

    Designed specifically for Render Free tier (512 MB cgroup memory limit)
    to achieve multilingual accuracy (Hindi, English, etc.) without loading
    PyTorch into the Python process.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_id: str = DEFAULT_TOKENIZER_ID,
        repo_id: str = DEFAULT_MODEL_REPO,
        onnx_filename: str = DEFAULT_ONNX_FILENAME,
        batch_size: int = 4,
        max_length: int = 512,
    ):
        self.batch_size = max(1, int(batch_size))
        self.max_length = int(max_length)
        self.tokenizer_id = tokenizer_id

        # 1. Resolve ONNX Model Path
        resolved_model_path = self._resolve_model_path(model_path, repo_id, onnx_filename)
        logger.info(f"Initializing ONNX session from {resolved_model_path}")

        # 2. Configure Memory-Safe Session Options
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        # Crucial for 512MB Render Free: disable large pre-allocated memory chunk arena
        opts.enable_cpu_mem_arena = False
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            resolved_model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_names: Set[str] = {inp.name for inp in self.session.get_inputs()}

        # 3. Initialize Tokenizer (Pure Rust tokenizers, avoids torch)
        logger.info(f"Loading tokenizer {tokenizer_id}")
        self.tokenizer = Tokenizer.from_pretrained(tokenizer_id)
        self.tokenizer.enable_padding(length=None)
        self.tokenizer.enable_truncation(max_length=self.max_length)

    def _resolve_model_path(
        self,
        model_path: Optional[str],
        repo_id: str,
        onnx_filename: str,
    ) -> str:
        if model_path and os.path.isfile(model_path):
            return model_path

        env_path = os.getenv("ONNX_MODEL_PATH")
        if env_path and os.path.isfile(env_path):
            return env_path

        # Check local common paths
        local_candidates = [
            os.path.join("models", "model_int8.onnx"),
            os.path.join("onnx_models", "model_int8.onnx"),
        ]
        for candidate in local_candidates:
            if os.path.isfile(candidate):
                return candidate

        # Download / resolve from Hugging Face Hub cache
        logger.info(f"Resolving model {onnx_filename} from HuggingFace repo {repo_id}...")
        return hf_hub_download(repo_id=repo_id, filename=onnx_filename)

    def _mean_pooling_and_normalize(
        self,
        last_hidden_state: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Attention-mask-aware mean pooling followed by L2 normalization.
        Reproduces the exact multilingual-e5 pooling specification.
        """
        mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = sum_embeddings / sum_mask

        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled / np.clip(norm, a_min=1e-9, a_max=None)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of document chunks, prepending E5 'passage: ' prefix."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = [f"passage: {t}" for t in texts[i : i + self.batch_size]]
            encodings = self.tokenizer.encode_batch(batch_texts)

            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

            feed = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)

            outputs = self.session.run(None, feed)
            last_hidden_state = outputs[0]

            normalized_batch = self._mean_pooling_and_normalize(last_hidden_state, attention_mask)
            all_embeddings.extend(normalized_batch.tolist())

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embeds a user query, prepending E5 'query: ' prefix."""
        prefixed_text = f"query: {text}"
        encoding = self.tokenizer.encode(prefixed_text)

        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "token_type_ids" in self._input_names:
            feed["token_type_ids"] = np.array([encoding.type_ids], dtype=np.int64)

        outputs = self.session.run(None, feed)
        last_hidden_state = outputs[0]

        normalized = self._mean_pooling_and_normalize(last_hidden_state, attention_mask)
        return normalized[0].tolist()
