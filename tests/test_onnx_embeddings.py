import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from backend.embeddings import ONNXE5Embeddings


class TestONNXE5Embeddings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Instantiate once for the test suite to avoid re-initializing session repeatedly
        cls.embeddings = ONNXE5Embeddings(batch_size=2)

    def test_01_session_loads_successfully(self):
        """Requirement 8: ONNX model/session loads successfully."""
        self.assertIsNotNone(self.embeddings.session)
        self.assertIsNotNone(self.embeddings.tokenizer)
        self.assertIn("input_ids", self.embeddings._input_names)
        self.assertIn("attention_mask", self.embeddings._input_names)

    def test_02_embed_query_dimension(self):
        """Requirement 1: embed_query returns 384 dimensions."""
        vec = self.embeddings.embed_query("What is dynamic programming?")
        self.assertEqual(len(vec), 384)
        self.assertIsInstance(vec, list)
        self.assertIsInstance(vec[0], float)

    def test_03_embed_documents_count(self):
        """Requirement 2: embed_documents returns one vector per input."""
        texts = [
            "Dynamic programming is an optimization technique.",
            "Python is a versatile programming language.",
            "AskTube generates transcript-grounded answers.",
        ]
        vecs = self.embeddings.embed_documents(texts)
        self.assertEqual(len(vecs), 3)
        for v in vecs:
            self.assertEqual(len(v), 384)

    def test_04_unit_normalized(self):
        """Requirement 3: vectors are approximately unit-normalized."""
        q_vec = np.array(self.embeddings.embed_query("sample query"))
        norm_q = np.linalg.norm(q_vec)
        self.assertAlmostEqual(norm_q, 1.0, places=5)

        d_vecs = np.array(self.embeddings.embed_documents(["doc one", "doc two"]))
        for v in d_vecs:
            norm_d = np.linalg.norm(v)
            self.assertAlmostEqual(norm_d, 1.0, places=5)

    def test_05_query_prefix_applied(self):
        """Requirement 4: query prefix is applied ('query: ')."""
        with patch.object(self.embeddings.tokenizer, "encode", wraps=self.embeddings.tokenizer.encode) as mock_encode:
            _ = self.embeddings.embed_query("test question")
            mock_encode.assert_called_once()
            call_arg = mock_encode.call_args[0][0]
            self.assertEqual(call_arg, "query: test question")

    def test_06_passage_prefix_applied(self):
        """Requirement 5: passage prefix is applied ('passage: ')."""
        with patch.object(self.embeddings.tokenizer, "encode_batch", wraps=self.embeddings.tokenizer.encode_batch) as mock_batch:
            _ = self.embeddings.embed_documents(["chunk alpha", "chunk beta"])
            mock_batch.assert_called_once()
            call_arg = mock_batch.call_args[0][0]
            self.assertEqual(call_arg, ["passage: chunk alpha", "passage: chunk beta"])

    def test_07_batching_works(self):
        """Requirement 6: batching works as expected for documents."""
        # 5 items with batch_size=2 should result in 3 batches: [2, 2, 1]
        texts = [f"Text chunk number {i}" for i in range(5)]
        with patch.object(self.embeddings.tokenizer, "encode_batch", wraps=self.embeddings.tokenizer.encode_batch) as mock_batch:
            vecs = self.embeddings.embed_documents(texts)
            self.assertEqual(len(vecs), 5)
            self.assertEqual(mock_batch.call_count, 3)

    def test_08_empty_input_behavior(self):
        """Requirement 7: empty input behavior matches existing embedding contract."""
        self.assertEqual(self.embeddings.embed_documents([]), [])

    def test_09_output_finite_no_nan_inf(self):
        """Requirement 9: output is finite and contains no NaN/Inf."""
        vec = self.embeddings.embed_query("Check numerical sanity")
        arr = np.array(vec)
        self.assertTrue(np.all(np.isfinite(arr)))
        self.assertFalse(np.any(np.isnan(arr)))
        self.assertFalse(np.any(np.isinf(arr)))


if __name__ == "__main__":
    unittest.main()
