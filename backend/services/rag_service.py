import os
import re
import json
import logging
import urllib.request
from urllib.parse import quote
from typing import Dict, Optional, Any

import main
from backend.config import (
    MAX_QUESTION_LENGTH,
    RETRIEVER_K,
    RELEVANCE_SCORE_THRESHOLD,
)
from backend.schemas import (
    ProcessVideoResponse,
    GetVideoResponse,
    QuestionResponse,
    RetrievedSection,
    ErrorDetail,
)

logger = logging.getLogger("asktube.rag_service")


def _fetch_youtube_title(video_id: str) -> Optional[str]:
    """Fetch real YouTube video title via public oEmbed endpoint.
    Fails safely without breaking video processing."""
    try:
        url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={quote(video_id)}&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "AskTube/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                title = data.get("title")
                if title and isinstance(title, str):
                    return title.strip()
    except Exception as e:
        logger.warning(f"Could not retrieve oEmbed title for {video_id}: {e}")
    return None


class ActiveSession:
    def __init__(
        self,
        video_id: str,
        vector_store: Any,
        retriever: Any,
        transcript_language: str,
        is_generated: Optional[bool],
        title: Optional[str] = None,
    ):
        self.video_id = video_id
        self.vector_store = vector_store
        self.retriever = retriever
        self.transcript_language = transcript_language
        self.is_generated = is_generated
        self.title = title


class RAGService:
    def __init__(self):
        self._sessions: Dict[str, ActiveSession] = {}


    def process_video(
        self,
        url: str,
        preferred_language: Optional[str] = None,
        force_refresh: bool = False,
    ) -> ProcessVideoResponse:
        try:
            video_id = main.extract_video_id(url)
        except main.VideoIDError as e:
            return ProcessVideoResponse(
                status="error",
                error=ErrorDetail(code="INVALID_URL", message=str(e)),
            )

        # Invalidate index if refetching transcript
        rebuild_index = force_refresh

        try:
            embeddings = main.get_embeddings()
            index_path = main._index_cache_path(video_id, preferred_language)
            vector_store = None
            is_cached = False
            transcript_language = preferred_language or "en"
            is_generated = False

            if not rebuild_index and os.path.isdir(index_path):
                meta = main._load_index_metadata(index_path)
                if main._index_metadata_is_valid(meta):
                    try:
                        vector_store = main.FAISS.load_local(
                            index_path,
                            embeddings,
                            allow_dangerous_deserialization=True,
                            distance_strategy=main.DistanceStrategy.COSINE,
                        )
                        transcript_language = meta.get("transcript_language", "en")
                        is_generated = meta.get("is_generated", False)
                        is_cached = True
                        logger.info(f"Loaded cached index for {video_id}")
                    except Exception as e:
                        logger.warning(f"Could not load index cache ({e}); rebuilding.")
                        vector_store = None

            if vector_store is None:
                snippets, transcript_language, is_generated = main.get_transcript_data(
                    video_id, force_refresh=force_refresh, preferred_language=preferred_language
                )
                chunks = main.create_chunks_from_snippets(snippets, video_id, transcript_language)
                vector_store = main.FAISS.from_documents(
                    chunks, embeddings, distance_strategy=main.DistanceStrategy.COSINE
                )
                main._save_index(vector_store, index_path, transcript_language, is_generated)
                is_cached = False

            retriever = main.create_retriever(
                vector_store, k=RETRIEVER_K, threshold=RELEVANCE_SCORE_THRESHOLD
            )

            title = _fetch_youtube_title(video_id) or f"YouTube Video ({video_id})"

            session = ActiveSession(
                video_id=video_id,
                vector_store=vector_store,
                retriever=retriever,
                transcript_language=transcript_language,
                is_generated=is_generated,
                title=title,
            )
            self._sessions[video_id] = session

            return ProcessVideoResponse(
                video_id=video_id,
                title=title,
                transcript_language=transcript_language,
                is_generated=is_generated,
                status="ready",
                cached=is_cached,
            )

        except main.TranscriptBlockedError as e:
            logger.warning(f"YouTube transcript blocked for {video_id}: {e}")
            return ProcessVideoResponse(
                video_id=video_id,
                status="error",
                error=ErrorDetail(code="YOUTUBE_TRANSCRIPT_BLOCKED", message=str(e)),
            )
        except main.TranscriptError as e:
            logger.error(f"Transcript error for {video_id}: {e}")
            return ProcessVideoResponse(
                video_id=video_id,
                status="error",
                error=ErrorDetail(code="TRANSCRIPT_UNAVAILABLE", message=str(e)),
            )
        except Exception as e:
            logger.exception(f"Unexpected processing error for {video_id}: {e}")
            return ProcessVideoResponse(
                video_id=video_id,
                status="error",
                error=ErrorDetail(code="PROCESSING_FAILED", message=main._describe_llm_error(e)),
            )

    def get_video(self, video_id: str) -> GetVideoResponse:
        # Check active sessions
        if video_id in self._sessions:
            s = self._sessions[video_id]
            return GetVideoResponse(
                video_id=video_id,
                title=s.title,
                transcript_language=s.transcript_language,
                is_generated=s.is_generated,
                status="ready",
                cached=True,
            )

        # Check on-disk index
        index_path = main._index_cache_path(video_id, None)
        if os.path.isdir(index_path):
            meta = main._load_index_metadata(index_path)
            if main._index_metadata_is_valid(meta):
                title = _fetch_youtube_title(video_id) or f"YouTube Video ({video_id})"
                return GetVideoResponse(
                    video_id=video_id,
                    title=title,
                    transcript_language=meta.get("transcript_language", "en"),
                    is_generated=meta.get("is_generated", False),
                    status="ready",
                    cached=True,
                )

        return GetVideoResponse(
            video_id=video_id,
            status="error",
            cached=False,
            error=ErrorDetail(code="VIDEO_NOT_READY", message="Video has not been processed yet."),
        )

    def _ensure_session_loaded(self, video_id: str) -> Optional[ActiveSession]:
        if video_id in self._sessions:
            return self._sessions[video_id]

        # Attempt to load from cache
        index_path = main._index_cache_path(video_id, None)
        if os.path.isdir(index_path):
            meta = main._load_index_metadata(index_path)
            if main._index_metadata_is_valid(meta):
                try:
                    embeddings = main.get_embeddings()
                    vector_store = main.FAISS.load_local(
                        index_path,
                        embeddings,
                        allow_dangerous_deserialization=True,
                        distance_strategy=main.DistanceStrategy.COSINE,
                    )
                    retriever = main.create_retriever(
                        vector_store, k=RETRIEVER_K, threshold=RELEVANCE_SCORE_THRESHOLD
                    )
                    title = _fetch_youtube_title(video_id) or f"YouTube Video ({video_id})"
                    session = ActiveSession(
                        video_id=video_id,
                        vector_store=vector_store,
                        retriever=retriever,
                        transcript_language=meta.get("transcript_language", "en"),
                        is_generated=meta.get("is_generated", False),
                        title=title,
                    )
                    self._sessions[video_id] = session
                    return session
                except Exception as e:
                    logger.warning(f"Failed to resurrect session from cache: {e}")
        return None

    def ask_question(self, video_id: str, question: str) -> QuestionResponse:
        session = self._ensure_session_loaded(video_id)
        if session is None:
            return QuestionResponse(
                status="error",
                error=ErrorDetail(
                    code="VIDEO_NOT_READY",
                    message="Video is not loaded. Please process the video first.",
                ),
            )

        q = (question or "").strip()
        if not q:
            return QuestionResponse(
                status="error",
                error=ErrorDetail(code="INVALID_QUESTION", message="Question cannot be empty."),
            )

        if len(q) > MAX_QUESTION_LENGTH:
            return QuestionResponse(
                status="error",
                error=ErrorDetail(
                    code="QUESTION_TOO_LONG",
                    message=f"Question is too long (over {MAX_QUESTION_LENGTH} characters).",
                ),
            )

        try:
            # Bug #5 adherence: retrieve documents using the relevance score threshold
            docs = session.retriever.invoke(q)

            if not docs:
                # Deterministic no-context short-circuit: do NOT call LLM
                return QuestionResponse(
                    answer="I couldn't find enough information in the transcript.",
                    relevant_context_found=False,
                    retrieved_sections=[],
                    status="no_context",
                )

            # Context generation
            context = main.format_documents(docs)
            generation_chain = main.prompt | main.get_model_for_question(q) | main.parser

            raw_answer = main.with_retries(
                generation_chain.invoke,
                {"context": context, "question": q},
                what="generation",
            )

            retrieved_sections = []
            for doc in docs:
                retrieved_sections.append(
                    RetrievedSection(
                        chunk_id=doc.metadata.get("chunk_id", 0),
                        timestamp=doc.metadata.get("timestamp", "??:??"),
                        start_seconds=float(doc.metadata.get("start_seconds", 0.0)),
                        end_seconds=float(doc.metadata.get("end_seconds", 0.0)),
                        text=doc.page_content.strip(),
                    )
                )

            return QuestionResponse(
                answer=raw_answer,
                relevant_context_found=True,
                retrieved_sections=retrieved_sections,
                status="success",
            )

        except Exception as e:
            logger.exception(f"Error answering question for {video_id}: {e}")
            return QuestionResponse(
                status="error",
                error=ErrorDetail(
                    code="INTERNAL_ERROR",
                    message=main._describe_llm_error(e),
                ),
            )


# Global singleton instance of RAGService
rag_service = RAGService()
