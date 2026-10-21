import os
import unittest
from unittest.mock import patch, MagicMock

import streamlit_app
from backend.schemas import (
    ProcessVideoResponse,
    QuestionResponse,
    RetrievedSection,
    ErrorDetail,
)


class TestStreamlitAppHelpers(unittest.TestCase):
    def test_extract_video_id_standard(self):
        """Verify URL parsing handles standard YouTube URLs."""
        self.assertEqual(
            streamlit_app.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            "dQw4w9WgXcQ",
        )
        self.assertEqual(
            streamlit_app.extract_video_id("https://youtu.be/dQw4w9WgXcQ"),
            "dQw4w9WgXcQ",
        )
        self.assertEqual(
            streamlit_app.extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ"),
            "dQw4w9WgXcQ",
        )
        self.assertEqual(
            streamlit_app.extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ"),
            "dQw4w9WgXcQ",
        )
        self.assertEqual(
            streamlit_app.extract_video_id("https://m.youtube.com/watch?v=dQw4w9WgXcQ"),
            "dQw4w9WgXcQ",
        )

    def test_extract_video_id_invalid(self):
        """Verify invalid URLs raise VideoIDError."""
        with self.assertRaises(streamlit_app.VideoIDError):
            streamlit_app.extract_video_id("https://invalid-url.com")
        with self.assertRaises(streamlit_app.VideoIDError):
            streamlit_app.extract_video_id("")
        with self.assertRaises(streamlit_app.VideoIDError):
            streamlit_app.extract_video_id("   ")
        with self.assertRaises(streamlit_app.VideoIDError):
            streamlit_app.extract_video_id("https://youtube.com/watch?v=short")

    def test_format_seconds(self):
        """Verify timestamp formatting produces [MM:SS]."""
        self.assertEqual(streamlit_app.format_seconds(0), "00:00")
        self.assertEqual(streamlit_app.format_seconds(5), "00:05")
        self.assertEqual(streamlit_app.format_seconds(65), "01:05")
        self.assertEqual(streamlit_app.format_seconds(600), "10:00")

    def test_get_youtube_timestamp_url(self):
        """Verify timestamp link generator formats valid YouTube jump links."""
        url = streamlit_app.get_youtube_timestamp_url("dQw4w9WgXcQ", 125.7)
        self.assertEqual(url, "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=125s")

    def test_resolve_hf_token_from_env(self):
        """Verify HF token resolution from environment."""
        with patch.dict(os.environ, {"HUGGINGFACEHUB_API_TOKEN": "hf_test_env_token"}):
            self.assertEqual(streamlit_app.resolve_hf_token(), "hf_test_env_token")

    def test_resolve_hf_token_from_secrets(self):
        """Verify HF token resolution fallback to Streamlit secrets."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(streamlit_app.st, "secrets", {"HUGGINGFACEHUB_API_TOKEN": "hf_test_secret_token"}, create=True):
                self.assertEqual(streamlit_app.resolve_hf_token(), "hf_test_secret_token")

    def test_resolve_hf_token_missing(self):
        """Verify missing token returns None cleanly."""
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(streamlit_app.st, "secrets", {}, create=True):
                self.assertIsNone(streamlit_app.resolve_hf_token())


class TestStreamlitDispatch(unittest.TestCase):
    @patch("streamlit_app.get_local_rag_service")
    def test_process_video_dispatch(self, mock_get_svc):
        """Verify process_video_dispatch calls rag_service with correct args."""
        mock_svc = MagicMock()
        expected_resp = ProcessVideoResponse(
            video_id="dQw4w9WgXcQ",
            title="Rick Astley",
            transcript_language="en",
            is_generated=False,
            status="ready",
            cached=True,
        )
        mock_svc.process_video.return_value = expected_resp
        mock_get_svc.return_value = mock_svc

        res = streamlit_app.process_video_dispatch(
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            preferred_language="en",
            force_refresh=True,
        )

        mock_svc.process_video.assert_called_once_with(
            url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            preferred_language="en",
            force_refresh=True,
        )
        self.assertEqual(res.video_id, "dQw4w9WgXcQ")
        self.assertEqual(res.status, "ready")

    @patch("streamlit_app.get_local_rag_service")
    def test_ask_question_dispatch(self, mock_get_svc):
        """Verify ask_question_dispatch calls rag_service with correct args."""
        mock_svc = MagicMock()
        expected_resp = QuestionResponse(
            status="success",
            answer="This is a grounded answer.",
            summary="Grounded answer.",
            key_points=["Point 1"],
            retrieved_sections=[
                RetrievedSection(
                    chunk_id=0,
                    start_seconds=10.0,
                    end_seconds=25.0,
                    timestamp="00:10",
                    text="Transcript snippet.",
                    relevance_score=0.85,
                )
            ],
        )
        mock_svc.ask_question.return_value = expected_resp
        mock_get_svc.return_value = mock_svc

        res = streamlit_app.ask_question_dispatch(
            video_id="dQw4w9WgXcQ",
            question="What is the video about?",
        )

        mock_svc.ask_question.assert_called_once_with(
            video_id="dQw4w9WgXcQ",
            question="What is the video about?",
        )
        self.assertEqual(res.status, "success")
        self.assertEqual(res.answer, "This is a grounded answer.")
        self.assertEqual(len(res.retrieved_sections), 1)


if __name__ == "__main__":
    unittest.main()
