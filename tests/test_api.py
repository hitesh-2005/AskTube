import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from backend.app import app
from backend.services.rag_service import rag_service, ActiveSession, _fetch_youtube_title
import main

client = TestClient(app)


class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        # Clear sessions before each test
        rag_service._sessions.clear()

    def test_health_check(self):
        res = client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"status": "ok"})

    def test_process_video_invalid_url(self):
        res = client.post("/api/videos/process", json={"url": "https://invalid-url.com"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "error")
        self.assertIsNotNone(data["error"])
        self.assertEqual(data["error"]["code"], "INVALID_URL")

    def test_get_video_not_ready(self):
        res = client.get("/api/videos/nonexistent_id")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error"]["code"], "VIDEO_NOT_READY")

    def test_ask_question_video_not_ready(self):
        res = client.post(
            "/api/videos/nonexistent_id/questions",
            json={"question": "What is Python?"}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error"]["code"], "VIDEO_NOT_READY")

    def test_ask_question_empty_rejected(self):
        # Setup mock active session
        mock_retriever = MagicMock()
        rag_service._sessions["dummy123456"] = ActiveSession(
            video_id="dummy123456",
            vector_store=MagicMock(),
            retriever=mock_retriever,
            transcript_language="en",
            is_generated=False,
        )

        res = client.post(
            "/api/videos/dummy123456/questions",
            json={"question": "   "}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error"]["code"], "INVALID_QUESTION")

    def test_ask_question_too_long_rejected(self):
        mock_retriever = MagicMock()
        rag_service._sessions["dummy123456"] = ActiveSession(
            video_id="dummy123456",
            vector_store=MagicMock(),
            retriever=mock_retriever,
            transcript_language="en",
            is_generated=False,
        )

        long_question = "A" * 2500
        res = client.post(
            "/api/videos/dummy123456/questions",
            json={"question": long_question}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["error"]["code"], "QUESTION_TOO_LONG")

    def test_ask_question_deterministic_no_context(self):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []  # No docs cleared threshold

        rag_service._sessions["dummy123456"] = ActiveSession(
            video_id="dummy123456",
            vector_store=MagicMock(),
            retriever=mock_retriever,
            transcript_language="en",
            is_generated=False,
        )

        res = client.post(
            "/api/videos/dummy123456/questions",
            json={"question": "What is the capital of France?"}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "no_context")
        self.assertFalse(data["relevant_context_found"])
        self.assertEqual(data["answer"], "I couldn't find enough information in the transcript.")
        self.assertEqual(data["retrieved_sections"], [])

    @patch("backend.services.rag_service.main.get_model")
    def test_ask_question_grounded_success(self, mock_get_model):
        # Mock documents returned by retriever
        doc = Document(
            page_content="TensorFlow was developed by Google Brain.",
            metadata={
                "chunk_id": 1,
                "timestamp": "01:15",
                "start_seconds": 75.0,
                "end_seconds": 95.0,
            },
        )
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [doc]

        # Mock LLM generation output
        mock_llm_chain = MagicMock()
        mock_llm_chain.invoke.return_value = (
            "Summary:\nTensorFlow was created by Google Brain.\n\n"
            "Key Points:\n- Developed by Google Brain\n- Widely used for ML"
        )
        mock_llm_chain.__or__.return_value = mock_llm_chain
        mock_get_model.return_value = mock_llm_chain

        rag_service._sessions["dummy123456"] = ActiveSession(
            video_id="dummy123456",
            vector_store=MagicMock(),
            retriever=mock_retriever,
            transcript_language="en",
            is_generated=False,
        )

        with patch("backend.services.rag_service.main.prompt") as mock_prompt:
            mock_prompt.__or__.return_value = mock_llm_chain
            with patch("backend.services.rag_service.main.parser") as mock_parser:
                res = client.post(
                    "/api/videos/dummy123456/questions",
                    json={"question": "Who created TensorFlow?"}
                )

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(data["relevant_context_found"])
        self.assertEqual(len(data["retrieved_sections"]), 1)
        section = data["retrieved_sections"][0]
        self.assertEqual(section["chunk_id"], 1)
        self.assertEqual(section["timestamp"], "01:15")
        self.assertEqual(section["start_seconds"], 75.0)
        self.assertEqual(section["end_seconds"], 95.0)
        self.assertIn("TensorFlow", section["text"])

    def test_get_video_ready_with_title(self):
        rag_service._sessions["dummy123456"] = ActiveSession(
            video_id="dummy123456",
            vector_store=MagicMock(),
            retriever=MagicMock(),
            transcript_language="en",
            is_generated=False,
            title="My Test Video Title",
        )
        res = client.get("/api/videos/dummy123456")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["video_id"], "dummy123456")
        self.assertEqual(data["title"], "My Test Video Title")

    def test_fetch_youtube_title_fallback_on_error(self):
        with patch("backend.services.rag_service.urllib.request.urlopen", side_effect=Exception("Network error")):
            title = _fetch_youtube_title("x7X9w_GIm1s")
            self.assertIsNone(title)

    def test_fetch_youtube_title_success(self):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"title": "Python in 100 Seconds"}'
        mock_response.__enter__.return_value = mock_response
        with patch("backend.services.rag_service.urllib.request.urlopen", return_value=mock_response):
            title = _fetch_youtube_title("x7X9w_GIm1s")
            self.assertEqual(title, "Python in 100 Seconds")

    def test_process_video_transcript_blocked_clean_response(self):
        blocked_msg = (
            "Couldn't retrieve this video's transcript. YouTube is temporarily blocking transcript "
            "requests from this connection. Please try again later or switch to another network."
        )
        with patch("backend.services.rag_service.main.get_embeddings"):
            with patch("backend.services.rag_service.main.get_transcript_data", side_effect=main.TranscriptBlockedError(blocked_msg)):
                res = client.post(
                    "/api/videos/process",
                    json={"url": "https://www.youtube.com/watch?v=dummy12345a"}
                )

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "error")
        self.assertIsNotNone(data["error"])
        self.assertEqual(data["error"]["code"], "YOUTUBE_TRANSCRIPT_BLOCKED")
        self.assertEqual(data["error"]["message"], blocked_msg)
        self.assertNotIn("github.com", data["error"]["message"])
        self.assertNotIn("traceback", data["error"]["message"].lower())
        # Assert no session was created
        self.assertNotIn("dummy12345a", rag_service._sessions)

    def test_process_video_generic_transcript_error(self):
        with patch("backend.services.rag_service.main.get_embeddings"):
            with patch("backend.services.rag_service.main.get_transcript_data", side_effect=main.TranscriptError("Captions are disabled for this video.")):
                res = client.post(
                    "/api/videos/process",
                    json={"url": "https://www.youtube.com/watch?v=dummy12345a"}
                )

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "error")
        self.assertIsNotNone(data["error"])
        self.assertEqual(data["error"]["code"], "TRANSCRIPT_UNAVAILABLE")
        self.assertEqual(data["error"]["message"], "Captions are disabled for this video.")


if __name__ == "__main__":
    unittest.main()
