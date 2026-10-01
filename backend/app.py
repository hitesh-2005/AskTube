import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.schemas import (
    ProcessVideoRequest,
    ProcessVideoResponse,
    GetVideoResponse,
    QuestionRequest,
    QuestionResponse,
)
from backend.services.rag_service import rag_service

logger = logging.getLogger("asktube.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up embedding model on startup
    logger.info("Initializing and warming up embedding model on startup...")
    try:
        import main
        main.get_embeddings()
        logger.info("Embedding model initialized successfully.")
    except Exception as e:
        logger.warning(f"Embedding model warm-up failed or skipped: {e}")
    yield


app = FastAPI(
    title="AskTube API",
    description="YouTube Video Question Answering using RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/diagnostics/transcript-provider")
def diagnostic_transcript_provider():
    """Temporary diagnostic endpoint to validate FreeTranscriptAPI connectivity from the host."""
    import urllib.request
    import urllib.error
    import json

    target_url = "https://api.freetranscriptapi.com/v1/transcript?video_url=https://www.youtube.com/watch?v=x7X9w_GIm1s"
    req = urllib.request.Request(
        target_url,
        headers={"User-Agent": "AskTube-Diagnostic/1.0", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            http_status = resp.status
            raw_body = resp.read().decode("utf-8")
            data = json.loads(raw_body)
            transcript = data.get("transcript", [])
            first_seg = transcript[0] if transcript else None
            first_seg_clean = None
            has_timestamps = False
            if first_seg and isinstance(first_seg, dict):
                has_timestamps = "start" in first_seg and "duration" in first_seg
                first_seg_clean = {
                    "text": first_seg.get("text", ""),
                    "start": first_seg.get("start"),
                    "duration": first_seg.get("duration"),
                }
            return {
                "provider": "FreeTranscriptAPI",
                "status": "success",
                "http_status": http_status,
                "language": data.get("language"),
                "title": data.get("title"),
                "segment_count": len(transcript),
                "has_timestamps": has_timestamps,
                "first_segment": first_seg_clean,
            }
    except urllib.error.HTTPError as e:
        logger.warning(f"Diagnostic FreeTranscriptAPI HTTP error: {e.code}")
        sanitized_msg = f"HTTP {e.code}: {e.reason}"
        try:
            err_data = json.loads(e.read().decode("utf-8"))
            if isinstance(err_data, dict) and "error" in err_data:
                err_info = err_data["error"]
                if isinstance(err_info, dict) and "message" in err_info:
                    sanitized_msg = err_info["message"]
        except Exception:
            pass
        return {
            "provider": "FreeTranscriptAPI",
            "status": "failed",
            "http_status": e.code,
            "error_type": "HTTPError",
            "message": sanitized_msg,
        }
    except urllib.error.URLError as e:
        logger.warning(f"Diagnostic FreeTranscriptAPI connection error: {e.reason}")
        return {
            "provider": "FreeTranscriptAPI",
            "status": "failed",
            "http_status": None,
            "error_type": "URLError",
            "message": f"Connection failed: {e.reason}",
        }
    except Exception as e:
        logger.warning(f"Diagnostic FreeTranscriptAPI unexpected error: {type(e).__name__}")
        return {
            "provider": "FreeTranscriptAPI",
            "status": "failed",
            "http_status": None,
            "error_type": type(e).__name__,
            "message": "An unexpected error occurred during the diagnostic check.",
        }


@app.post("/api/videos/process", response_model=ProcessVideoResponse)
def process_video(request: ProcessVideoRequest):
    result = rag_service.process_video(
        url=request.url,
        preferred_language=request.language,
        force_refresh=request.force_refresh,
    )
    return result


@app.get("/api/videos/{video_id}", response_model=GetVideoResponse)
def get_video(video_id: str):
    result = rag_service.get_video(video_id)
    return result


@app.post("/api/videos/{video_id}/questions", response_model=QuestionResponse)
def ask_question(video_id: str, request: QuestionRequest):
    result = rag_service.ask_question(video_id=video_id, question=request.question)
    return result


# Mount frontend static files if directory exists
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.app:app", host="0.0.0.0", port=port, reload=False)

