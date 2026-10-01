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

