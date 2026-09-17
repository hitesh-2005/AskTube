from typing import Optional, List
from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str


class ProcessVideoRequest(BaseModel):
    url: str
    language: Optional[str] = None
    force_refresh: bool = False


class ProcessVideoResponse(BaseModel):
    video_id: Optional[str] = None
    title: Optional[str] = None
    transcript_language: Optional[str] = None
    is_generated: Optional[bool] = None
    status: str
    cached: bool = False
    error: Optional[ErrorDetail] = None


class GetVideoResponse(BaseModel):
    video_id: str
    title: Optional[str] = None
    transcript_language: Optional[str] = None
    is_generated: Optional[bool] = None
    status: str
    cached: bool = True
    error: Optional[ErrorDetail] = None


class RetrievedSection(BaseModel):
    chunk_id: int
    timestamp: str
    start_seconds: float
    end_seconds: float
    text: str


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: Optional[str] = None
    relevant_context_found: bool = False
    retrieved_sections: List[RetrievedSection] = Field(default_factory=list)
    status: str
    error: Optional[ErrorDetail] = None
