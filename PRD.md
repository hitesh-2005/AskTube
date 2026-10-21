# AskTube — Product Requirements Document

## 1. Product
AskTube is a YouTube Video Q&A application that lets a user provide a YouTube URL, processes the video's transcript, builds a semantic retrieval index, and asks questions grounded only in the transcript.

## 2. Goal
Demonstrate strong practical RAG engineering: reliable transcript ingestion, multilingual retrieval, grounded generation, timestamped context, caching, clear failure states, evaluation, and a polished UX.

## 3. User Flow
1. Paste a YouTube URL.
2. Validate the URL.
3. Retrieve or load the transcript.
4. Chunk the transcript.
5. Build or load the FAISS index.
6. Show processing/readiness state.
7. Ask questions.
8. Retrieve relevant transcript sections.
9. Generate a grounded answer when context exists.
10. Show the answer and retrieved timestamped sections.
11. Ask more questions or change the video.

## 4. Requirements
- Preserve the existing YouTube URL and transcript fallback behavior.
- Preserve transcript and FAISS caching.
- Use multilingual E5 embeddings and FAISS retrieval.
- Use the configured Qwen model for grounded generation.
- Answer only from retrieved transcript context.
- Return the deterministic no-context response when relevant context is absent.
- Expose timestamp metadata through the application layer.
- Support multiple questions for one video.
- Provide clear processing, empty, and error states.
- Provide an intuitive, interactive UI via Streamlit.

## 5. Non-Goals
Do not add Kubernetes, microservices, Redis, PostgreSQL, complex authentication, multi-agent systems, web-search fallback, external knowledge retrieval, unnecessary DevOps, or other infrastructure without an explicit requirement.

## 6. Success Criteria
- Existing RAG behavior remains intact.
- Automated test suite passes (101 tests across core RAG, ONNX embeddings, and Streamlit).
- Streamlit application executes in-process against RAG service.
- Common errors are handled safely.
- The application remains simple, modular, and understandable.

## 7. Definition of Done
The Streamlit application, core RAG pipeline, test suite, setup instructions, and README are complete, consistent, and verified.
