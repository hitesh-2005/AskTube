# AskTube

AskTube is a YouTube Video Q&A application built around transcript-grounded RAG.

## Core Stack
- Python
- LangChain
- FAISS
- `intfloat/multilingual-e5-base`
- `Qwen/Qwen3-8B`
- YouTube Transcript API
- Hugging Face

## Important Files
- `main.py` — current working RAG implementation
- `test_main.py` — fast network-free logic/unit tests
- `evaluate_rag.py` — actual retrieval evaluation/calibration
- `design.md` — product/UI specification
- `architecture.md` — system architecture
- `api-contract.md` — frontend/backend contract
- `technical-decisions.md` — locked initial decisions

## Setup
```bash
python -m venv .venv
pip install -r requirements.txt
```

Create `.env` from `.env.example` and provide the required Hugging Face credential.

## Baseline Checks
```bash
python test_main.py
python evaluate_rag.py
```

## Development Principle
Keep the existing RAG core stable. Build the application/API layer and frontend around it.

AskTube intentionally avoids Kubernetes, microservices, unnecessary databases, complex authentication, multi-agent systems, web-search fallbacks, and unnecessary DevOps.
