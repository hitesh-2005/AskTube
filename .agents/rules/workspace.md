# AskTube Workspace Rules

Protect the existing RAG core: `main.py`, `test_main.py`, and `evaluate_rag.py`.

Use this architecture:
frontend → thin application/API layer → existing RAG core.

Locked stack:
- Python
- LangChain
- FAISS
- multilingual E5
- Qwen3-8B
- YouTube Transcript API
- Hugging Face

Preserve transcript fallback/cache, FAISS cache validation, E5 prefixes, normalized embeddings, cosine retrieval, timestamps, prompt-injection defense, deterministic no-context behavior, and programmatic retrieved-section output.

Do not add Kubernetes, microservices, Redis, PostgreSQL, complex authentication, multi-agent systems, web search, LangGraph, distributed processing, or full observability without explicit justification.

Prefer minimal changes and avoid speculative abstractions.

Before substantial changes, inspect the code, make a plan, implement incrementally, and test.

If an API or locked technical decision changes, update its documentation.
