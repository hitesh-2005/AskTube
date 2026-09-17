# AskTube — Technical Decisions

These are locked for the initial implementation unless a concrete requirement justifies a change.

| Area | Decision |
|---|---|
| Language | Python |
| RAG framework | LangChain |
| Vector store | FAISS |
| Embeddings | `intfloat/multilingual-e5-base` |
| LLM | `Qwen/Qwen3-8B` |
| Transcript source | YouTube Transcript API |
| Retrieval | cosine-compatible normalized embeddings |
| Initial chunk size | 1000 |
| Initial overlap | 200 |
| Initial k | 4 |
| Initial relevance threshold | 0.5 |
| Authentication | none |
| Database | none |
| Background queue | none |

The current E5 implementation uses `passage:` for documents and `query:` for questions. Do not remove these without evaluation.

Do not change chunking, k, or threshold by intuition alone; use `evaluate_rag.py`.

The application should have a simple frontend → API/application layer → existing RAG core structure.

Do not introduce microservices, Kubernetes, Redis, PostgreSQL, complex auth, agent frameworks, web search, or a full observability stack without an explicit requirement.

Preserve URL validation, transcript fallback/cache, FAISS cache validation, timestamp metadata, prompt-injection defense, deterministic no-context behavior, and programmatic source construction.
