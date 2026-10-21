# AskTube — Technical Decisions

These are locked for the initial implementation unless a concrete requirement justifies a change.

| Area | Decision |
|---|---|
| Language | Python |
| RAG framework | LangChain |
| Vector store | FAISS |
| Embeddings | `intfloat/multilingual-e5-small-onnx-int8` (ONNX Runtime INT8) |
| LLM | `Qwen/Qwen3-8B` |
| Transcript source | YouTube Transcript API |
| Retrieval | cosine-compatible normalized embeddings |
| Initial chunk size | 1000 |
| Initial overlap | 200 |
| Initial k | 4 |
| Initial relevance threshold | 0.52 |
| Authentication | none |
| Database | none |
| Background queue | none |

The current E5 implementation uses `passage:` for documents and `query:` for questions. Do not remove these without evaluation.

Do not change chunking, k, or threshold without empirical validation against a representative evaluation dataset.

The application has a clean Streamlit UI → RAG service layer → RAG core structure.

Do not introduce microservices, Kubernetes, Redis, PostgreSQL, complex auth, agent frameworks, web search, or a full observability stack without an explicit requirement.

Preserve URL validation, transcript fallback/cache, FAISS cache validation, timestamp metadata, prompt-injection defense, deterministic no-context behavior, and programmatic source construction.
