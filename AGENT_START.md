# AskTube — Agent Start Instructions

Before changing code, read:
1. `PRD.md`
2. `architecture.md`
3. `api-contract.md`
4. `technical-decisions.md`
5. `design.md`
6. `evaluation-plan.md`
7. `tasks.md`
8. `.agents/rules/workspace.md`
9. `.agents/skills/rag-engineering/SKILL.md`

## Critical Rule
The existing RAG implementation is the core asset. Treat `main.py`, `test_main.py`, and `evaluate_rag.py` as the baseline.

Do not rewrite or simplify the RAG core merely to fit a new architecture.

Before a substantial implementation:
- inspect existing code;
- identify affected files;
- create a plan;
- state API/contract impact;
- identify tests;
- then implement incrementally.

Build a thin application/API layer and frontend around the RAG core.

Do not add Kubernetes, microservices, Redis, PostgreSQL, complex auth, multi-agent systems, web search, external knowledge retrieval, or unnecessary infrastructure.

Use `design.md` as the product/UI specification. The three design files are visual references routed by `design.md`.

The frontend must consume structured application responses, not terminal output, logs, LangChain Documents, or FAISS internals.

`evaluate_rag.py` must remain the retrieval evaluation tool. `test_main.py` is not a substitute for it.

For every meaningful change:
1. explain why;
2. make the smallest appropriate change;
3. run relevant tests;
4. check documentation/contracts;
5. avoid unrelated refactoring.

Prefer a simple, understandable solution over technically fashionable infrastructure.
