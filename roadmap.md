# AskTube — Implementation Roadmap

## Phase 0 — Freeze RAG Core
Keep `main.py`, `test_main.py`, and `evaluate_rag.py` as the baseline.

## Phase 1 — Specifications
Read `PRD.md`, `architecture.md`, `api-contract.md`, `technical-decisions.md`, `design.md`, `evaluation-plan.md`, `tasks.md`, and `AGENT_START.md`.

## Phase 2 — Application Layer
Build a thin API/application layer around the existing RAG functionality:
- validation
- processing lifecycle
- structured responses
- error mapping
- video/session state

Do not move RAG logic into the frontend.

## Phase 3 — Frontend
Implement the design in this order:
1. URL input
2. processing states
3. video context
4. question input
5. answers
6. retrieved sections
7. timestamp seeking
8. new/change video
9. responsive layout
10. accessibility/polish

## Phase 4 — Integration
Connect frontend to `api-contract.md` and test ready, no-context, transcript-error, processing-error, validation, multiple-question, and timestamp flows.

## Phase 5 — Testing
Run unit/logic tests, retrieval evaluation, API/application tests, and frontend checks.

## Phase 6 — RAG Calibration
Only after the application works, evaluate chunking, k, threshold, and multilingual retrieval.

## Phase 7 — Polish
Improve loading states, empty states, accessibility, timestamp interaction, visual consistency, documentation, and demo readiness.

## Phase 8 — Final Review
Verify clean setup, passing tests, reproducible evaluation, consistent documentation, and no unnecessary infrastructure.
