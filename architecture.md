# AskTube — System Architecture Specification

## 1. Purpose

This document defines the technical architecture for **AskTube**, a YouTube Video Question Answering application built around an existing Retrieval-Augmented Generation (RAG) backend.

The purpose of this architecture is to transform the existing working RAG pipeline into a usable web application while preserving the correctness, grounding, caching, retrieval, and transcript-handling behavior already implemented.

The architecture must prioritize:

- RAG correctness.
- Grounded answers.
- Timestamp-aware evidence.
- Clear separation of responsibilities.
- Simple maintainable code.
- Good frontend/backend communication.
- Minimal unnecessary infrastructure.
- Easy local development.
- Portfolio-quality engineering.

---

# 2. Core Architectural Principle

AskTube is fundamentally a **RAG application**.

The architecture must therefore be organized around:

```text
YouTube Video
      ↓
Transcript
      ↓
Transcript Processing
      ↓
Chunking
      ↓
Embeddings
      ↓
FAISS Index
      ↓
Retriever
      ↓
Relevant Transcript Context
      ↓
Prompt
      ↓
LLM
      ↓
Grounded Answer
      ↓
Retrieved Evidence + Timestamp
      ↓
User Interface
```

The frontend is not the center of the architecture.

The RAG pipeline is the core.

The application/API layer exists to expose the RAG functionality safely and cleanly to the UI.

---

# 3. High-Level System

```text
                         ASK TUBE
                            │
             ┌──────────────┴──────────────┐
             │                             │
          FRONTEND                     BACKEND
             │                             │
             │                    ┌────────┴────────┐
             │                    │                 │
             │              Application/API     RAG Core
             │                    │                 │
             │                    │        ┌────────┴────────┐
             │                    │        │                 │
             │                    │   Transcript        Retrieval
             │                    │        │                 │
             │                    │   Chunking          FAISS
             │                    │        │                 │
             │                    │   Embeddings         LLM
             │                    │        │                 │
             │                    │   Cache              Prompt
             │                    │
             └──────────── HTTP ──┘
```

---

# 4. Architectural Layers

AskTube should use four logical layers.

```text
┌──────────────────────────────────────┐
│              FRONTEND                │
│      UI / Player / Conversation      │
└──────────────────┬───────────────────┘
                   │
                   │ HTTP / JSON
                   ▼
┌──────────────────────────────────────┐
│         APPLICATION / API            │
│ Session / Validation / DTOs / Errors │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│              RAG CORE                │
│ Transcript → Chunk → Embed → Retrieve│
│ → Prompt → LLM → Answer              │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│        EXTERNAL / LOCAL SYSTEMS      │
│ YouTube Transcript API / HF / FAISS  │
│ Local Cache                           │
└──────────────────────────────────────┘
```

Each layer has a specific responsibility.

---

# 5. Layer 1 — Frontend

The frontend is responsible for presentation and user interaction.

It must NOT implement RAG logic.

The frontend is responsible for:

- YouTube URL input.
- Video player.
- Processing states.
- Conversation UI.
- Question submission.
- Answer rendering.
- Retrieved transcript rendering.
- Timestamp interactions.
- Error states.
- Responsive design.
- Accessibility.

The frontend must not directly interact with:

- FAISS.
- Hugging Face.
- YouTube Transcript API.
- LangChain internals.
- Embedding models.
- Local cache files.

All such functionality must be accessed through the backend/application layer.

---

# 6. Layer 2 — Application/API Layer

The application layer is the bridge between the UI and the existing RAG system.

Its responsibility is to:

- Accept frontend requests.
- Validate input.
- Manage application-level state.
- Call the appropriate RAG functionality.
- Convert internal objects into structured response objects.
- Return consistent JSON responses.
- Handle expected application errors.
- Prevent frontend coupling to implementation details.

This layer should remain thin.

It must NOT duplicate the RAG logic.

---

# 7. Layer 3 — RAG Core

The existing RAG implementation is the most important part of the backend.

The current logic includes:

- YouTube URL parsing.
- Video ID extraction.
- Transcript retrieval.
- Transcript language fallback.
- Transcript caching.
- Timestamp preservation.
- Chunking.
- Multilingual E5 embeddings.
- E5 query/passage prefixes.
- FAISS indexing.
- FAISS cache validation.
- Similarity threshold retrieval.
- Prompt construction.
- Transcript prompt-injection protection.
- No-context deterministic fallback.
- Programmatic source generation.
- Retry handling.
- Lazy model initialization.
- CLI support.

This functionality should be preserved.

The application layer should call this functionality rather than recreate it.

---

# 8. Layer 4 — External and Local Dependencies

AskTube currently relies on:

### External

- YouTube Transcript API.
- Hugging Face inference/model access.

### Local

- FAISS.
- Transcript cache.
- FAISS index cache.
- Application logs.

No additional infrastructure should be introduced unless explicitly required.

Do NOT introduce:

- Redis.
- PostgreSQL.
- MongoDB.
- Pinecone.
- Weaviate.
- Elasticsearch.
- Kubernetes.
- Docker orchestration.
- Microservices.
- Message queues.

The current application does not require them.

---

# 9. Recommended Project Structure

The final project should evolve toward:

```text
AskTube/
│
├── .agents/
│   ├── rules/
│   │   └── workspace.md
│   │
│   └── skills/
│       └── rag-engineering/
│           └── SKILL.md
│
├── docs/
│   ├── PRD.md
│   ├── design.md
│   ├── architecture.md
│   ├── api-contract.md
│   ├── technical-decisions.md
│   ├── evaluation-plan.md
│   └── roadmap.md
│
├── backend/
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   ├── application/
│   │   └── rag_service.py
│   │
│   └── rag/
│       ├── transcript.py
│       ├── chunking.py
│       ├── embeddings.py
│       ├── retrieval.py
│       ├── generation.py
│       ├── cache.py
│       └── pipeline.py
│
├── frontend/
│   └── ...
│
├── tests/
│   ├── test_main.py
│   ├── test_api.py
│   └── ...
│
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

This is a target structure.

Do not blindly create every file immediately.

First identify which existing functions belong to each responsibility.

---

# 10. Existing `main.py` Migration Strategy

The current `main.py` should NOT be rewritten from scratch.

The migration should happen incrementally.

## Step 1

Identify existing functionality:

```text
Configuration
Validation
Transcript retrieval
Caching
Chunking
Embeddings
FAISS
Retrieval
Prompt
LLM
Answer parsing
Source formatting
CLI
```

## Step 2

Group functionality by responsibility.

## Step 3

Move functions into modules without changing behavior.

## Step 4

Run the existing tests.

## Step 5

Only after behavior remains stable, introduce the application/API layer.

---

# 11. RAG Core Boundary

The application layer should conceptually interact with the RAG system through operations similar to:

```text
process_video(url, language)
        ↓
VideoProcessingResult

ask_question(video_session, question)
        ↓
AnswerResult
```

The exact function names may differ.

The important principle is that the application layer should not need to know:

```text
How FAISS works
How embeddings are generated
How E5 prefixes work
How prompts are constructed
How cache files are named
How LangChain Documents are represented
```

Those are RAG-core concerns.

---

# 12. Video Processing Flow

When the user submits a YouTube URL:

```text
Frontend
   │
   │ POST video URL
   ▼
API Layer
   │
   │ validate request
   ▼
Application Service
   │
   ▼
URL / Video ID Validation
   │
   ▼
Transcript Retrieval
   │
   ├── Cached transcript exists
   │       ↓
   │     Load cache
   │
   └── Otherwise
           ↓
       Fetch transcript
   │
   ▼
Timestamp-aware Chunking
   │
   ▼
Embedding / FAISS preparation
   │
   ├── Valid cached index
   │       ↓
   │     Load index
   │
   └── Otherwise
           ↓
       Build index
   │
   ▼
Video Ready
   │
   ▼
Structured API Response
   │
   ▼
Frontend
```

---

# 13. Question Answering Flow

When the user asks:

```text
"What is RAG?"
```

the flow is:

```text
Frontend
   │
   │ question + session/video ID
   ▼
API Layer
   │
   │ validate question
   ▼
Application Service
   │
   ▼
Retriever
   │
   ▼
Similarity Search
   │
   ├── Relevant context found
   │          ↓
   │      Prompt construction
   │          ↓
   │         LLM
   │          ↓
   │      Answer parsing
   │
   └── No relevant context
              ↓
       Deterministic fallback
```

After generation:

```text
Answer
   +
Retrieved Transcript Sections
   +
Timestamps
   ↓
Structured AnswerResult
   ↓
Frontend
```

---

# 14. No-Context Behavior

This behavior is critical.

If retrieval produces no relevant transcript sections:

```text
Retriever
    ↓
No relevant documents
    ↓
DO NOT call LLM
    ↓
Return deterministic fallback
```

The application must preserve the existing fallback:

```text
I couldn't find enough information in the transcript.
```

The frontend may style this response differently, but must not change its meaning.

This prevents unnecessary hallucination and unnecessary LLM calls.

---

# 15. Grounding Boundary

The RAG system must remain transcript-grounded.

The answer generation pipeline should remain conceptually:

```text
User Question
      +
Retrieved Transcript Context
      ↓
Prompt
      ↓
LLM
```

The LLM must not receive arbitrary external information.

Do not add web search to compensate for missing transcript information.

If information is absent:

```text
No information
      ↓
"I couldn't find enough information in the transcript."
```

---

# 16. Prompt Injection Protection

Transcript content must always be treated as untrusted data.

The current prompt architecture establishes a clear distinction:

```text
User Question = instruction
Transcript = data/context
```

This must be preserved.

Do not allow transcript text to override system/application instructions.

Do not turn transcript content into executable instructions.

Do not remove transcript sanitization.

---

# 17. Timestamp Architecture

Timestamp support is a core product feature.

During transcript processing, each chunk should retain:

```text
chunk_id
start_seconds
end_seconds
timestamp
text
```

The application layer must expose these values to the frontend.

Example:

```text
Retrieved Section
      │
      ├── text
      ├── timestamp = "02:14"
      ├── start_seconds = 134.2
      └── end_seconds = 158.7
```

The frontend uses:

```text
start_seconds
```

to seek the YouTube player.

The backend should not require the frontend to parse:

```text
"02:14"
```

into seconds.

The frontend receives machine-readable seconds directly.

---

# 18. Source Architecture

The frontend must not parse the backend's CLI source string.

The current CLI presentation such as:

```text
Retrieved Transcript Sections:
[02:14] ...
```

is a presentation concern.

For the web application, the application layer must return structured sections.

Example:

```json
{
  "chunk_id": 4,
  "timestamp": "02:14",
  "start_seconds": 134.2,
  "end_seconds": 158.7,
  "text": "Relevant transcript content..."
}
```

The frontend decides how this information is visually presented.

---

# 19. Session Model

AskTube needs a lightweight application-level video session.

Conceptually:

```text
Session
│
├── video_id
├── transcript language
├── processing status
├── retriever/index reference
└── conversation
```

A session should ensure that:

```text
Question → Correct Video → Correct Retriever
```

The application must never accidentally use the retriever from a previous video for a new video.

---

# 20. Conversation Model

For the initial version, questions can be treated independently.

Example:

```text
Question 1
    ↓
Retriever
    ↓
Answer 1

Question 2
    ↓
Retriever
    ↓
Answer 2
```

The RAG core does not need conversational memory initially.

Do NOT add:

- Long-term memory.
- Vectorized chat history.
- Complex conversational agents.

If later testing shows that follow-up questions need context rewriting, add a small query-rewriting layer deliberately.

---

# 21. API Responsibilities

The API should provide only the operations needed by the frontend.

Conceptually:

```text
POST /api/videos/process
POST /api/videos/{video_id}/questions
GET  /api/videos/{video_id}
```

Additional endpoints may be introduced only when justified.

---

# 22. Video Processing API

Conceptual request:

```json
{
  "url": "https://www.youtube.com/watch?v=..."
}
```

Optional language configuration may be supported.

Conceptual response:

```json
{
  "video_id": "abc123",
  "title": "Example Video",
  "transcript_language": "en",
  "is_generated": true,
  "status": "ready"
}
```

For processing:

```json
{
  "status": "processing",
  "stage": "transcript"
}
```

Possible stages:

```text
validating
transcript
indexing
ready
error
```

The exact API implementation may differ.

---

# 23. Question API

Conceptual request:

```json
{
  "question": "What is RAG?"
}
```

Conceptual success response:

```json
{
  "status": "success",
  "answer": {
    "summary": "RAG combines retrieval with generation...",
    "key_points": [
      "Relevant information is retrieved.",
      "The retrieved context is given to the model.",
      "The model generates an answer using that context."
    ]
  },
  "has_context": true,
  "sources": [
    {
      "chunk_id": 4,
      "timestamp": "02:14",
      "start_seconds": 134.2,
      "end_seconds": 158.7,
      "text": "Relevant transcript content..."
    }
  ]
}
```

---

# 24. No-Context API Response

When no relevant information exists:

```json
{
  "status": "success",
  "has_context": false,
  "answer": {
    "summary": "I couldn't find enough information in the transcript.",
    "key_points": []
  },
  "sources": []
}
```

The exact schema may be adjusted during implementation.

The semantic behavior must remain unchanged.

---

# 25. Error Architecture

Errors should be categorized.

## User Input Errors

Examples:

```text
Invalid YouTube URL
Invalid question
Question too long
```

These should return clear user-facing messages.

---

## Transcript Errors

Examples:

```text
Transcript unavailable
Video unavailable
Requested language unavailable
```

The frontend should receive a meaningful error state.

---

## External Service Errors

Examples:

```text
Hugging Face unavailable
Rate limit
Temporary YouTube/API issue
```

The backend should preserve retry behavior where appropriate.

The frontend should receive a generic user-friendly message rather than internal stack traces.

---

## Unexpected Errors

Unexpected exceptions should:

1. Be logged.
2. Return a safe generic error.
3. Never expose secrets, stack traces, or internal filesystem paths.

---

# 26. Configuration

The current configuration remains centralized.

Important values include:

```text
EMBEDDING_MODEL
LLM_REPO_ID
RETRIEVER_K
RELEVANCE_SCORE_THRESHOLD
CHUNK_SIZE
CHUNK_OVERLAP
MAX_QUESTION_LENGTH
MAX_RETRIES
CACHE_SCHEMA_VERSION
```

These should not be duplicated across frontend and backend.

The frontend should not know about:

```text
RETRIEVER_K
RELEVANCE_SCORE_THRESHOLD
CHUNK_SIZE
CHUNK_OVERLAP
```

unless a deliberate developer/debug interface is later introduced.

---

# 27. Caching Architecture

The current cache strategy should remain.

There are two important cached resources:

```text
Transcript Cache
       +
FAISS Index Cache
```

FAISS cache validity depends on configuration such as:

```text
video_id
requested language
embedding model
chunk size
chunk overlap
schema version
normalization strategy
distance strategy
```

The application layer should simply request the RAG pipeline to prepare the video.

It should not manually manipulate cache files.

---

# 28. Model Loading

The existing lazy-loading strategy should remain.

The application should not initialize large models unnecessarily.

Conceptually:

```text
Application starts
      ↓
No LLM loaded
No embedding model loaded
      ↓
User processes video
      ↓
Embedding model initialized when required
      ↓
User asks question
      ↓
LLM initialized when required
```

This improves startup behavior and avoids unnecessary resource usage.

---

# 29. Security Boundaries

The architecture must protect:

### API

- Validate input.
- Limit question length.
- Avoid arbitrary file paths from users.
- Avoid exposing internal errors.

### RAG

- Treat transcript as untrusted.
- Preserve prompt-injection defense.
- Do not allow external knowledge to enter the grounded pipeline unintentionally.

### Cache

- Only load trusted locally generated FAISS caches.
- Preserve existing safe deserialization assumptions.

### Secrets

API keys and credentials must remain server-side.

Never expose:

```text
HF_TOKEN
API keys
.env contents
```

to the frontend.

---

# 30. Frontend ↔ Backend Separation

The frontend should know:

```text
Video
Question
Answer
Retrieved Transcript Section
Timestamp
Status
Error
```

The frontend should NOT know:

```text
FAISS
LangChain
Hugging Face
E5 prefixes
Embedding dimensions
Cache schema
Prompt construction
Retriever implementation
```

This separation makes the application easier to change and maintain.

---

# 31. Development Strategy

Implementation should proceed incrementally.

## Stage 1 — Refactor Without Behavior Changes

Move the existing RAG logic into logical modules.

Goal:

```text
Old behavior == New behavior
```

Run existing tests.

---

## Stage 2 — Application Service

Create a thin service around the RAG pipeline.

Example conceptual operations:

```text
process_video()
ask_question()
```

No frontend yet.

---

## Stage 3 — API

Expose the application service through HTTP.

Verify with API tests.

---

## Stage 4 — Frontend

Build the UI according to `design.md`.

Do not change RAG behavior simply to make a UI component easier.

---

## Stage 5 — Integration

Connect:

```text
URL
 ↓
Backend
 ↓
Video
 ↓
Question
 ↓
Answer
 ↓
Sources
 ↓
Timestamp
 ↓
Player
```

---

## Stage 6 — Evaluation

Run the RAG evaluation dataset.

Measure:

- Retrieval quality.
- Groundedness.
- No-context behavior.
- Timestamp relevance.
- Multilingual behavior.

Only then consider retrieval optimization.

---

# 32. Testing Strategy

Testing must exist at multiple levels.

## Unit Tests

Test:

- URL parsing.
- Validation.
- Timestamp formatting.
- Transcript processing.
- Chunk metadata.
- Configuration.
- Error classification.
- Source transformation.

---

## RAG Tests

Test:

- Retrieval.
- Threshold behavior.
- No-context fallback.
- Multilingual retrieval.
- Prompt construction.
- Source preservation.

---

## API Tests

Test:

- Valid video processing.
- Invalid URL.
- Question submission.
- Empty question.
- Excessively long question.
- No-context answer.
- Expected errors.

---

## Frontend Tests

Test:

- URL submission.
- Loading states.
- Answer rendering.
- Source expansion.
- Timestamp click.
- New Video flow.
- Responsive behavior.

---

# 33. Evaluation Before Optimization

Do not change:

```text
CHUNK_SIZE
CHUNK_OVERLAP
RETRIEVER_K
RELEVANCE_SCORE_THRESHOLD
```

based only on intuition.

The evaluation process should compare configurations empirically.

Example:

```text
Configuration A
k = 4
threshold = 0.5

        ↓

Evaluate

        ↓

Configuration B
k = 6
threshold = 0.45

        ↓

Compare results
```

Use metrics defined in `evaluation-plan.md`.

---

# 34. Future Retrieval Improvements

These are intentionally NOT part of the initial implementation.

Possible future improvements:

- Query rewriting.
- MMR retrieval.
- Adjacent chunk expansion.
- Transcript fingerprinting.
- Better timestamp links.
- Summary-specific retrieval.
- Improved multilingual retrieval.
- Reranking.

These should only be introduced when evaluation demonstrates a real problem.

---

# 35. Explicitly Out of Scope

The following are NOT required for AskTube v1:

```text
Microservices
Kubernetes
Docker orchestration
Cloud vector databases
Multiple databases
Redis
Kafka
Agent frameworks
Multi-agent systems
LangGraph
Autonomous web search
Long-term memory
Complex authentication
Enterprise RBAC
Payment systems
Real-time collaboration
Distributed inference
```

Adding these technologies does not automatically improve the project.

AskTube should demonstrate **strong RAG engineering**, not infrastructure complexity.

---

# 36. Architecture Quality Goals

The final architecture should demonstrate:

### RAG Engineering

```text
Strong retrieval
Good chunking
Correct embeddings
Grounded generation
No-context handling
Evaluation
```

### Software Engineering

```text
Clear modules
API boundaries
Validation
Error handling
Testing
Configuration management
```

### Product Engineering

```text
Excellent UX
Fast feedback
Clear evidence
Timestamp navigation
Responsive design
Accessibility
```

The project should communicate:

> "I understand how to build a reliable RAG application."

Not:

> "I added as many technologies as possible."

---

# 37. Final Architecture

The intended final system is:

```text
                         ┌───────────────────┐
                         │      USER         │
                         └─────────┬─────────┘
                                   │
                                   ▼
                         ┌───────────────────┐
                         │    ASK TUBE UI    │
                         │                   │
                         │ Video + Chat      │
                         │ Sources + Player  │
                         └─────────┬─────────┘
                                   │
                              HTTP / JSON
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │     APPLICATION / API    │
                    │                          │
                    │ Validation               │
                    │ Session                  │
                    │ DTO Transformation       │
                    │ Error Handling            │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │        RAG CORE          │
                    │                          │
                    │ URL Validation            │
                    │ Transcript Retrieval      │
                    │ Transcript Cache           │
                    │ Timestamp Chunking        │
                    │ E5 Embeddings             │
                    │ FAISS Retrieval           │
                    │ Relevance Threshold       │
                    │ Grounded Prompt           │
                    │ Qwen3-8B                  │
                    │ Source Extraction         │
                    └───────┬───────────┬──────┘
                            │           │
                  ┌─────────┘           └──────────┐
                  ▼                                ▼
        ┌──────────────────┐              ┌──────────────────┐
        │ YouTube Transcript│              │ Hugging Face     │
        │ API               │              │ Models           │
        └──────────────────┘              └──────────────────┘

                         Local Storage
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
             Transcript Cache     FAISS Index Cache
```

The architectural principle is:

```text
                     KEEP THE RAG CORE STRONG
                                +
                 ADD A THIN APPLICATION LAYER
                                +
                    BUILD A POLISHED UI
                                =
                         ASK TUBE
```

Any implementation decision that conflicts with this principle should be reviewed before being implemented.