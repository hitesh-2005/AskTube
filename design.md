# AskTube — Product & UI Design Specification

## 1. Implementation Instructions

You are building **AskTube**, a YouTube Video Question Answering application powered by Retrieval-Augmented Generation (RAG).

AskTube allows a user to:

1. Paste a YouTube video URL.
2. Process or load the video's transcript.
3. Build or load a cached semantic search index.
4. Ask questions specifically about the video's transcript.
5. Receive answers grounded only in retrieved transcript content.
6. View the transcript sections used to answer the question.
7. Click timestamps to jump directly to the relevant point in the video.

This is a **RAG application**, not an autonomous AI agent.

Do NOT introduce unnecessary agent frameworks, autonomous tool use, multi-agent systems, web search, memory systems, or external knowledge retrieval unless explicitly requested.

The implementation must prioritize:

- Clear user experience.
- High visual polish.
- Fast perceived performance.
- Transcript-grounded answers.
- Strong connection between answers and video timestamps.
- Minimal and understandable UI.
- Responsive design.
- Accessibility.
- Clear separation between frontend, application/API layer, and existing RAG core.

You have access to three supplementary visual design reference files:

1. `@design-mobbin.md`
2. `@design-claude.md`
3. `@design-intercom.md`

Do NOT randomly mix styles or invent unrelated visual systems.

Component styling must follow the routing rules defined in this document.

---

# 2. Product Principles

AskTube should feel like a focused workspace where the user can:

> Watch → Ask → Understand → Verify → Jump to the relevant moment.

The UI should make the relationship between these things immediately clear:

```text
YouTube Video
      ↓
Transcript
      ↓
Retrieved Context
      ↓
AI Answer
      ↓
Relevant Timestamp
```

The application should never feel like a generic chatbot with a YouTube video attached.

The video and transcript are the primary source of truth.

The AI answer is a layer built on top of retrieved transcript evidence.

---

# 3. Global Brand & Visual System

Apply these global overrides to all referenced design systems.

## Brand Accent

Primary accent:

```text
YouTube Red
#FF0000
```

Hover and active accent:

```text
#CC0000
```

Use the accent color intentionally.

Do NOT make the entire interface red.

Red should primarily be used for:

- Primary actions.
- Submit/send buttons.
- Active states.
- Important video-related interactions.
- Selected timestamp indicators.
- Loading/progress accents where appropriate.

---

## Backgrounds

Primary application background:

```text
#FFFFFF
```

Secondary surfaces:

```text
#F9F9F9
```

Subtle tertiary surfaces may use very light neutral grays where necessary.

Avoid dark backgrounds for the main application experience.

The visual tone should feel:

- Clean.
- Editorial.
- Modern.
- Premium.
- Calm.
- Focused.

---

## Borders

All borders must be:

```text
Thin.
Soft.
Minimal.
#E5E5E5
```

Avoid heavy borders around every component.

Use spacing, typography, and subtle surface differences to establish hierarchy.

---

## Border Radius

Use rounded corners consistently.

Recommended hierarchy:

- Small controls: subtle rounding.
- Cards and transcript sections: medium rounding.
- URL input and selected tags: large rounded corners.
- Primary URL input: stadium/pill shape.

Do not excessively round every element.

---

## Shadows

Use shadows sparingly.

Shadows should primarily be used for:

- Floating prompt bar.
- Elevated dropdowns or overlays.
- Important floating controls.

Avoid heavy card shadows throughout the application.

The UI should rely more on whitespace and hierarchy than shadows.

---

# 4. Main Application Architecture

## Desktop Layout

Use a split-screen workspace.

```text
┌───────────────────────────────┬──────────────────────────────────┐
│                               │                                  │
│                               │                                  │
│        VIDEO CONTEXT          │          CONVERSATION            │
│                               │                                  │
│         ~40% WIDTH            │            ~60% WIDTH            │
│                               │                                  │
│   ┌───────────────────────┐   │   User Question                  │
│   │                       │   │                                  │
│   │     VIDEO PLAYER      │   │   AI Answer                      │
│   │                       │   │                                  │
│   └───────────────────────┘   │   Retrieved Context              │
│                               │                                  │
│   Video Information           │                                  │
│                               │                                  │
│   Transcript Context          │                                  │
│                               │                                  │
│                               │                                  │
│                               │──────────────────────────────────│
│                               │ Ask a question...         [Send] │
└───────────────────────────────┴──────────────────────────────────┘
```

### Left Column

Width:

```text
Approximately 40%
```

Behavior:

- Sticky/fixed on desktop where appropriate.
- Contains video context.
- Should remain visible while the conversation scrolls.
- Must not have excessive nested scrolling.

### Right Column

Width:

```text
Approximately 60%
```

Behavior:

- Primary conversational workspace.
- Conversation stream scrolls independently.
- Prompt input remains accessible at all times.

---

# 5. Responsive Layout

## Tablet

The split layout may remain visible if there is sufficient width.

Reduce spacing and adjust proportions intelligently.

Do not simply shrink the desktop layout until it becomes unusable.

---

## Mobile

Stack vertically:

```text
┌─────────────────────────────┐
│         HEADER              │
├─────────────────────────────┤
│                             │
│        VIDEO PLAYER         │
│                             │
├─────────────────────────────┤
│      VIDEO INFORMATION      │
├─────────────────────────────┤
│                             │
│       CONVERSATION          │
│                             │
│       AI ANSWERS            │
│                             │
├─────────────────────────────┤
│ Ask a question...    [Send] │
└─────────────────────────────┘
```

The prompt bar should remain sticky or easily accessible near the bottom.

Timestamp interactions must remain easy to tap.

Avoid horizontal scrolling.

---

# 6. Component Styling Routing Map

## A. Layout Structure

**Reference: `@design-intercom.md`**

Use Intercom-inspired structure for:

- Split workspace.
- Application layout.
- Information hierarchy.
- Secondary panels.
- Collapsible sections.
- Metadata presentation.

The application should feel structured and purposeful rather than like a dashboard full of cards.

---

## B. Top Navigation and Video Panel

**Reference: `@design-mobbin.md`**

### Header

Build a clean sticky header.

Include:

- AskTube logo/wordmark.
- Primary URL input or a compact representation of the active video.
- New Video / Change Video action when a video session is active.

The main URL input must use a polished stadium/pill shape.

Example structure:

```text
[ YouTube URL.................................... ] [Ask]
```

The submit button should use YouTube Red.

---

## C. Video Player

The video player is the visual anchor of the left column.

Requirements:

- Standard 16:9 aspect ratio.
- Clean embedded YouTube player.
- Proper responsive resizing.
- No decorative framing that distracts from the video.

Timestamp interactions elsewhere in the application must be able to seek the player to the relevant time.

The frontend must use the retrieved transcript section's `start_seconds` value for seeking.

---

# 7. Video Context Panel

Below the player, display concise video information.

Recommended structure:

```text
Video

[Video Title]

Transcript
English • Auto-generated

Status
● Ready
```

Possible metadata:

- Video title.
- Transcript language.
- Whether transcript is generated or manually created, if available.
- Processing status.

Do not overload this area with technical information such as:

- FAISS internals.
- Embedding model names.
- Vector dimensions.
- Cache paths.
- LangChain implementation details.

Technical implementation details belong in logs or developer/debug tooling, not the primary user interface.

---

# 8. Video Processing Experience

Processing a video may involve:

```text
URL
 ↓
Transcript Retrieval
 ↓
Transcript Processing
 ↓
Chunking
 ↓
Embedding / Search Index Preparation
 ↓
Ready
```

The UI must communicate progress clearly.

## Processing State

Example:

```text
Preparing your video

✓ Video recognized
✓ Transcript retrieved
✓ Transcript processed
⟳ Preparing search...
```

Do not expose unnecessary technical jargon unless useful.

Prefer:

```text
Preparing search...
```

instead of:

```text
Generating FAISS embeddings...
```

---

## Cached State

If an existing processed video can be loaded:

```text
✓ Your video is ready

Previous processing was loaded successfully.
```

The UI may use a subtle indication of faster loading, but should not expose internal cache paths.

---

## Processing Failure

Clearly explain failures.

Examples:

### Transcript unavailable

```text
We couldn't access a transcript for this video.
```

### Invalid YouTube URL

```text
Please enter a valid YouTube video URL.
```

### Video unavailable

```text
This video is unavailable, private, or has been removed.
```

### Temporary service issue

```text
Something went wrong while preparing this video. Please try again.
```

Provide a clear retry action where appropriate.

---

# 9. Transcript Context

Use the clean Mobbin-inspired gallery/canvas aesthetic.

The transcript context should help users understand the video without becoming visually overwhelming.

Do not display the entire transcript by default if it creates excessive visual density.

Recommended options include:

- A compact transcript preview.
- Scrollable transcript context.
- Expand/collapse interaction.
- Relevant transcript sections highlighted when a source is selected.

Timestamp styling:

```text
[ 02:14 ]
```

Each timestamp should appear as a compact, clickable pill.

### Timestamp Interaction

Clicking a timestamp must:

1. Seek the YouTube player to `start_seconds`.
2. Begin or prepare playback at that point.
3. Optionally highlight the corresponding transcript section.

The timestamp is not merely decorative.

It is an important interaction connecting:

```text
AI Answer → Evidence → Video Moment
```

---

# 10. RAG Conversation Experience

**Reference: `@design-claude.md`**

The right column is the primary conversation workspace.

Prioritize:

- Excellent readability.
- Generous spacing.
- Strong typography.
- Editorial hierarchy.
- Clear separation between user questions and AI answers.

Avoid excessive chatbot decorations.

---

## User Messages

User questions should:

- Be visually distinct.
- Align to the right.
- Use soft neutral gray backgrounds.
- Have comfortable padding.
- Avoid overly large rounded bubbles.

Example:

```text
                        ┌──────────────────────┐
                        │ What is RAG?         │
                        └──────────────────────┘
```

---

## AI Answers

AI answers should:

- Align naturally to the left.
- Sit primarily on the white canvas.
- Use minimal or no visible container border.
- Prioritize readable text over bubble styling.

The AI response should visually support:

```text
Summary

The concise explanation...

Key Points

• Point one
• Point two
• Point three
```

Typography should make summaries and key points immediately scannable.

Use:

- Clear heading hierarchy.
- Comfortable line height.
- Adequate spacing between sections.
- Proper bullet styling.

---

# 11. Answer States

The UI must support explicit answer states.

## Searching

Before an answer is available:

```text
Searching the video...
```

Use a subtle loading treatment.

Avoid distracting animations.

---

## Answer Found

Display:

1. Summary.
2. Key points.
3. Retrieved transcript context.

The source context must be visually secondary to the answer but easy to access.

---

## Information Not Found

The backend may determine that no transcript chunks satisfy the relevance threshold.

In this case, the LLM should not be called.

Display a neutral, helpful state:

```text
I couldn't find enough information in the transcript.
```

Use a soft neutral or subtle amber information card.

Do NOT claim:

```text
"The similarity score was too low."
```

The user should not be exposed to implementation-level retrieval metrics.

The UI should describe the actual user-level outcome:

> No relevant information was found in the available transcript context.

---

# 12. Retrieved Transcript Context

**Reference: `@design-intercom.md` + `@design-mobbin.md`**

Under every grounded AI answer, provide a collapsible section.

Default label:

```text
Retrieved Transcript Context
```

Example:

```text
▸ Retrieved Transcript Context (3)
```

When expanded:

```text
┌─────────────────────────────────────────────┐
│ [02:14]                                    │
│                                             │
│ The speaker explains how retrieval is used  │
│ to provide relevant context to the model... │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ [05:42]                                    │
│                                             │
│ Additional explanation from the transcript  │
│ related to the user's question...           │
└─────────────────────────────────────────────┘
```

Do NOT force transcript sections into dense side-by-side cards.

Transcript content is primarily reading material.

Use vertically stacked sections that remain readable on desktop and mobile.

Each section must support timestamp interaction.

Clicking the timestamp should seek the video to:

```text
start_seconds
```

---

# 13. Prompt Bar

**Reference: `@design-mobbin.md`**

The prompt bar is a key interaction element.

Location:

- Sticky near the bottom of the conversation column.
- Easily accessible while scrolling.

Style:

- Floating or subtly elevated.
- Rounded container.
- Clean input.
- Minimal visual noise.
- Soft shadow.

Structure:

```text
┌───────────────────────────────────────────────┐
│ Ask anything about this video...      [Send] │
└───────────────────────────────────────────────┘
```

The Send button uses:

```text
#FF0000
```

Hover:

```text
#CC0000
```

---

## Prompt Behavior

Support:

- Enter to submit where appropriate.
- Shift + Enter for multiline input if multiline questions are supported.
- Disabled state while an answer is being generated.
- Clear visual feedback during processing.

Do not allow accidental duplicate submissions while a request is already being processed.

---

# 14. Empty States

The application must have intentional empty states.

## Before Video Submission

```text
Ask questions about any YouTube video.

Paste a YouTube URL to get started.
```

The initial screen should feel welcoming rather than empty.

---

## After Video Is Ready but Before the First Question

```text
Your video is ready.

Ask anything about what is discussed in the video.
```

Optional generic example questions may be displayed.

These must remain clearly generic unless they are generated from actual transcript content.

Examples:

```text
• What are the main points discussed?
• Can you explain the key concept?
• What conclusion does the speaker reach?
```

Do NOT invent video-specific suggestions without grounding them in the transcript.

---

# 15. New Video / Change Video Flow

The user must be able to start a new video session without refreshing the application.

Provide a visible but non-distracting:

```text
New Video
```

or:

```text
Change Video
```

action.

Expected behavior:

```text
Current Video
      ↓
Ask Questions
      ↓
New Video
      ↓
Paste New URL
      ↓
Process New Video
      ↓
New Conversation
```

Previous conversation state should not accidentally mix with the new video's context.

A new video should create a clearly separate application session.

---

# 16. Loading and Interaction States

Every primary interaction should have a visible state.

Examples:

### URL submission

```text
Processing video...
```

### Question submission

```text
Searching transcript...
```

### Answer generation

```text
Preparing answer...
```

### Timestamp seeking

Provide subtle feedback if the player is loading or unavailable.

Do not leave users wondering whether their interaction worked.

---

# 17. Accessibility Requirements

The UI must support:

- Keyboard navigation.
- Visible focus states.
- Sufficient color contrast.
- Accessible buttons and inputs.
- Labels for icon-only controls.
- Touch-friendly timestamp pills.
- Responsive text sizing.

Do not rely only on red color to communicate meaning.

For example:

```text
Error = Icon + text + color
```

not just:

```text
Red text
```

---

# 18. Motion and Animation

Animations should be subtle and purposeful.

Appropriate uses:

- Accordion expansion.
- Loading transitions.
- New answer appearance.
- Timestamp selection.
- Video processing progress.

Avoid:

- Excessive bouncing.
- Continuous decorative animation.
- Slow transitions that delay interaction.
- Unnecessary animated backgrounds.

The application should feel fast and calm.

---

# 19. Backend ↔ UI Contract

The frontend must consume structured application-level results.

The frontend must NOT parse:

- CLI print statements.
- Log output.
- FAISS internals.
- LangChain `Document` objects directly.
- A single formatted string containing both the answer and sources.

The existing RAG core should remain logically independent from the UI.

An application/API layer should convert internal objects into UI-friendly data.

---

## Video Processing Result

The application layer should expose conceptually:

```text
video_id
video_title
transcript_language
is_generated
status
error
```

Example conceptual structure:

```json
{
  "video_id": "abc123",
  "video_title": "Example Video",
  "transcript_language": "en",
  "is_generated": true,
  "status": "ready"
}
```

The exact implementation format may vary.

---

## Retrieved Transcript Section

Each retrieved section should expose:

```text
chunk_id
timestamp
start_seconds
end_seconds
text
```

Conceptual example:

```json
{
  "chunk_id": 4,
  "timestamp": "02:14",
  "start_seconds": 134.2,
  "end_seconds": 158.7,
  "text": "Relevant transcript content..."
}
```

`start_seconds` is required for timestamp-to-video navigation.

---

## Answer Result

The application layer should conceptually expose:

```text
answer
sources
has_context
status
error
```

Example:

```json
{
  "answer": {
    "summary": "RAG combines retrieval with generation...",
    "key_points": [
      "Retrieves relevant information.",
      "Provides context to the language model.",
      "Reduces unsupported answers."
    ]
  },
  "sources": [
    {
      "chunk_id": 4,
      "timestamp": "02:14",
      "start_seconds": 134.2,
      "end_seconds": 158.7,
      "text": "..."
    }
  ],
  "has_context": true,
  "status": "success"
}
```

The exact API schema may evolve, but the UI must not depend on parsing presentation strings.

---

# 20. Existing RAG Core Protection

The existing RAG pipeline already contains important logic for:

- Video ID validation.
- Transcript fetching.
- Language preference and fallback.
- Transcript caching.
- Exact timestamp preservation.
- Chunk creation.
- Multilingual embeddings.
- E5 query/passage prefixes.
- FAISS vector search.
- Relevance thresholds.
- Hallucination prevention when no context is retrieved.
- Prompt-injection protection for transcript content.
- Source generation from retrieved documents.
- Retry handling.
- Lazy model initialization.
- Versioned cache validation.

Do NOT arbitrarily rewrite or remove this logic while implementing the UI.

The UI layer must adapt around the RAG core.

If a new visual interaction requires backend data that is not currently exposed:

1. Identify the exact data required.
2. Define the application/API contract.
3. Add only the minimal application-layer functionality necessary.
4. Do not silently redesign the retrieval or generation pipeline.

---

# 21. Strict Implementation Rules

The implementation must NOT:

- Introduce agent frameworks.
- Add autonomous agents.
- Add web search.
- Add external knowledge retrieval.
- Add unnecessary databases.
- Add authentication unless explicitly requested.
- Replace the existing RAG architecture without approval.
- Change the embedding model without approval.
- Remove multilingual support.
- Remove E5 query/passage prefixes.
- Remove caching.
- Remove relevance threshold protection.
- Call the LLM when no relevant transcript context exists.
- Parse CLI-formatted output in the frontend.
- Expose internal FAISS or LangChain implementation details to normal users.
- Invent video-specific facts.
- Generate unsupported answers outside the transcript context.
- Add random UI components or unrelated dashboard features.

Prefer minimal, well-structured additions.

---

# 22. Implementation Priority

Build in this order.

## Phase 1 — Application Foundation

- Project structure.
- Backend/application layer.
- API contracts.
- Preserve existing RAG core.
- Health/error handling.

## Phase 2 — Video Processing

- YouTube URL input.
- Video validation.
- Transcript processing.
- Processing states.
- Video metadata.
- Embedded player.

## Phase 3 — Question Answering

- Conversation interface.
- Question submission.
- Loading states.
- Grounded answers.
- Missing-context state.

## Phase 4 — Source Experience

- Retrieved transcript context.
- Expand/collapse behavior.
- Timestamp pills.
- Video seeking.
- Transcript highlighting where appropriate.

## Phase 5 — Polish

- Empty states.
- Error states.
- Responsive design.
- Accessibility.
- Keyboard behavior.
- Animation refinement.
- Mobile optimization.

---

# 23. Definition of Done

AskTube is complete when a user can:

1. Open the application.
2. Paste a valid YouTube URL.
3. See clear processing progress.
4. View the YouTube video.
5. See relevant video/transcript information.
6. Ask a question about the video.
7. Receive an answer grounded in retrieved transcript content.
8. Receive a clear message when relevant information is unavailable.
9. Expand retrieved transcript context.
10. Click a timestamp.
11. Jump directly to the relevant video moment.
12. Ask multiple questions in the same session.
13. Switch to another video without refreshing the application.
14. Use the application comfortably on desktop and mobile.

The final experience should feel:

> Focused, intelligent, trustworthy, visually polished, and purpose-built for understanding YouTube videos.

The central product experience is:

```text
Watch
  ↓
Ask
  ↓
Understand
  ↓
Verify
  ↓
Jump to the exact moment
```