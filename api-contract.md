# AskTube — API Contract

The frontend consumes structured application-level results. It must never parse CLI output, logs, LangChain Documents, or FAISS internals.

## Process Video
`POST /api/videos/process`

Request:
```json
{"url":"https://www.youtube.com/watch?v=VIDEO_ID","language":"en"}
```

Success:
```json
{
  "video_id":"VIDEO_ID",
  "title":"Video Title Here",
  "transcript_language":"en",
  "is_generated":false,
  "status":"ready",
  "cached":false,
  "error":null
}
```

Possible status values:
`validating`, `fetching_transcript`, `processing_transcript`, `building_index`, `ready`, `error`.

## Get Video
`GET /api/videos/{video_id}`

Response:
```json
{
  "video_id":"VIDEO_ID",
  "title":"Video Title Here",
  "transcript_language":"en",
  "is_generated":false,
  "status":"ready",
  "cached":true,
  "error":null
}
```

Video title metadata is fetched via lightweight oEmbed at the application layer, falling back to null or a default title if unavailable without blocking video processing.

## Ask Question
`POST /api/videos/{video_id}/questions`

Request:
```json
{"question":"What is the main idea explained in the video?"}
```

Success:
```json
{
  "answer":"Answer grounded in the retrieved transcript.",
  "relevant_context_found":true,
  "retrieved_sections":[
    {
      "chunk_id":3,
      "timestamp":"02:14",
      "start_seconds":134.0,
      "end_seconds":166.0,
      "text":"Transcript section..."
    }
  ],
  "status":"success",
  "error":null
}
```

## No Context
```json
{
  "answer":"I couldn't find enough information in the transcript.",
  "relevant_context_found":false,
  "retrieved_sections":[],
  "status":"no_context",
  "error":null
}
```

## Error
```json
{
  "answer":null,
  "relevant_context_found":false,
  "retrieved_sections":[],
  "status":"error",
  "error":{"code":"TRANSCRIPT_UNAVAILABLE","message":"A user-safe explanation."}
}
```

Suggested stable error codes:
`INVALID_URL`, `TRANSCRIPT_UNAVAILABLE`, `YOUTUBE_TRANSCRIPT_BLOCKED`, `PROCESSING_FAILED`, `VIDEO_NOT_READY`, `QUESTION_TOO_LONG`, `INTERNAL_ERROR`.

Retrieved sections contain:
- `chunk_id`
- `timestamp`
- `start_seconds`
- `end_seconds`
- `text`

Contract changes require updating this file and relevant tests.
