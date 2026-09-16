# Podcast-to-Knowledge Platform

## Product & Engineering Specification

**Document version:** 1.0
**Status:** Initial Product Specification
**Purpose:** Source of truth for building the application with Claude Code

---

# 1. Product Vision

Build a modern web platform that transforms long-form podcast conversations into high-quality, beautifully written, easy-to-read knowledge articles.

The core idea is simple:

> **Turn 2 hours of conversation into 15–30 minutes of high-quality reading.**

The platform should help users consume the **ideas, arguments, insights, examples, and important discussions** from long-form podcasts without requiring them to spend the entire podcast duration listening to the episode.

This is NOT intended to be a basic "AI summarizer."

The product should behave more like an:

> **AI editorial engine for long-form conversations.**

The system should understand the entire conversation, identify its important ideas, reorganize those ideas into a coherent narrative, generate a readable article, and verify that the article remains faithful to the original transcript.

---

# 2. Core Product Principle

The system should NOT simply do:

```text
2-hour transcript
        ↓
      LLM
        ↓
   short summary
```

Instead, the system should perform a multi-stage transformation:

```text
YouTube Episode
      ↓
Transcript Acquisition
      ↓
Transcript Cleaning
      ↓
Speaker / Timestamp Preservation
      ↓
Semantic Chunking
      ↓
Chunk-level Analysis
      ↓
Topic & Theme Extraction
      ↓
Conversation Map
      ↓
Article Planning
      ↓
Grounded Article Generation
      ↓
Factuality / Fidelity Verification
      ↓
Editorial Refinement
      ↓
Published Article
```

The architecture must preserve the relationship between the generated article and the original transcript.

---

# 3. Target Users

Primary users:

* People interested in podcasts but who don't have enough time to listen to entire episodes.
* Professionals who want to consume ideas quickly.
* Students and researchers.
* Developers and AI/ML professionals.
* Entrepreneurs.
* People interested in technology, science, business, philosophy, economics, health, history, etc.
* Readers who prefer articles over audio.

Secondary users:

* Podcast creators who want written versions of their episodes.
* Publishers.
* Knowledge workers.
* Researchers.

---

# 4. Example User Problem

A user sees:

```text
Podcast:
2 hours 17 minutes

Topic:
Future of Artificial Intelligence
```

They may not have 2+ hours available.

Our platform should provide:

```text
Original podcast:
2h 17m

Our article:
~23 min read
```

The user should feel:

> "I now understand the important ideas from this conversation without spending two hours listening to it."

However, the system must NOT intentionally distort the conversation merely to achieve a shorter reading time.

Accuracy and preservation of meaning are more important than aggressive compression.

---

# 5. Product Positioning

Do NOT position the product merely as:

> "AI Podcast Summarizer"

Prefer positioning such as:

> **Turn conversations into knowledge.**

or:

> **The world's best conversations, distilled into beautiful reading.**

The product differentiator should be:

1. High-quality editorial writing.
2. Logical organization rather than chronological transcript summarization.
3. Grounding in the original transcript.
4. Timestamp-linked evidence.
5. Multiple reading lengths.
6. Podcast-specific knowledge extraction.
7. Ability to ask questions about the episode.
8. Strong factuality/fidelity checks.

---

# 6. MVP Scope

The first version should be intentionally focused.

## MVP must support

### Admin

An administrator can:

1. Enter a YouTube URL.
2. Fetch podcast metadata.
3. Fetch the transcript.
4. Store the transcript.
5. Process the transcript.
6. Generate an article.
7. Review processing status.
8. Review generated article.
9. Publish/unpublish article.

### Public users

Users can:

1. Browse published podcast articles.
2. Search articles.
3. Open an article.
4. See podcast metadata.
5. See estimated original duration.
6. See estimated reading time.
7. Read the generated article.
8. Jump to relevant podcast timestamps.
9. View key takeaways.
10. Ask questions about the podcast/article.

---

# 7. Do NOT Overbuild V1

Do NOT initially build:

* Native mobile apps.
* Complex recommendation engines.
* Social networking.
* User profiles.
* Comments.
* Creator dashboards.
* Payments.
* Subscriptions.
* Complex analytics.
* Automatic scraping of hundreds of channels.
* Fully autonomous publishing.
* Multiple LLM providers unless abstraction is needed.
* Microservices.
* Kubernetes.
* Complex distributed infrastructure.

Build a clean modular monolith first.

The architecture should make future expansion possible without prematurely introducing complexity.

---

# 8. Important Copyright / Content Policy Consideration

The application must NOT assume that arbitrary third-party podcast transcripts can automatically be republished.

The initial system should be designed around content for which we have appropriate permission or rights.

Possible MVP sources:

* Our own podcast.
* Content explicitly authorized by creators.
* Creator opt-in.
* Appropriately licensed/public-domain content.
* Private/personal transformation workflows.

Do not implement automatic large-scale scraping and publishing of third-party podcast content as a default behavior.

Store source metadata and original YouTube URL.

Every article should clearly attribute the original podcast/channel.

---

# 9. Technology Stack

Use a modular monolith.

## Frontend

* Next.js
* TypeScript
* Tailwind CSS
* Modern responsive UI
* Server/client components where appropriate

## Backend

* Python
* FastAPI

## AI

* LLM provider abstraction
* LangGraph for orchestration
* LangChain only where it materially simplifies implementation
* Structured outputs wherever possible
* Embeddings for semantic retrieval

## Database

* PostgreSQL
* pgvector

## Background processing

* Redis
* Background job system appropriate for Python

Use asynchronous processing for long-running podcast processing.

## Storage

Use object storage abstraction if required for:

* Raw transcript files
* Generated artifacts
* Images

Do not tightly couple business logic to one storage provider.

## Deployment

Docker-first.

The entire application should be runnable locally with Docker Compose.

---

# 10. Architecture Principles

## Principle 1 — Modular Monolith

Use clear modules instead of microservices.

Suggested backend modules:

```text
backend/
    app/
        api/
        core/
        config/
        models/
        schemas/
        repositories/
        services/
        providers/
        workflows/
        prompts/
        evaluation/
        utils/
```

---

# 11. Provider Abstraction

External dependencies must be abstracted.

For example:

```python
class TranscriptProvider:
    async def get_transcript(self, video_url: str):
        raise NotImplementedError
```

Implementation:

```python
class SupadataTranscriptProvider(TranscriptProvider):
    async def get_transcript(self, video_url: str):
        ...
```

Similarly create abstractions where useful:

```python
class LLMProvider:
    ...

class EmbeddingProvider:
    ...

class StorageProvider:
    ...
```

The application should not directly call Supadata or an LLM throughout arbitrary parts of the codebase.

---

# 12. Transcript Acquisition

Use:

> Supadata

as the initial YouTube transcript provider.

Supadata:

https://supadata.ai/

Use its YouTube transcript API to retrieve transcripts.

The implementation must read the API key from an environment variable.

Never hard-code API keys.

Example:

```text
SUPADATA_API_KEY=...
```

The exact API request/response implementation should follow the current Supadata API documentation.

Do not assume the API response schema if documentation differs.

---

# 13. Transcript Provider Responsibilities

The transcript provider is responsible only for acquiring transcript data.

It should return a normalized internal representation.

Example:

```python
class TranscriptSegment:
    text: str
    start_ms: int
    duration_ms: int
    speaker: str | None
```

Normalized transcript:

```python
class Transcript:
    language: str
    segments: list[TranscriptSegment]
```

The rest of the application must work with this internal model.

It must NOT depend directly on Supadata's response structure.

---

# 14. Preserve Timestamps

Timestamp preservation is a critical product requirement.

Never discard timestamps.

Example:

```json
{
  "text": "Scaling laws have changed how we think about AI.",
  "start_ms": 253000,
  "duration_ms": 6200
}
```

The article should eventually be able to reference this content.

For example:

```text
🎧 Listen to this section — 42:13
```

Clicking should open the original YouTube video at approximately the corresponding timestamp.

---

# 15. Podcast Metadata

Store metadata such as:

```text
video_id
youtube_url
title
description
channel_name
channel_id
thumbnail_url
duration_seconds
published_at
language
```

Also store:

```text
ingested_at
processed_at
published_at
processing_status
```

---

# 16. Database Design

Use PostgreSQL.

Use pgvector for embeddings.

Suggested entities:

```text
Podcast
Episode
Transcript
TranscriptSegment
Chunk
ChunkAnalysis
Topic
EpisodeTopic
Article
ArticleSection
ArticleCitation
ProcessingJob
EvaluationResult
```

A possible relationship:

```text
Podcast
   │
   └── Episode
          │
          ├── Transcript
          │       └── TranscriptSegment
          │
          ├── Chunk
          │       └── ChunkAnalysis
          │
          ├── Topic
          │
          └── Article
                  ├── ArticleSection
                  └── ArticleCitation
```

---

# 17. Processing Status

Episodes should have explicit processing states.

Example:

```text
INGESTING
TRANSCRIPT_FETCHED
CLEANING
CHUNKING
ANALYZING
PLANNING
GENERATING
VERIFYING
REVISING
READY_FOR_REVIEW
PUBLISHED
FAILED
```

The UI should expose meaningful status to the admin.

---

# 18. Transcript Cleaning

Raw transcripts may contain:

* filler words
* repetitions
* transcription mistakes
* fragmented sentences
* duplicate phrases
* awkward punctuation
* speaker transitions

Create a cleaning stage.

However:

## DO NOT

Aggressively clean the transcript in a way that changes meaning.

The original raw transcript must always be retained.

Maintain:

```text
raw transcript
      +
clean transcript
```

The clean version is used for downstream AI processing.

The raw version is the source of truth.

---

# 19. Speaker Preservation

If speaker information is available, preserve it.

Example:

```text
HOST:
...

GUEST:
...
```

If speaker identification is unavailable, do not invent speaker identities.

Use:

```text
Speaker 1
Speaker 2
```

or null.

Never hallucinate speaker names.

---

# 20. Semantic Chunking

Do NOT simply split the transcript into arbitrary token sizes.

The chunking system should attempt to preserve semantic coherence.

Possible pipeline:

```text
Transcript segments
       ↓
Sentence segmentation
       ↓
Embedding / semantic similarity
       ↓
Topic boundary detection
       ↓
Meaningful chunks
```

Each chunk should preserve references to the original transcript segments.

Example:

```text
Chunk ID: chunk_017

Topic:
AI scaling

Transcript segments:
segment_124
segment_125
segment_126
...
```

---

# 21. Chunk Size

Chunk size should be configurable.

Do not hard-code a magic number throughout the system.

Example configuration:

```text
CHUNK_TARGET_TOKENS
CHUNK_MIN_TOKENS
CHUNK_MAX_TOKENS
CHUNK_OVERLAP_TOKENS
```

Use semantic boundaries where possible.

Token limits should also respect the selected LLM's context window.

---

# 22. Chunk-Level Analysis

Each chunk should be analyzed independently.

The analysis should extract structured information.

Example:

```json
{
  "main_topic": "AI scaling",
  "subtopics": [
    "compute",
    "data",
    "algorithms"
  ],
  "key_points": [
    "..."
  ],
  "claims": [
    {
      "claim": "...",
      "importance": 0.92
    }
  ],
  "examples": [
    "..."
  ],
  "quotes": [
    "..."
  ],
  "questions": [
    "..."
  ]
}
```

Use structured output.

Do not depend on free-form text parsing if JSON/schema output is available.

---

# 23. Conversation Map

After chunk-level analysis, construct a global representation of the conversation.

The conversation map should identify:

* Major themes.
* Subthemes.
* Arguments.
* Supporting examples.
* Counterarguments.
* Important questions.
* Conclusions.
* Disagreements.
* Recurring ideas.
* Important stories/examples.
* Chronological information when relevant.

Example:

```text
Conversation
│
├── AI Scaling
│   ├── Compute
│   ├── Data
│   └── Algorithms
│
├── AGI
│   ├── Definition
│   ├── Capabilities
│   └── Timeline
│
├── Economics
│   ├── Jobs
│   ├── Productivity
│   └── Companies
│
└── Future
    ├── Risks
    └── Opportunities
```

This representation should be stored.

---

# 24. Do Not Preserve Podcast Chronology Automatically

A podcast may discuss:

```text
AI
→ childhood
→ AI
→ company history
→ AI
→ philosophy
→ AI
```

The final article should not necessarily follow this order.

The article planner should reorganize the ideas logically.

The goal is:

> **logical narrative rather than chronological transcript reproduction.**

However, chronological order should be preserved when chronology itself is important to the subject.

---

# 25. Article Planning

The article planner receives the global conversation representation.

It should produce a structured article plan.

Example:

```json
{
  "title": "...",
  "subtitle": "...",
  "sections": [
    {
      "title": "...",
      "purpose": "...",
      "source_chunk_ids": [
        "chunk_001",
        "chunk_007",
        "chunk_019"
      ]
    }
  ]
}
```

The article planner should determine:

* Article title.
* Introduction.
* Section order.
* Section objectives.
* Relevant source chunks.
* Important claims.
* Supporting examples.
* Conclusion.
* Key takeaways.

---

# 26. Article Length

The product should support reading-time targets.

Initial modes:

```text
Quick
Standard
Deep Dive
```

Suggested targets:

```text
Quick:
5–8 minutes

Standard:
15–25 minutes

Deep Dive:
30–40 minutes
```

These are targets, not rigid requirements.

The system must prioritize completeness and fidelity over hitting an exact word count.

---

# 27. Article Generation

Generate each section separately rather than asking the LLM to write the entire article from scratch.

For each section:

```text
Article section plan
       ↓
Relevant transcript chunks
       ↓
Relevant chunk analyses
       ↓
LLM
       ↓
Draft section
```

This is effectively grounded generation using retrieval.

---

# 28. Section Generation Requirements

Each section must:

* Stay grounded in the transcript.
* Avoid inventing facts.
* Explain ideas clearly.
* Use natural transitions.
* Avoid unnecessary repetition.
* Avoid generic AI language.
* Preserve speaker intent.
* Distinguish facts from opinions.
* Attribute claims appropriately.
* Avoid changing the meaning of arguments.

---

# 29. Writing Style

The article should feel like high-quality editorial writing.

It should NOT sound like:

```text
"In this fascinating podcast, the host and guest delve into..."
```

Avoid generic AI phrases such as:

* "In today's rapidly evolving world..."
* "Let's dive into..."
* "It's important to note..."
* "This fascinating discussion..."
* "The conversation sheds light on..."

Prefer direct, intelligent prose.

Example:

Bad:

```text
The guest talks about the importance of AI scaling.
```

Better:

```text
The guest argues that scaling is not merely a matter of making models larger. The gains come from the interaction between compute, data, and algorithmic improvements.
```

---

# 30. Preserve Opinions

If a speaker expresses an opinion, do not rewrite it as objective fact.

For example:

Transcript:

```text
I believe AGI could arrive within the next decade.
```

Article:

```text
The guest believes AGI could arrive within the next decade.
```

NOT:

```text
AGI will arrive within the next decade.
```

This distinction is critical.

---

# 31. Preserve Disagreements

If the host and guest disagree, preserve the disagreement.

Example:

```text
The host takes a more cautious view of AI capabilities, while the guest argues that current scaling trends suggest much faster progress.
```

Do not merge conflicting opinions into one artificial consensus.

---

# 32. Quotes

Quotes should be used sparingly.

When a quote is used, it must correspond to the transcript.

Never fabricate quotes.

Prefer paraphrasing for most content.

Store source transcript segment IDs for quotes.

---

# 33. Timestamp Linking

Article sections and important claims should be linked to source timestamps where possible.

Example:

```json
{
  "article_section_id": "section_04",
  "transcript_segment_ids": [
    "segment_120",
    "segment_121"
  ]
}
```

Generate YouTube timestamp URLs using the original video ID and timestamp.

Example conceptual format:

```text
https://www.youtube.com/watch?v=VIDEO_ID&t=2530s
```

Do not hard-code the final URL format throughout the application.

Create a helper:

```python
def build_youtube_timestamp_url(video_id: str, seconds: int) -> str:
    ...
```

---

# 34. Article Citation Model

Create an explicit article-to-source relationship.

Example:

```text
Article
   ↓
Section
   ↓
Citation
   ↓
TranscriptSegment
```

This allows future features such as:

> "Show me where this came from."

---

# 35. Fidelity Verification

This is a core component.

After article generation, run a verification stage.

For each important claim:

```text
Generated claim
       ↓
Retrieve supporting transcript
       ↓
Verifier
       ↓
SUPPORTED
PARTIALLY_SUPPORTED
UNSUPPORTED
```

Example:

```json
{
  "claim": "The guest expects AGI by 2030.",
  "status": "UNSUPPORTED",
  "evidence": "The guest discussed rapid progress but gave no specific 2030 prediction."
}
```

---

# 36. Verification Rules

The verifier should look for:

* Unsupported claims.
* Fabricated facts.
* Misattribution.
* Incorrect speaker attribution.
* Changed opinions.
* Missing qualifiers.
* Contradictions.
* Fabricated quotes.
* Incorrect numbers.
* Incorrect dates.
* Incorrect causal relationships.

---

# 37. Revision Loop

If verification identifies problems:

```text
Generate
   ↓
Verify
   ↓
Problems?
   ├── No → Continue
   │
   └── Yes
         ↓
      Revise
         ↓
      Verify again
```

Set a maximum number of revision attempts.

Example:

```text
MAX_REVISION_ATTEMPTS=2
```

Avoid infinite loops.

---

# 38. LangGraph Workflow

Use LangGraph to orchestrate the AI processing pipeline.

Conceptual graph:

```text
START
  ↓
fetch_transcript
  ↓
clean_transcript
  ↓
semantic_chunk
  ↓
analyze_chunks
  ↓
build_conversation_map
  ↓
generate_article_plan
  ↓
generate_sections
  ↓
verify_article
  ↓
 ┌───────────────┐
 │ verification  │
 │ successful?   │
 └───────┬───────┘
       yes│       │no
          │       ↓
          │    revise_article
          │       │
          │       ↓
          │    verify_article
          │
          ↓
calculate_reading_time
          ↓
prepare_publishable_article
          ↓
END
```

The workflow should use typed state.

Example conceptual state:

```python
class PodcastProcessingState(TypedDict):
    episode_id: str
    transcript_id: str
    chunk_ids: list[str]
    analyses: list[dict]
    conversation_map: dict
    article_plan: dict
    generated_sections: list[dict]
    verification_results: list[dict]
    revision_count: int
    final_article: dict
```

---

# 39. LangGraph Design Principles

Each node should perform one logical responsibility.

Do NOT create one giant LangGraph node containing the entire application.

Bad:

```python
def process_everything():
    ...
```

Better:

```python
fetch_transcript()
clean_transcript()
chunk_transcript()
analyze_chunks()
build_conversation_map()
create_article_plan()
generate_sections()
verify_article()
revise_article()
```

---

# 40. RAG Architecture

RAG should be used for grounding article generation and question answering.

The vector store should contain transcript chunks.

Each vector record should have metadata:

```json
{
  "episode_id": "...",
  "chunk_id": "...",
  "start_ms": 123000,
  "end_ms": 135000,
  "topic": "AI scaling"
}
```

This allows retrieval by:

* semantic similarity
* episode
* topic
* timestamp
* chunk ID

Always filter retrieval to the correct episode when answering questions about an episode.

---

# 41. Podcast Q&A

The article page should eventually provide:

```text
Ask about this podcast...

"What did the guest say about AGI?"

"Did the host disagree with the guest?"

"What were the three strongest arguments?"
```

Pipeline:

```text
User Question
      ↓
Episode-filtered retrieval
      ↓
Relevant transcript chunks
      ↓
LLM
      ↓
Grounded answer
      ↓
Timestamp references
```

The assistant must not answer from general model knowledge when the user is asking what the podcast said.

---

# 42. Q&A Grounding

Answers should cite relevant transcript timestamps.

Example:

```text
The guest argues that scaling remains one of the strongest
drivers of capability improvements.

🎧 42:13
```

If the transcript does not contain enough evidence:

```text
The transcript does not provide enough information to answer this confidently.
```

Do not hallucinate.

---

# 43. Article Page

The article page should contain:

```text
Podcast title

Host × Guest

Original duration:
2h 17m

Reading time:
21 min

[Listen to Original Podcast]
```

Then:

```text
TL;DR
```

Then:

```text
What you'll learn
```

Then the main article.

Then:

```text
Key Takeaways
```

Then:

```text
Original Podcast
```

with link and metadata.

---

# 44. Reading Experience

The reading UI is extremely important.

Design goals:

* Clean.
* Minimal.
* Excellent typography.
* Large readable text.
* Strong hierarchy.
* Good spacing.
* Mobile friendly.
* Dark/light mode if easy to implement.
* Minimal distractions.

Avoid making it look like a generic AI dashboard.

The primary experience is reading.

---

# 45. Article Components

Create reusable components such as:

```text
ArticleHeader
PodcastMetadata
ReadingTime
TLDR
LearningObjectives
ArticleSection
TimestampLink
KeyTakeaways
PodcastSourceCard
RelatedEpisodes
PodcastQA
```

---

# 46. Homepage

Homepage should communicate the product immediately.

Hero:

```text
Turn conversations into knowledge.

Read the ideas from hours-long podcasts
in a fraction of the time.

[Explore Podcasts]
```

Then:

```text
Featured Episodes
```

Then categories/topics.

Then:

```text
How it works
```

Example:

```text
1. We process the conversation
2. AI extracts the important ideas
3. AI organizes and verifies them
4. You read the result
```

---

# 47. Podcast Library

The public library should support:

* Search.
* Podcast/channel filtering.
* Topic filtering.
* Sort by latest/popular.
* Episode cards.

Episode card:

```text
Thumbnail

Episode title

Podcast name

Original duration
Estimated reading time

Short description

[Read Article]
```

---

# 48. Admin Dashboard

Create an admin interface.

Pages:

```text
/admin
/admin/episodes
/admin/episodes/new
/admin/episodes/[id]
/admin/processing
/admin/articles
```

The admin should be able to:

* Add URL.
* Fetch metadata.
* Fetch transcript.
* Start processing.
* See progress.
* Inspect transcript.
* Inspect article.
* Inspect verification results.
* Edit article.
* Publish.
* Unpublish.

---

# 49. Processing UI

Show meaningful progress.

Example:

```text
Processing Episode

✓ Fetching metadata
✓ Fetching transcript
✓ Cleaning transcript
✓ Creating semantic chunks
✓ Analyzing conversation
● Building article
○ Verifying
○ Publishing
```

If a step fails, show the reason.

---

# 50. Error Handling

Every external call must handle:

* Timeout.
* Rate limit.
* Invalid URL.
* Missing transcript.
* API authentication failure.
* Provider failure.
* LLM failure.
* Invalid structured output.
* Database failure.

Use retries where appropriate.

Retries must have limits.

---

# 51. Idempotency

Processing the same YouTube URL twice should not blindly duplicate the episode.

Normalize and identify the YouTube video ID.

Use the video ID as a unique logical identifier.

Before ingestion:

```text
Does episode already exist?
       │
       ├── yes → return existing episode
       │
       └── no → create episode
```

Allow explicit reprocessing when desired.

---

# 52. Caching

Transcript acquisition should be cached.

Do not repeatedly call Supadata for the same episode.

Cache/store:

* Metadata.
* Transcript.
* Chunks.
* Embeddings.
* Analysis.
* Generated article.

Re-running a later stage should not require repeating all earlier stages unless explicitly requested.

For example:

```text
Article generation changed
        ↓
Don't fetch transcript again.
```

---

# 53. Configuration

Use environment variables.

Example:

```text
APP_ENV=development

DATABASE_URL=...

REDIS_URL=...

SUPADATA_API_KEY=...

LLM_API_KEY=...

EMBEDDING_MODEL=...

VECTOR_DIMENSION=...
```

Provide:

```text
.env.example
```

Never commit `.env`.

---

# 54. Secrets

Never:

* hard-code secrets.
* print secrets.
* commit secrets.
* include API keys in frontend code.

All secret operations belong on the backend.

---

# 55. API Design

FastAPI endpoints should be clean and versioned.

Example:

```text
/api/v1/episodes
/api/v1/episodes/{episode_id}
/api/v1/episodes/{episode_id}/transcript
/api/v1/episodes/{episode_id}/process
/api/v1/episodes/{episode_id}/article
/api/v1/episodes/{episode_id}/qa
```

Admin endpoints should be separated logically.

---

# 56. Suggested API Flow

Create episode:

```http
POST /api/v1/episodes
```

Request:

```json
{
  "youtube_url": "..."
}
```

Response:

```json
{
  "episode_id": "...",
  "status": "INGESTING"
}
```

Then:

```http
POST /api/v1/episodes/{episode_id}/process
```

The processing should happen asynchronously.

---

# 57. Background Processing

Podcast processing can take significant time.

Do NOT make the user wait on a single synchronous HTTP request.

Use:

```text
API
 ↓
Create processing job
 ↓
Queue
 ↓
Worker
 ↓
LangGraph workflow
```

Frontend polls or receives status updates.

---

# 58. Observability

Log each major pipeline stage.

Example:

```text
episode_id
job_id
node_name
start_time
end_time
status
token_usage
model
error
```

Do not log secrets or sensitive content unnecessarily.

---

# 59. LLM Cost Tracking

Track LLM usage per episode.

Store:

```text
model
input_tokens
output_tokens
estimated_cost
node
episode_id
timestamp
```

This allows us to answer:

> How much does it cost to process one 2-hour podcast?

This should become a product metric.

---

# 60. Evaluation

Evaluation is a first-class component.

Do not judge the system only by whether the article "looks good."

Measure:

### Compression

```text
original estimated reading time
vs
generated reading time
```

### Coverage

Did we retain the important ideas?

### Fidelity

Are claims supported by the transcript?

### Hallucination

How many unsupported claims?

### Attribution

Are claims attributed to the correct speaker?

### Citation accuracy

Does the timestamp actually contain the relevant discussion?

### Readability

Is the article coherent and easy to read?

---

# 61. Evaluation Dataset

Create a small manually reviewed dataset.

Example:

```text
evaluation/
    episodes/
        episode_001/
        episode_002/
        episode_003/
```

For each episode store expected:

* major topics
* important claims
* important disagreements
* key takeaways

Use this dataset to evaluate changes to prompts/models.

---

# 62. Prompt Versioning

Do not scatter prompts across Python files.

Create:

```text
prompts/
    chunk_analysis/
    conversation_map/
    article_planning/
    article_generation/
    verification/
    revision/
    qa/
```

Version prompts where useful.

Example:

```text
article_generation_v1
article_generation_v2
```

---

# 63. Structured Outputs

Where the LLM produces machine-consumed information, use schemas.

For example:

```python
class Claim(BaseModel):
    text: str
    importance: float
```

and:

```python
class ChunkAnalysis(BaseModel):
    main_topic: str
    subtopics: list[str]
    key_points: list[str]
    claims: list[Claim]
```

Do not use fragile regex parsing for structured LLM responses.

---

# 64. LLM Provider Abstraction

The application should not be tightly coupled to one model.

Conceptually:

```python
class LLMProvider(ABC):

    async def generate_structured(...):
        ...

    async def generate_text(...):
        ...
```

The initial implementation can use one provider.

The architecture should allow another provider later.

---

# 65. Embedding Provider

Similarly:

```python
class EmbeddingProvider:

    async def embed_documents(...):
        ...

    async def embed_query(...):
        ...
```

The provider should be replaceable.

---

# 66. Security

Implement basic security from the beginning.

Requirements:

* Authentication for admin routes.
* Authorization.
* Input validation.
* URL validation.
* Rate limiting where appropriate.
* Secure secret management.
* SQL injection protection through ORM/parameterized queries.
* XSS protection in article rendering.
* Sanitization of generated HTML/Markdown.

Never trust LLM-generated HTML directly.

---

# 67. Article Storage Format

Prefer storing structured article content rather than only raw HTML.

Example:

```json
{
  "title": "...",
  "subtitle": "...",
  "sections": [
    {
      "heading": "...",
      "content": "...",
      "citations": [...]
    }
  ],
  "takeaways": [...]
}
```

Render this through trusted frontend components.

This gives us flexibility later.

---

# 68. Markdown / MDX

Markdown/MDX may be used as an internal representation if convenient.

However, article rendering should remain controlled.

Do not blindly render arbitrary generated HTML.

---

# 69. Search

Initial search can use PostgreSQL.

Search:

* Episode title.
* Podcast name.
* Guest name.
* Topic.
* Article title.

Semantic search can be added later.

Do not build Elasticsearch/OpenSearch in V1 unless actually necessary.

---

# 70. Future Features

These are NOT MVP requirements but architecture should not prevent them.

Possible future features:

## Multiple reading modes

```text
5 min
20 min
40 min
```

## Podcast Q&A

Already planned as a basic feature.

## Personalized reading

```text
AI engineer perspective
Entrepreneur perspective
Beginner perspective
Researcher perspective
```

## Topic pages

```text
Artificial Intelligence
Startups
Science
Business
Philosophy
Economics
```

## Cross-podcast knowledge

Example:

> "Compare what Lex Fridman guests have said about AGI."

## Knowledge graph

```text
Person
   ↓
Podcast
   ↓
Topic
   ↓
Concept
   ↓
Other Podcast
```

## Recommendations

Recommend episodes based on:

* topics
* authors
* guests
* reading history

## Creator dashboard

Allow creators to submit podcasts and approve generated articles.

## Automatic ingestion

Eventually:

```text
New YouTube episode
       ↓
Detect
       ↓
Transcript
       ↓
Process
       ↓
Review
       ↓
Publish
```

Do not implement this automatically in MVP.

---

# 71. Future Provider Architecture

The system should eventually support:

```text
TranscriptProvider
├── Supadata
├── Provider B
└── Provider C

LLMProvider
├── Provider A
├── Provider B
└── Local Model

EmbeddingProvider
├── Local
└── Hosted
```

This is important for cost control and reliability.

---

# 72. Local Development

The repository should support:

```bash
docker compose up
```

for:

* PostgreSQL
* pgvector
* Redis

Application services can run locally during development.

Provide a clear README.

---

# 73. Repository Structure

Suggested structure:

```text
podcast-to-knowledge/

├── README.md
├── PRODUCT_SPEC.md
├── docker-compose.yml
├── .env.example
├── .gitignore
│
├── backend/
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── core/
│   │   ├── config/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── repositories/
│   │   ├── services/
│   │   ├── providers/
│   │   ├── workflows/
│   │   ├── prompts/
│   │   ├── evaluation/
│   │   └── utils/
│   │
│   └── tests/
│
├── frontend/
│   ├── package.json
│   ├── app/
│   ├── components/
│   ├── lib/
│   └── tests/
│
├── evaluation/
│
├── docs/
│   ├── architecture/
│   ├── adr/
│   └── prompts/
│
└── scripts/
```

Claude Code may improve this structure if there is a strong engineering reason.

---

# 74. Testing Strategy

Tests should exist at multiple levels.

## Unit tests

Test:

* YouTube ID extraction.
* Timestamp conversion.
* URL generation.
* Transcript normalization.
* Chunking.
* Reading time.
* Data validation.

## Integration tests

Test:

```text
Supadata → normalized transcript
Database operations
Vector storage/retrieval
LLM provider abstraction
```

## Workflow tests

Test:

```text
Transcript
→ chunks
→ analysis
→ article plan
→ article
→ verification
```

Use mocks for external services.

---

# 75. Failure Recovery

If the workflow fails at:

```text
article generation
```

do not rerun:

```text
transcript acquisition
cleaning
chunking
analysis
```

unless required.

Persist intermediate results.

The workflow should be resumable.

---

# 76. Reprocessing

Admin should be able to choose:

```text
Reprocess entire episode
```

or:

```text
Regenerate article only
```

or:

```text
Re-run verification only
```

This is why intermediate state must be persisted.

---

# 77. Editorial Review

Even though the system is AI-driven, V1 should support human review.

Workflow:

```text
AI Generated
      ↓
Verification
      ↓
READY_FOR_REVIEW
      ↓
Admin edits
      ↓
Publish
```

Do not assume the LLM is always correct.

---

# 78. Publishing

Only articles explicitly marked as published should be visible publicly.

Article states:

```text
DRAFT
PROCESSING
READY_FOR_REVIEW
PUBLISHED
UNPUBLISHED
ARCHIVED
```

---

# 79. SEO

The public article pages should be SEO-friendly.

Each article should have:

* SEO title.
* Meta description.
* Canonical URL.
* Open Graph metadata.
* Structured metadata where appropriate.
* Clean URL slug.

Example:

```text
/articles/future-of-ai-guest-name
```

---

# 80. Performance

Public article pages should load quickly.

Prefer:

* Server-side rendering where useful.
* Static/cached article content.
* Optimized images.
* Lazy loading.
* Minimal client-side JavaScript.

The reading experience should feel fast.

---

# 81. Accessibility

Follow basic accessibility practices:

* Semantic HTML.
* Keyboard navigation.
* Proper headings.
* Alt text.
* Sufficient contrast.
* Accessible controls.
* Responsive design.

---

# 82. Product Analytics

Eventually track:

```text
article views
reading completion
average reading time
timestamp clicks
Q&A usage
search queries
popular podcasts
```

Do not add complex analytics infrastructure in MVP.

---

# 83. Core Metrics

The most important product metrics are:

### Time compression

```text
Podcast duration
vs
Article reading time
```

### Completion

Percentage of users who finish an article.

### Fidelity

Percentage of verified claims supported by transcript.

### Engagement

Timestamp clicks and Q&A interactions.

### Quality

Human/editorial quality score.

---

# 84. Example End-to-End Workflow

Input:

```text
YouTube URL
```

Step 1:

```text
Extract YouTube video ID
```

Step 2:

```text
Fetch metadata
```

Step 3:

```text
Fetch transcript using Supadata
```

Step 4:

```text
Normalize transcript
```

Step 5:

```text
Store raw transcript
```

Step 6:

```text
Clean transcript
```

Step 7:

```text
Semantic chunking
```

Step 8:

```text
Generate chunk analyses
```

Step 9:

```text
Build conversation map
```

Step 10:

```text
Generate article plan
```

Step 11:

```text
Retrieve relevant transcript chunks for each section
```

Step 12:

```text
Generate article sections
```

Step 13:

```text
Generate key takeaways
```

Step 14:

```text
Run fidelity verification
```

Step 15:

```text
Revise if necessary
```

Step 16:

```text
Calculate reading time
```

Step 17:

```text
Generate timestamp links
```

Step 18:

```text
READY_FOR_REVIEW
```

Step 19:

```text
Admin reviews
```

Step 20:

```text
Publish
```

---

# 85. Example Article Output

Given:

```text
Podcast:
2h 05m

Topic:
The Future of AI
```

The article should look conceptually like:

```text
THE FUTURE OF AI

Host × Guest

🎧 Original podcast: 2h 05m
📖 Reading time: 21 min


TL;DR

The conversation explores...


WHAT YOU'LL LEARN

• Why scaling has driven recent AI progress
• Where current systems remain limited
• How the guest thinks about AGI
• What AI could mean for human work


1. WHY AI SYSTEMS KEEP IMPROVING

...

🎧 Listen to this discussion — 37:42


2. THE ROAD TO AGI

...

🎧 Listen to this discussion — 1:02:14


3. WHAT HAPPENS TO HUMAN WORK?

...


KEY TAKEAWAYS

1. ...
2. ...
3. ...
4. ...
5. ...


ORIGINAL PODCAST

[Watch on YouTube]
```

---

# 86. Important Quality Rule

The article should NOT attempt to preserve every sentence.

It should preserve:

```text
meaning
+
important arguments
+
important evidence/examples
+
important disagreements
+
important conclusions
```

It may remove:

```text
filler
+
repetition
+
small talk
+
irrelevant tangents
+
redundant explanations
```

But removal must not change the overall meaning of the conversation.

---

# 87. Avoid "Summary Voice"

The article should not repeatedly say:

```text
"The guest says..."
"The host says..."
"The guest then explains..."
```

Use natural editorial prose.

Use attribution only where necessary.

Example:

Bad:

```text
The guest says that AI is changing programming.
The guest also says that programmers need to adapt.
The guest then says that...
```

Better:

```text
AI is already changing the economics of software development. Rather than eliminating programmers outright, the guest argues that increasingly capable tools will shift where developers spend their time—from writing routine code toward designing systems, evaluating outputs, and solving higher-level problems.
```

---

# 88. Do Not Hallucinate

The system must NEVER invent:

* Quotes.
* Statistics.
* Dates.
* Events.
* People.
* Companies.
* Opinions.
* Predictions.
* Scientific claims.

If something isn't supported:

```text
Do not include it.
```

or explicitly qualify uncertainty.

---

# 89. Source Attribution

Every public article must identify the source podcast.

Example:

```text
Source:
Podcast Name
Episode Name
Host / Guest
Original YouTube video
```

The platform should make it clear that the article is derived from the original conversation.

---

# 90. Prompt Engineering Philosophy

Prompts should emphasize:

```text
Grounding
Accuracy
Clarity
Editorial quality
Preservation of meaning
Attribution
Conciseness
```

Do not optimize prompts only for "shortest possible summary."

---

# 91. Architecture Decision: No Microservices

Do NOT split into:

```text
transcript-service
LLM-service
article-service
vector-service
verification-service
```

for V1.

Use modules inside one backend.

Example:

```text
FastAPI
│
├── ingestion
├── transcript
├── processing
├── retrieval
├── generation
├── verification
├── articles
└── admin
```

This is easier to develop and deploy.

---

# 92. Architecture Decision: PostgreSQL + pgvector

Use PostgreSQL as the primary database.

Use pgvector for semantic search.

Do not introduce a dedicated vector database initially.

The number of documents in V1 does not justify unnecessary infrastructure.

---

# 93. Architecture Decision: Supadata as Provider

Use Supadata as the initial transcript provider.

But isolate it behind:

```python
TranscriptProvider
```

This allows replacement later.

---

# 94. Architecture Decision: LangGraph

Use LangGraph specifically for the multi-stage AI workflow.

Do not use LangGraph for:

* CRUD.
* Simple database calls.
* Authentication.
* Basic API endpoints.

Use it where stateful AI orchestration provides real value.

---

# 95. Architecture Decision: RAG

Use RAG for:

* Grounded article section generation.
* Fidelity verification.
* Podcast Q&A.

Do not use RAG merely because it is an expected buzzword.

---

# 96. Architecture Decision: Human-in-the-loop

V1 should support human review before publication.

This protects against:

* hallucinations
* bad summaries
* incorrect attribution
* poor writing
* inappropriate compression

---

# 97. Development Strategy

Build incrementally.

## Phase 1 — Foundation

Build:

```text
Repository
Docker
PostgreSQL
Redis
FastAPI
Next.js
Environment configuration
Basic health checks
```

## Phase 2 — Ingestion

Build:

```text
YouTube URL validation
Supadata provider
Metadata
Transcript normalization
Database persistence
```

## Phase 3 — Processing

Build:

```text
Cleaning
Chunking
Embeddings
Chunk analysis
```

## Phase 4 — Intelligence

Build:

```text
Conversation map
Article planner
Article generator
```

## Phase 5 — Verification

Build:

```text
Claim extraction
Claim verification
Revision loop
```

## Phase 6 — UI

Build:

```text
Homepage
Library
Article page
Admin dashboard
Processing status
```

## Phase 7 — Q&A

Build:

```text
Episode-filtered RAG
Question answering
Timestamp citations
```

## Phase 8 — Evaluation

Build:

```text
Evaluation dataset
Metrics
Regression tests
Cost tracking
```

---

# 98. Claude Code Instructions

You are acting as a senior software architect and AI engineer.

Before implementing a major component:

1. Understand the existing repository.
2. Inspect existing files.
3. Avoid unnecessary rewrites.
4. Follow the architecture in this document.
5. If an architectural change is necessary, explain why.
6. Prefer simple solutions over unnecessary infrastructure.
7. Keep modules loosely coupled.
8. Write tests alongside important functionality.
9. Never hard-code secrets.
10. Use environment variables.
11. Add logging for important pipeline stages.
12. Make external providers replaceable.
13. Preserve timestamps throughout the pipeline.
14. Preserve source attribution.
15. Treat the transcript as the source of truth.
16. Never invent transcript content.
17. Use structured LLM outputs.
18. Make processing resumable.
19. Make processing idempotent.
20. Keep the application runnable locally.

---

# 99. Claude Code Development Rules

Do NOT immediately generate the entire application in one huge implementation.

Work in vertical slices.

For each feature:

```text
Understand
   ↓
Design
   ↓
Implement
   ↓
Test
   ↓
Run
   ↓
Fix
   ↓
Document
```

Do not create placeholder implementations that pretend to work.

If an external API is unavailable during development, use a clearly marked mock/provider implementation.

Do not silently replace real functionality with fake data.

---

# 100. First Task

Before writing application code:

1. Inspect the repository.
2. Determine what already exists.
3. Create or update:

```text
PRODUCT_SPEC.md
ARCHITECTURE.md
README.md
.env.example
```

4. Propose the initial repository structure.
5. Identify architectural decisions.
6. Identify required environment variables.
7. Identify dependencies.
8. Identify the MVP implementation order.

Then implement only the foundation.

Do NOT jump directly to article generation.

---

# 101. First Milestone

The first functional milestone should be:

```text
User/Admin enters YouTube URL
          ↓
Backend validates URL
          ↓
Supadata retrieves metadata
          ↓
Supadata retrieves transcript
          ↓
Transcript normalized
          ↓
Transcript stored in PostgreSQL
          ↓
Admin can inspect transcript
```

Only after this works reliably should the AI processing pipeline be implemented.

---

# 102. Second Milestone

Then implement:

```text
Stored transcript
       ↓
Cleaning
       ↓
Semantic chunking
       ↓
Embeddings
       ↓
Chunk analysis
       ↓
Conversation map
```

---

# 103. Third Milestone

Then:

```text
Conversation map
       ↓
Article outline
       ↓
Section generation
       ↓
Article
```

---

# 104. Fourth Milestone

Then:

```text
Article
 ↓
Claim extraction
 ↓
Transcript retrieval
 ↓
Verification
 ↓
Revision
```

---

# 105. Fifth Milestone

Then build the polished public reading experience.

---

# 106. Definition of Done — MVP

The MVP is considered complete when:

### Ingestion

* [ ] Admin can submit a YouTube URL.
* [ ] Video ID is correctly extracted.
* [ ] Metadata is retrieved.
* [ ] Transcript is retrieved through Supadata.
* [ ] Raw transcript is preserved.
* [ ] Transcript is normalized.
* [ ] Timestamps are preserved.

### Processing

* [ ] Transcript can be semantically chunked.
* [ ] Chunks are stored.
* [ ] Embeddings are stored.
* [ ] Chunk analyses are generated.
* [ ] Conversation map is generated.
* [ ] Article outline is generated.

### Generation

* [ ] Article sections are generated.
* [ ] Article is grounded in transcript chunks.
* [ ] Reading time is calculated.
* [ ] Key takeaways are generated.
* [ ] Timestamp references are generated.

### Verification

* [ ] Important claims are checked.
* [ ] Unsupported claims are detected.
* [ ] Revision loop works.
* [ ] Verification results are stored.

### Admin

* [ ] Admin can inspect processing status.
* [ ] Admin can inspect transcript.
* [ ] Admin can inspect generated article.
* [ ] Admin can edit article.
* [ ] Admin can publish/unpublish.

### Public

* [ ] Homepage exists.
* [ ] Podcast/episode library exists.
* [ ] Search exists.
* [ ] Article page exists.
* [ ] Podcast source is clearly attributed.
* [ ] Timestamp links work.
* [ ] Reading experience is responsive.

### Engineering

* [ ] Tests exist.
* [ ] Docker Compose works.
* [ ] `.env.example` exists.
* [ ] Secrets are not committed.
* [ ] README contains setup instructions.
* [ ] Processing is resumable.
* [ ] External providers are abstracted.

---

# 107. Final Product Philosophy

The product should optimize for:

```text
                         QUALITY
                           ▲
                           │
                           │
                ┌──────────┴──────────┐
                │                     │
           FIDELITY              READABILITY
                │                     │
                └──────────┬──────────┘
                           │
                      TIME SAVED
```

The user should come away thinking:

> "I didn't listen to the entire two-hour conversation, but after reading this, I understand what was actually discussed."

That is the product.

The system should therefore be treated as an **AI-powered editorial and knowledge extraction platform**, not merely a transcript summarizer.

---

# 108. Guiding Principle

At every stage ask:

> **Does this help us turn a long conversation into a shorter, clearer, more trustworthy representation of the ideas contained in it?**

If yes, consider it.

If no, do not add it merely because it is technically interesting.

Build the simplest production-quality system that achieves the core product promise:

> **2 hours of conversation → 15–30 minutes of high-quality, grounded reading.**
