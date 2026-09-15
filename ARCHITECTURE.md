# Architecture

This document translates [`PRODUCT_SPEC.md`](./PRODUCT_SPEC.md) into concrete engineering decisions: the repository layout, the module boundaries, the data model, the processing pipeline, required configuration, and the build order. It is derived from the spec and should stay in sync with it — if they disagree, the spec is the source of truth for product intent and this file should be updated to match.

No application code exists yet. This document describes the target architecture for the foundation phase and beyond.

---

## 1. Guiding constraints (from the spec)

- **Modular monolith, not microservices** (§7, §91). One FastAPI backend with clear internal module boundaries; one Next.js frontend.
- **No premature infrastructure**: no Kubernetes, no dedicated vector DB, no multi-provider abstraction gold-plating, no Elasticsearch (§7, §69, §92).
- **Every external dependency is abstracted** behind a provider interface (§11, §64, §65, §93): `TranscriptProvider`, `LLMProvider`, `EmbeddingProvider`, `StorageProvider`.
- **The raw transcript is the source of truth** (§18, §98). Cleaning never overwrites it; generation must be grounded in it; nothing is ever invented (§88).
- **Timestamps and speaker labels survive the entire pipeline** (§14, §19, §98) so the UI can always link back to `youtube.com/watch?v=VIDEO_ID&t=Ns`.
- **Everything long-running is async and resumable** (§57, §75, §76): a processing job persists intermediate state per stage so a later stage can be re-run without repeating earlier ones.
- **Human review gates publishing** (§77, §96): nothing reaches the public site without passing through `READY_FOR_REVIEW`.

---

## 2. Repository structure

```text
podcast-to-knowledge-platform/
├── README.md
├── PRODUCT_SPEC.md
├── ARCHITECTURE.md
├── docker-compose.yml
├── .env.example
├── .gitignore
│
├── backend/
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py                # FastAPI app factory, router registration
│   │   ├── api/                   # HTTP layer only — request/response, no business logic
│   │   │   └── v1/
│   │   │       ├── episodes.py
│   │   │       ├── articles.py
│   │   │       ├── qa.py
│   │   │       └── admin/
│   │   ├── core/                  # cross-cutting: logging, security, exceptions, retries
│   │   ├── config/                # pydantic settings, env var loading
│   │   ├── models/                # SQLAlchemy ORM models (DB shape, §16)
│   │   ├── schemas/                # Pydantic request/response + structured-output schemas (§63)
│   │   ├── repositories/          # DB access, one per aggregate root
│   │   ├── services/              # orchestration/business logic, calls repositories + providers
│   │   ├── providers/             # external-system adapters (§11)
│   │   │   ├── transcript/        # TranscriptProvider, SupadataTranscriptProvider
│   │   │   ├── llm/               # LLMProvider + concrete implementation
│   │   │   ├── embedding/         # EmbeddingProvider + concrete implementation
│   │   │   └── storage/           # StorageProvider (object storage abstraction)
│   │   ├── workflows/             # LangGraph graphs + node functions (§38, §39)
│   │   ├── prompts/               # versioned prompt templates (§62), one dir per pipeline stage
│   │   ├── evaluation/            # metrics + evaluation harness (§60, §61)
│   │   └── utils/                 # youtube id extraction, timestamp/url helpers, reading time
│   │
│   └── tests/
│       ├── unit/
│       ├── integration/
│       └── workflow/
│
├── frontend/
│   ├── package.json
│   ├── app/                       # Next.js app router: public site + /admin
│   ├── components/                # ArticleHeader, TimestampLink, KeyTakeaways, PodcastQA, ...
│   ├── lib/                       # API client, formatting helpers
│   └── tests/
│
├── evaluation/
│   └── episodes/                  # manually reviewed eval dataset (§61)
│
├── docs/
│   ├── architecture/
│   ├── adr/                       # architecture decision records
│   └── prompts/
│
└── scripts/
```

This matches §73 with one addition: `frontend/app` explicitly hosts both the public site and `/admin` as route groups, since the spec treats admin as part of the same Next.js app rather than a separate project.

---

## 3. Backend module responsibilities

| Module | Responsibility | Must NOT contain |
|---|---|---|
| `api/` | HTTP request/response, validation, status codes | business logic, direct DB/provider calls |
| `core/` | logging, auth/authz, exception types, retry helpers | feature-specific logic |
| `config/` | typed `Settings` object loaded from env vars | secrets committed as literals |
| `models/` | SQLAlchemy table definitions | query logic beyond relationships |
| `schemas/` | Pydantic I/O and LLM structured-output schemas (§63) | ORM concerns |
| `repositories/` | CRUD + queries per entity | business rules |
| `services/` | use-case orchestration (e.g. `EpisodeIngestionService`, `ArticlePublishingService`) | raw SQL, raw HTTP calls to providers |
| `providers/` | adapters implementing the abstract provider interfaces (§11) | business rules |
| `workflows/` | LangGraph graph definitions + per-stage node functions (§38–39) | HTTP concerns |
| `prompts/` | prompt templates, one subfolder per stage, versioned (§62) | logic |
| `evaluation/` | fidelity/coverage/compression metrics (§60) | — |

Rule of thumb from §94: LangGraph is used only for the stateful AI pipeline (transcript → article → verification), never for CRUD, auth, or plain API endpoints.

---

## 4. Provider abstractions (§11, §64, §65, §93)

```python
class TranscriptProvider(ABC):
    async def get_transcript(self, video_url: str) -> Transcript: ...

class LLMProvider(ABC):
    async def generate_structured(self, prompt: str, schema: type[BaseModel], **kw) -> BaseModel: ...
    async def generate_text(self, prompt: str, **kw) -> str: ...

class EmbeddingProvider(ABC):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...

class StorageProvider(ABC):
    async def put(self, key: str, data: bytes, content_type: str) -> str: ...
    async def get(self, key: str) -> bytes: ...
```

V1 concrete implementations:

- `TranscriptProvider` → `SupadataTranscriptProvider` (§12–13). Normalizes Supadata's response into the internal `Transcript`/`TranscriptSegment` model; the rest of the app never sees Supadata's schema.
- `LLMProvider` → a single provider implementation (provider choice is an implementation detail behind the interface; no multi-provider routing in V1, per §7/§64).
- `EmbeddingProvider` → a single hosted embedding implementation, dimension driven by `VECTOR_DIMENSION`.
- `StorageProvider` → local filesystem or S3-compatible implementation for raw transcripts/artifacts (§9 storage note); swappable without touching business logic.

`services/` and `workflows/` depend only on the abstract interfaces, injected via `config`/DI — never import a concrete provider directly.

---

## 5. Data model (§16)

```text
Podcast
   └── Episode
          ├── Transcript
          │       └── TranscriptSegment
          ├── Chunk
          │       └── ChunkAnalysis
          ├── Topic (via EpisodeTopic)
          ├── ProcessingJob
          ├── EvaluationResult
          └── Article
                  ├── ArticleSection
                  └── ArticleCitation → TranscriptSegment
```

Key fields per entity (non-exhaustive, expand in `models/` as built):

- **Podcast**: `channel_id`, `channel_name`, `thumbnail_url`.
- **Episode**: `video_id` (unique, normalized — see §51 idempotency), `youtube_url`, `title`, `description`, `duration_seconds`, `published_at`, `language`, `ingested_at`, `processed_at`, `published_at`, `processing_status` (§17 enum).
- **Transcript**: `episode_id`, `is_raw: bool` — both raw and cleaned versions are stored as rows, never overwritten (§18).
- **TranscriptSegment**: `text`, `start_ms`, `duration_ms`, `speaker: str | None` — `speaker` is `null` or `Speaker N` unless real diarization data exists; never invented (§19).
- **Chunk**: `episode_id`, `topic`, ordered list of `transcript_segment_ids` (§20).
- **ChunkAnalysis**: structured output per §22 (`main_topic`, `subtopics`, `key_points`, `claims[]`, `examples[]`, `quotes[]`, `questions[]`).
- **Article** / **ArticleSection**: structured content per §67 (JSON sections, not raw HTML) plus `status` (§78 enum).
- **ArticleCitation**: `article_section_id` → `transcript_segment_ids[]`, used for timestamp linking (§33–34) and quote sourcing (§32).
- **ProcessingJob**: `episode_id`, `status`, per-stage timestamps, `revision_count`, error detail — this is what makes the workflow resumable (§75) and drives the admin progress UI (§49).
- **EvaluationResult**: per-episode compression/coverage/fidelity/hallucination scores (§60–61).

Embeddings live on `Chunk` (or a dedicated `ChunkEmbedding` table) as a `pgvector` column, with metadata (`episode_id`, `chunk_id`, `start_ms`, `end_ms`, `topic`) mirrored into normal columns so retrieval can filter without unpacking vector metadata (§40).

### Processing status enum (§17)

```
INGESTING → TRANSCRIPT_FETCHED → CLEANING → CHUNKING → ANALYZING →
PLANNING → GENERATING → VERIFYING → (REVISING → VERIFYING)* →
READY_FOR_REVIEW → PUBLISHED
                  ↘ FAILED (from any stage)
```

### Article status enum (§78)

```
DRAFT → PROCESSING → READY_FOR_REVIEW → PUBLISHED ⇄ UNPUBLISHED → ARCHIVED
```

---

## 6. AI pipeline (LangGraph, §38–39)

One graph, typed state (`PodcastProcessingState`), one node per responsibility:

```
fetch_transcript → clean_transcript → semantic_chunk → analyze_chunks →
build_conversation_map → generate_article_plan → generate_sections →
verify_article → (revise_article → verify_article)* →
calculate_reading_time → prepare_publishable_article
```

- Each node reads/writes only the state keys it owns and persists its output via the corresponding repository, so a crash mid-graph doesn't lose completed stages (§75).
- The verify/revise loop is bounded by `MAX_REVISION_ATTEMPTS` (§37, §60 default `2`).
- The graph runs inside a background worker consuming a Redis-backed queue, triggered by `POST /api/v1/episodes/{id}/process`; the API returns immediately with a job id (§57).

RAG (§40, §95) is used in exactly three places, per the spec's "don't use RAG as a buzzword" rule: grounded section generation, fidelity verification (retrieve evidence for a claim), and podcast Q&A — always filtered by `episode_id`.

---

## 7. API surface (§55–56)

```
POST   /api/v1/episodes                       create/idempotently return an episode from a YouTube URL
GET    /api/v1/episodes/{id}
GET    /api/v1/episodes/{id}/transcript
POST   /api/v1/episodes/{id}/process           enqueue processing job (supports reprocess scope, §76)
GET    /api/v1/episodes/{id}/article
POST   /api/v1/episodes/{id}/qa
```

Admin-only mutations (auth required, §66) live under the same routers but are separated by dependency-injected admin auth, not a separate app — matching the "admin dashboard is part of the one Next.js app" decision in §48/§9.

---

## 8. Frontend structure

- Next.js App Router, TypeScript, Tailwind (§9).
- Public routes: `/`, `/articles/[slug]`, library/search views (§46–47, §79 SEO slugs).
- Admin routes: `/admin`, `/admin/episodes`, `/admin/episodes/new`, `/admin/episodes/[id]`, `/admin/processing`, `/admin/articles` (§48), gated behind admin auth.
- Server components for the read-heavy public pages (performance, §80); client components only where interactivity is required (processing status polling, Q&A box, admin forms).
- Shared components per §45: `ArticleHeader`, `PodcastMetadata`, `ReadingTime`, `TLDR`, `LearningObjectives`, `ArticleSection`, `TimestampLink`, `KeyTakeaways`, `PodcastSourceCard`, `RelatedEpisodes`, `PodcastQA`.
- Article content is rendered from the structured JSON model (§67), never from raw LLM-generated HTML (§66, §68).

---

## 9. Local development (§72)

```bash
docker compose up   # postgres (pgvector), redis
```

Backend and frontend run locally (or in Compose once stabilized) against those services. `docker-compose.yml` and `.env.example` are added as part of the Phase 1 foundation.

---

## 10. Required environment variables

See [`.env.example`](./.env.example) for the authoritative, always-up-to-date list. Summary by concern:

- **App**: `APP_ENV`
- **Database**: `DATABASE_URL`
- **Queue/cache**: `REDIS_URL`
- **Transcript provider**: `SUPADATA_API_KEY`
- **LLM provider**: `LLM_API_KEY`, `LLM_MODEL`
- **Embeddings**: `EMBEDDING_MODEL`, `VECTOR_DIMENSION`
- **Chunking** (§21): `CHUNK_TARGET_TOKENS`, `CHUNK_MIN_TOKENS`, `CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP_TOKENS`
- **Revision loop** (§37): `MAX_REVISION_ATTEMPTS`
- **Admin auth** (§66): `ADMIN_AUTH_SECRET`
- **Storage** (§9, if object storage is enabled beyond local disk): `STORAGE_PROVIDER`, plus provider-specific credentials
- **Frontend**: `NEXT_PUBLIC_API_BASE_URL`

Secrets are never read by, or embedded in, frontend code (§54, §66) — the browser only ever talks to the FastAPI backend.

---

## 11. Expected core dependencies

These are not yet pinned (no `pyproject.toml`/`package.json` exist); listed here so Phase 1 scaffolding has a concrete target.

**Backend**: `fastapi`, `uvicorn`, `sqlalchemy` (async), `alembic`, `psycopg` (or `asyncpg`), `pgvector`, `pydantic`/`pydantic-settings`, `redis`, an async task runner appropriate for Python (e.g. `arq` or `celery` — decide during Phase 1 based on the async-first requirement in §9/§57), `langgraph`, `langchain` (only where it simplifies a specific integration, per §9), an LLM SDK, `httpx`, `pytest`/`pytest-asyncio`.

**Frontend**: `next`, `react`, `typescript`, `tailwindcss`, a typed API client layer, a test runner (e.g. `vitest`/`playwright`).

**Infra**: `postgres` (pgvector-enabled image), `redis`, via `docker-compose.yml`.

---

## 12. MVP build order

This mirrors §97 and §101–105; each milestone is only started once the previous one is reliable.

1. **Phase 1 — Foundation** *(this PR's docs land here; code comes in a follow-up)*: repo scaffolding, Docker Compose, FastAPI skeleton with health check, Next.js skeleton, env config, no business logic yet.
2. **Phase 2 — Ingestion** (§101, first functional milestone): YouTube URL validation → `SupadataTranscriptProvider` → metadata + transcript normalization → persisted to Postgres → admin can inspect the raw transcript. No AI processing yet.
3. **Phase 3 — Processing** (§102): cleaning, semantic chunking, embeddings, chunk analysis.
4. **Phase 4 — Intelligence** (§103): conversation map → article plan → grounded section generation.
5. **Phase 5 — Verification** (§104): claim extraction, fidelity verification, revision loop.
6. **Phase 6 — UI** (§105): homepage, library, article page, admin dashboard, processing status.
7. **Phase 7 — Q&A**: episode-filtered RAG question answering with timestamp citations.
8. **Phase 8 — Evaluation**: evaluation dataset, metrics, regression tests, LLM cost tracking.

Definition of done for the MVP as a whole is the checklist in §106 of the product spec.

---

## 13. Open architectural decisions (to resolve when their phase starts)

- **Background task runner**: `arq` (Redis-native, async-first, lighter) vs `celery` (more mature, broker-flexible). Lean `arq` given the stack is already async FastAPI + Redis, but not yet decided in code.
- **Object storage implementation**: local filesystem for development vs an S3-compatible provider; both should sit behind `StorageProvider` so the choice is not load-bearing elsewhere.
- **Admin auth mechanism**: simplest viable (e.g. a single admin credential + session/JWT) vs a fuller auth provider — V1 only needs to satisfy §66's "authentication for admin routes," not a multi-user system.

These are called out rather than pre-decided because §98 instructs explaining *why* before making an architectural change, and none of them are forced by Phase 1 scaffolding.
