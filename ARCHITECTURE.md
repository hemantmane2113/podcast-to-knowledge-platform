# Architecture

This document translates [`PRODUCT_SPEC.md`](./PRODUCT_SPEC.md) into concrete engineering decisions: the repository layout, the module boundaries, the data model, the processing pipeline, required configuration, and the build order. It is derived from the spec and should stay in sync with it — if they disagree, the spec is the source of truth for product intent and this file should be updated to match.

## Status convention

Every section below is written in the target/final tense (this is architecture, not a changelog), but each component is explicitly tagged so this document can never be mistaken for describing more than actually exists:

- **✅ IMPLEMENTED** — exists in the repo today, has tests, has been run against a real database/queue (not just unit-mocked).
- **🚧 PARTIAL** — some of it exists (e.g. the interface but not every implementation).
- **⬜ PLANNED** — described here for context on where it's headed; no code yet.

As of this revision: **Phase 1 (Foundation)** and **Phase 2 (Transcript Ingestion)** are ✅ IMPLEMENTED. Everything from Phase 3 onward (cleaning, chunking, embeddings, article generation, verification, RAG, LangGraph, the public reading UI, the real admin dashboard) is ⬜ PLANNED.

---

## 1. Guiding constraints (from the spec)

- **Modular monolith, not microservices** (§7, §91). One FastAPI backend with clear internal module boundaries; one Next.js frontend. ✅
- **No premature infrastructure**: no Kubernetes, no dedicated vector DB, no multi-provider abstraction gold-plating, no Elasticsearch (§7, §69, §92). ✅
- **Every external dependency is abstracted** behind a provider interface (§11, §64, §65, §93): `TranscriptProvider` ✅, `LLMProvider` ⬜, `EmbeddingProvider` ⬜, `StorageProvider` ⬜.
- **The raw transcript is the source of truth** (§18, §98). Cleaning never overwrites it; generation must be grounded in it; nothing is ever invented (§88). ✅ for ingestion (transcript persisted as returned by the provider, speaker labels never fabricated); cleaning itself is ⬜.
- **Timestamps and speaker labels survive the entire pipeline** (§14, §19, §98). ✅ `start_ms`/`duration_ms`/`speaker` persisted per segment, kept separate from database timestamps (see §5). The YouTube-timestamp-link UI itself is ⬜ (no reading UI yet).
- **Everything long-running is async and resumable** (§57, §75, §76). ✅ for ingestion: it's a background job from the start (see §6), and a crash mid-job leaves no partial transcript (single transaction) with the episode/job recoverable in a clear FAILED state, not silently stuck.
- **Human review gates publishing** (§77, §96). ⬜ — nothing to gate yet; there's no article/publishing concept in the schema.

---

## 2. Repository structure

```text
podcast-to-knowledge-platform/
├── README.md
├── PRODUCT_SPEC.md
├── ARCHITECTURE.md
├── docker-compose.yml                 ✅
├── .env.example                       ✅
├── .gitignore                         ✅
├── .github/workflows/ci.yml           ✅ backend pytest, frontend build+audit
│
├── backend/
│   ├── pyproject.toml                 ✅
│   ├── alembic.ini, alembic/          ✅ one migration (episodes/transcripts/transcript_segments/processing_jobs)
│   ├── app/
│   │   ├── main.py                    ✅ FastAPI app factory, router + error-handler registration
│   │   ├── api/
│   │   │   ├── deps.py                ✅ DB session / job queue / service DI
│   │   │   └── v1/
│   │   │       ├── health.py          ✅
│   │   │       ├── episodes.py        ✅ POST /episodes, GET /episodes/{id}, GET /episodes/{id}/transcript
│   │   │       ├── articles.py        ⬜
│   │   │       ├── qa.py              ⬜
│   │   │       └── admin/             ⬜
│   │   ├── core/
│   │   │   ├── db.py                  ✅ async engine/session (process-wide, importable from API + worker)
│   │   │   ├── exceptions.py          ✅ AppError hierarchy (see §7)
│   │   │   └── error_handlers.py      ✅ maps AppError -> consistent JSON error body
│   │   ├── config/settings.py         ✅ pydantic-settings, fails fast outside development (see §10)
│   │   ├── models/                    ✅ Episode, Transcript, TranscriptSegment, ProcessingJob (see §5)
│   │   ├── schemas/                   ✅ transcript.py (provider-agnostic normalized shape), episode.py (API I/O)
│   │   ├── repositories/              ✅ EpisodeRepository, TranscriptRepository, ProcessingJobRepository
│   │   ├── services/                  ✅ EpisodeService (create/idempotency/reads), IngestionService (worker-side), JobQueue (arq wrapper)
│   │   ├── providers/
│   │   │   ├── transcript/            ✅ TranscriptProvider ABC, SupadataTranscriptProvider
│   │   │   ├── llm/                   ⬜
│   │   │   ├── embedding/             ⬜
│   │   │   └── storage/               ⬜
│   │   ├── worker/                    ✅ arq WorkerSettings + the one ingestion task (see §6)
│   │   ├── workflows/                 ⬜ LangGraph graphs (Phase 4+)
│   │   ├── prompts/                   ⬜
│   │   ├── evaluation/                ⬜
│   │   └── utils/youtube.py           ✅ extract_video_id + InvalidYouTubeURLError
│   │
│   └── tests/
│       ├── unit/                      ✅ pure-function tests (URL parsing, Settings)
│       ├── integration/               ✅ real Postgres (via a real local/CI Postgres, not mocks) + respx-mocked Supadata
│       └── workflow/                  ⬜ (meaningful once workflows/ exists)
│
├── frontend/
│   ├── package.json                   ✅
│   ├── app/
│   │   ├── page.tsx                   ✅ placeholder homepage
│   │   ├── api/health/route.ts        ✅
│   │   └── dev/ingest/page.tsx        ✅ tiny internal tool to manually trigger/inspect ingestion — NOT the Phase 6 admin dashboard
│   ├── components/                    ⬜ (ArticleHeader, TimestampLink, KeyTakeaways, PodcastQA, ...)
│   └── lib/                           ⬜ (typed API client)
│
├── evaluation/                        ⬜
│
├── docs/
│   └── adr/                           ⬜
│
└── scripts/                           ⬜
```

This matches §73 with the admin note from the previous revision superseded: `/admin` (the real, styled admin dashboard) is still Phase 6 and doesn't exist. `frontend/app/dev/ingest` is a deliberately minimal, separate, temporary tool (PRODUCT_SPEC.md §100's "vertical slice" instruction and the Phase 2 task's explicit "keep it extremely small, do not build the final product design yet") — it should be deleted or folded into the real `/admin` once Phase 6 builds it, not grown in place.

---

## 3. Backend module responsibilities

| Module | Responsibility | Status | Must NOT contain |
|---|---|---|---|
| `api/` | HTTP request/response, validation, status codes | ✅ (episodes only) | business logic, direct DB/provider calls |
| `core/` | logging, exceptions, error handling, DB engine | ✅ | feature-specific logic |
| `config/` | typed `Settings` object loaded from env vars | ✅ | secrets committed as literals |
| `models/` | SQLAlchemy table definitions | ✅ (4 of ~12 eventual entities) | query logic beyond relationships |
| `schemas/` | Pydantic I/O and provider-normalized schemas (§63) | ✅ | ORM concerns |
| `repositories/` | CRUD + queries per entity | ✅ | business rules |
| `services/` | use-case orchestration | ✅ (`EpisodeService`, `IngestionService`) | raw SQL, raw HTTP calls to providers |
| `providers/` | adapters implementing the abstract provider interfaces (§11) | 🚧 (transcript only) | business rules |
| `worker/` | arq task definitions + WorkerSettings | ✅ | HTTP concerns |
| `workflows/` | LangGraph graph definitions + per-stage node functions (§38–39) | ⬜ | HTTP concerns |
| `prompts/` | prompt templates, one subfolder per stage, versioned (§62) | ⬜ | logic |
| `evaluation/` | fidelity/coverage/compression metrics (§60) | ⬜ | — |

Rule of thumb from §94, still honored: LangGraph is used only for the stateful AI pipeline, never for CRUD, auth, or plain API endpoints. The Phase 2 ingestion job is a single external call + normalize + persist — deliberately plain async code (see §6), not a graph; introducing LangGraph here would be exactly the "unnecessary infrastructure" the spec warns against.

---

## 4. Provider abstractions (§11, §64, §65, §93)

```python
class TranscriptProvider(ABC):                              # ✅ IMPLEMENTED
    async def get_metadata(self, video_url: str) -> EpisodeMetadata: ...
    async def get_transcript(self, video_url: str) -> NormalizedTranscript: ...

class LLMProvider(ABC): ...                                  # ⬜ PLANNED (Phase 4)
class EmbeddingProvider(ABC): ...                             # ⬜ PLANNED (Phase 3)
class StorageProvider(ABC): ...                                # ⬜ PLANNED (not yet needed — nothing binary to store)
```

`TranscriptProvider` differs from the original sketch in one way: it has two methods (`get_metadata` + `get_transcript`), not one — Supadata exposes metadata and transcript as separate endpoints, and the Episode-metadata requirement (spec §15/task item 8) needs somewhere to live that isn't transcript-shaped.

`SupadataTranscriptProvider` (`app/providers/transcript/supadata.py`) — ✅ implemented against Supadata's documented `/transcript` and `/metadata` endpoints (`x-api-key` auth, `mode=native` to avoid AI-generated captions per §88, transparent 202/job-polling for large videos). One caveat, stated plainly in the module docstring: Supadata's own docs site was not directly fetchable from this build environment (network policy blocked it), so the exact response shape was assembled from third-party summaries of the documented behavior. Parsing is defensive (`MalformedProviderResponseError` on anything unexpected) precisely because of that uncertainty — verify against a live call or the current docs before pointing this at a production key.

`services/` and `worker/` depend only on the abstract `TranscriptProvider` interface — `IngestionService` takes a `TranscriptProvider` in its constructor and has never imported `SupadataTranscriptProvider` or `httpx` directly; tests inject a `StubProvider` (`tests/fakes.py`) instead.

---

## 5. Data model (§16)

```text
Episode                                            ✅
   ├── Transcript (1:1)                            ✅
   │       └── TranscriptSegment (ordered)         ✅
   └── ProcessingJob (1:many)                      ✅

--- everything below is ⬜ PLANNED, Phase 3+ ---

Podcast → Episode   (see note below — not yet introduced)
Episode
   ├── Chunk
   │       └── ChunkAnalysis
   ├── Topic (via EpisodeTopic)
   ├── EvaluationResult
   └── Article
           ├── ArticleSection
           └── ArticleCitation → TranscriptSegment
```

### Deviations from the original sketch (and why)

- **No `Podcast` entity yet.** The previous revision of this document put `Podcast` above `Episode` as the root of the hierarchy. Per explicit instruction when Phase 2 was scoped ("a Podcast entity is optional at this stage; do not introduce it unless it is genuinely useful for the current ingestion flow"), it's been deferred — nothing in ingestion needs to group episodes by podcast/channel yet. Channel metadata (`channel_name`, `channel_id`, `thumbnail_url`) lives directly on `Episode` instead. Introduce `Podcast` when something actually needs it — most likely Phase 6 (browsing/filtering episodes by podcast) — as a genuine schema migration then, not speculative scaffolding now.
- **`Transcript` is 1:1 with `Episode`, no `is_raw` split.** The previous revision anticipated a raw/cleaned pair per §18. Phase 2 doesn't clean anything, so there's exactly one transcript row per episode — the provider's normalized output, verbatim. The raw/cleaned split becomes real schema work in Phase 3 when `clean_transcript` is implemented (most likely: either an `is_raw` flag reintroduced here, or a distinct table — decide then, informed by how cleaning actually needs to read/write).

### IDs and timestamps (locked decisions)

- **Every table uses a UUID primary key** (`uuid.uuid4()`, assigned application-side so it's available before flush — see `app/models/base.py::UUIDPrimaryKeyMixin`). Never an auto-increment integer, and never a natural/external identifier as the PK.
- **`Episode.youtube_video_id`** is the YouTube video ID, stored as a separate, uniquely-indexed column — never the primary key. It's what idempotency is keyed on (§51; see §6).
- **`created_at`/`updated_at`** (`app/models/base.py::TimestampMixin`) are always `timestamptz`, `server_default=now()`, UTC. Transcript timing (`start_ms`, `duration_ms`) is a completely separate concept — position within the source media — and is never coerced into a datetime column.

### Current fields

- **Episode**: `id`, `youtube_video_id` (unique, indexed), `youtube_url`, `title`/`description`/`channel_name`/`channel_id`/`thumbnail_url`/`duration_seconds`/`published_at`/`language` (all nullable — populated from provider metadata, never fabricated), `status` (`ProcessingStatus` enum — see below), `last_error`, `created_at`, `updated_at`.
- **Transcript**: `id`, `episode_id` (FK, unique — enforces 1:1), `language`, `created_at`, `updated_at`.
- **TranscriptSegment**: `id`, `transcript_id` (FK, indexed), `sequence_number`, `text`, `start_ms`, `duration_ms`, `speaker` (nullable, never invented), `created_at`. Unique constraint on `(transcript_id, sequence_number)` — this is both the ordering guarantee and the query index.
- **ProcessingJob**: `id`, `episode_id` (FK, indexed), `job_type` (currently only `TRANSCRIPT_INGESTION`), `status` (`PENDING`/`RUNNING`/`COMPLETED`/`FAILED`, indexed), `started_at`, `completed_at`, `error_message` (safe/sanitized — see §7), `created_at`, `updated_at`.

### Processing status enum (§17)

The full spec vocabulary is declared in `app/models/episode.py::ProcessingStatus` up front (so the Postgres enum type never needs an `ALTER TYPE` migration per future phase), but Phase 2 only ever produces three of its values:

```
INGESTING → TRANSCRIPT_FETCHED
          ↘ FAILED
```

`CLEANING` through `PUBLISHED` are ⬜, reserved for the phases that set them.

---

## 6. Ingestion pipeline — async job architecture (locked decision)

This is the concrete implementation of §57 for the one pipeline that exists today.

```
POST /api/v1/episodes {youtube_url}
        │
        ▼
extract_video_id(url)                    -- app/utils/youtube.py, raises InvalidYouTubeURLError (400)
        │
        ▼
EpisodeService.create_episode             -- app/services/episode_service.py
        │
        ├─ youtube_video_id already exists?
        │     ├─ still INGESTING with an active job → EpisodeAlreadyProcessingError (409)
        │     └─ terminal (TRANSCRIPT_FETCHED/FAILED) → return existing episode as-is (200)
        │
        └─ new: create Episode(status=INGESTING) + ProcessingJob(status=PENDING)
              in one transaction, commit
                  │
                  ├─ IntegrityError (lost a race on the unique constraint)?
                  │     → rollback, re-fetch the winner, return it (still no duplicate)
                  │
                  └─ enqueue_transcript_ingestion(episode_id, job_id) via arq/Redis
                        → API returns 201 immediately (never waits for Supadata)


arq worker (app/worker/tasks.py::ingest_episode_transcript), separate process
        │
        ▼
mark job RUNNING
        │
        ▼
IngestionService.run  -- app/services/ingestion_service.py
        │
        ├─ provider.get_transcript(youtube_url)   -- the load-bearing call
        ├─ provider.get_metadata(youtube_url)      -- best-effort; failure here doesn't fail the job
        │
        ▼
persist Transcript + all TranscriptSegments + Episode metadata + Episode.status=TRANSCRIPT_FETCHED
+ ProcessingJob.status=COMPLETED, all in ONE transaction/commit
        │
        ▼ (on any exception instead)
rollback, then:
  - non-retryable (bad URL/auth/not-found/malformed) → mark FAILED immediately, don't re-raise
  - retryable, attempts remain → re-raise so arq retries (max 3 attempts total)
  - retryable, attempts exhausted → mark FAILED, don't re-raise
```

Why this differs from the original §7/§56 sketch (a separate `POST /episodes/{id}/process` call): the Phase 2 task specified the create-and-enqueue flow as a single step (`POST /episodes` → validate → create episode → create job → return immediately), and there's currently exactly one job type, so a second endpoint to "start" it would be a distinction without a difference. `POST /episodes/{id}/process` is still the right shape for explicit reprocessing (§76) — reintroduce it when Phase 3+ needs to re-run a *specific* stage on an already-ingested episode.

**Queue**: arq (resolves the open `arq` vs `celery` decision from the previous revision) — async-native, Redis-backed, matches the stack already in docker-compose. `app/worker/settings.py::WorkerSettings` registers the one task; `on_startup`/`on_shutdown` hooks build/tear down one shared `SupadataTranscriptProvider` per worker process (in `ctx["provider"]`), not one per job. If `SUPADATA_API_KEY` is unset, `startup` logs a warning and leaves `ctx["provider"] = None` rather than crashing the worker process — every job then fails fast and individually with a clear message instead of crash-looping the whole worker (verified live: the worker stays up and marks jobs FAILED with "Transcript provider is not configured").

**Retry** (§16 of the task, §50 of the spec): `MAX_TRIES = 3` (`app/worker/tasks.py`). Retryability is a property on the exception itself (`AppError.retryable`, default `True`; `False` for `TranscriptProviderAuthError`, `TranscriptProviderNotFoundError`, `MalformedProviderResponseError`, `InvalidYouTubeURLError`) — retrying an invalid URL or bad API key wastes attempts and never changes the outcome.

**Transaction boundaries** (§17 of the task, §75 of the spec): exactly one commit for the success path (transcript + all segments + episode + job land together or none do) and exactly one commit for the terminal-failure path (rollback first, discarding any partial work-in-progress, then write the failure state). Verified with a dedicated test (`test_run_transcript_failure_raises_and_leaves_no_partial_transcript`) and live (killing ingestion mid-flight never leaves an orphaned `Transcript` row).

---

## 7. Error model (§23 of the Phase 2 task)

`app/core/exceptions.py` defines one hierarchy (`AppError` base, `code: str` + `status_code: int` on every subclass) shared by the API and the worker; `app/core/error_handlers.py` registers a single FastAPI exception handler that turns any `AppError` into `{"code": ..., "message": ...}` at the right status code, and a catch-all handler that turns anything else into a generic `500 INTERNAL_ERROR` (never leaking a stack trace or exception internals to the client).

| Status | Code | Raised when |
|---|---|---|
| 400 | `INVALID_YOUTUBE_URL` | URL doesn't match a recognized YouTube form |
| 404 | `EPISODE_NOT_FOUND` | no episode with that ID |
| 404 | `TRANSCRIPT_NOT_FOUND` | episode exists but has no transcript yet |
| 409 | `EPISODE_ALREADY_PROCESSING` | a job for this video is already running |
| 502 | `TRANSCRIPT_PROVIDER_ERROR` | Supadata failed (and subclasses for auth/rate-limit/not-found/malformed — these currently surface as `Episode.last_error`/`ProcessingJob.error_message` rather than a live HTTP response, since ingestion is async; the hierarchy exists so a future synchronous "retry now" endpoint has somewhere to raise into) |
| 500 | `INTERNAL_ERROR` | anything unexpected |

Worker error messages are always the exception's own `.message` (text we wrote, never a provider's raw response or a credential) or a fixed generic string for anything else — see `app/worker/tasks.py::_safe_message`.

---

## 8. API surface (§55–56)

```
POST   /api/v1/episodes                       ✅ create/idempotently return an episode; enqueues ingestion
GET    /api/v1/episodes/{id}                   ✅
GET    /api/v1/episodes/{id}/transcript        ✅
POST   /api/v1/episodes/{id}/process           ⬜ (see §6 — not needed until explicit reprocessing exists)
GET    /api/v1/episodes/{id}/article           ⬜
POST   /api/v1/episodes/{id}/qa                ⬜
```

Admin auth (§66) is still ⬜ — every endpoint above is currently unauthenticated. `ADMIN_AUTH_SECRET` is declared in `Settings`/`.env.example` and enforced as a *required, non-empty config value outside development* (see §10), but nothing reads it yet to actually gate a request. This is safe today only because there's no admin-only mutation distinct from the public surface (there is no public surface yet either) — closing this gap is a prerequisite for Phase 6, not before.

---

## 9. Frontend structure

- Next.js App Router, TypeScript, Tailwind (§9). ✅ scaffolding, 🚧 content.
- `/` — placeholder homepage. ✅
- `/dev/ingest` — tiny internal tool (URL input, Ingest button, polls episode status) to exercise the Phase 2 API without curl. ✅ Explicitly not the Phase 6 admin dashboard — see §2.
- Public routes (`/articles/[slug]`, library/search) — ⬜, nothing to render (no Article model yet).
- Real admin routes (`/admin/*`, §48) — ⬜.
- Shared components (§45) — ⬜.

---

## 10. Local development

```bash
cp .env.example .env
docker compose up      # postgres (pgvector), redis, backend, worker, frontend
```

`docker-compose.yml` now includes a `worker` service (`arq app.worker.settings.WorkerSettings`) alongside `backend`/`frontend`/`postgres`/`redis`. Backend/worker share the same image and bind mount; the backend container runs `uvicorn --reload` (bind-mount-friendly local dev, per Phase 1.1).

Outside Docker:

```bash
cd backend && pip install -e ".[dev]"
alembic upgrade head
uvicorn app.main:app --reload      # API
arq app.worker.settings.WorkerSettings   # worker, separate process
pytest                             # needs a running Postgres — see tests/conftest.py
```

---

## 11. Configuration (§53–54)

See [`.env.example`](./.env.example) for the authoritative list. `app/config/settings.py` (`Settings`, pydantic-settings) now does more than load values:

- **Fails fast outside development.** A `model_validator` requires `SUPADATA_API_KEY`, `LLM_API_KEY`, and `ADMIN_AUTH_SECRET` to be non-empty whenever `APP_ENV != "development"` — a misconfigured non-dev deploy refuses to boot instead of silently running with empty secrets. `APP_ENV=development` (the default) never requires them, so the dev inner loop stays credential-free.
- **Never exposes secrets to the frontend.** The only backend-configured value the frontend ever sees is `NEXT_PUBLIC_API_BASE_URL` (not a secret) — verified structurally: nothing in `frontend/` imports or proxies any other env var, and Next.js itself only ever bundles `NEXT_PUBLIC_*` into client code.

---

## 12. Core dependencies

**Backend** (`backend/pyproject.toml`): `fastapi`, `uvicorn`, `sqlalchemy[asyncio]`, `asyncpg`, `alembic` ✅, `pydantic`/`pydantic-settings`, `redis`, `arq` ✅ (resolves the task-runner decision), `httpx` (also the Supadata client). Dev-only: `pytest`, `pytest-asyncio`, `respx` ✅ (HTTP mocking for the provider tests). Not yet added: `pgvector` (Python client — needed once embeddings exist, Phase 3), `langgraph`/`langchain`, an LLM SDK.

**Frontend** (`frontend/package.json`): `next@16`, `react@19`/`react-dom@19`, `typescript`, `tailwindcss` — all still auditing clean (`npm audit` → 0 vulnerabilities is a bar for landing any dependency change, not just a nice-to-have). A typed API client layer and a test runner are still ⬜.

**Infra**: `pgvector/pgvector:pg16`, `redis:7-alpine`, via `docker-compose.yml`. The pgvector extension itself is not yet `CREATE EXTENSION`'d anywhere (no migration needs it yet — no vector columns exist until Phase 3).

**CI**: `.github/workflows/ci.yml` ✅ — backend job runs migrations against a real Postgres service container then `pytest`; frontend job runs `npm ci && npm run build && npm audit`. Both must pass for a PR to be considered green.

---

## 13. MVP build order

This mirrors §97 and §101–105; each milestone is only started once the previous one is reliable.

1. **Phase 1 — Foundation** ✅: repo scaffolding, Docker Compose, FastAPI skeleton with health check, Next.js skeleton, env config, CI, fail-fast settings.
2. **Phase 2 — Ingestion** ✅ (§101, first functional milestone): YouTube URL validation → `SupadataTranscriptProvider` → metadata + transcript normalization → persisted to Postgres via an async job → queryable through the API. No AI processing.
3. **Phase 3 — Processing** ⬜ (§102): cleaning, semantic chunking, embeddings, chunk analysis. First phase that needs pgvector and a `Chunk`/`ChunkAnalysis` schema.
4. **Phase 4 — Intelligence** ⬜ (§103): conversation map → article plan → grounded section generation. First phase that needs `LLMProvider` and LangGraph.
5. **Phase 5 — Verification** ⬜ (§104): claim extraction, fidelity verification, revision loop.
6. **Phase 6 — UI** ⬜ (§105): homepage content, library, article page, the real admin dashboard, processing status UI. `frontend/app/dev/ingest` gets replaced, not extended.
7. **Phase 7 — Q&A** ⬜: episode-filtered RAG question answering with timestamp citations.
8. **Phase 8 — Evaluation** ⬜: evaluation dataset, metrics, regression tests, LLM cost tracking.

Definition of done for the MVP as a whole is the checklist in §106 of the product spec.

---

## 14. Open architectural decisions (to resolve when their phase starts)

- **Object storage implementation**: local filesystem for development vs an S3-compatible provider — `StorageProvider` still doesn't exist because nothing binary needs storing yet (transcripts are text, in Postgres).
- **Admin auth mechanism**: simplest viable (a single admin credential + session/JWT) vs a fuller auth provider. `ADMIN_AUTH_SECRET` is reserved and required outside dev, but unused — needs an actual decision before Phase 6 adds admin-only mutations.
- **Raw vs. cleaned transcript storage shape**: deferred to Phase 3 (see §5) — decide once `clean_transcript` exists and its actual read/write pattern is known, rather than guessing now.
- **`Podcast` entity**: deferred (see §5) until something needs to group episodes by channel/podcast.

These are called out rather than pre-decided because §98 instructs explaining *why* before making an architectural change, and none of them are forced by Phase 2.
