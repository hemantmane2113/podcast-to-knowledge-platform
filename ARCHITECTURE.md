# Architecture

This document translates [`PRODUCT_SPEC.md`](./PRODUCT_SPEC.md) into concrete engineering decisions: the repository layout, the module boundaries, the data model, the processing pipeline, required configuration, and the build order. It is derived from the spec and should stay in sync with it — if they disagree, the spec is the source of truth for product intent and this file should be updated to match.

## Status convention

Every section below is written in the target/final tense (this is architecture, not a changelog), but each component is explicitly tagged so this document can never be mistaken for describing more than actually exists:

- **✅ IMPLEMENTED** — exists in the repo today, has tests, has been run against a real database/queue (not just unit-mocked).
- **🚧 PARTIAL** — some of it exists (e.g. the interface but not every implementation).
- **⬜ PLANNED** — described here for context on where it's headed; no code yet.

As of this revision: **Phase 1 (Foundation)**, **Phase 2 (Transcript Ingestion)**, **Phase 3A (Transcript Cleaning + Chunking)**, and the **V1 AI pipeline (Phases C-H — topic analysis through a reviewable article, §17 below)** are ✅ IMPLEMENTED. pgvector/embeddings-based retrieval is deliberately **not** built (§17.7) — the architecture supports adding it later where it provides a clear benefit, but nothing in V1 needs it. The real editorial CMS and public reading experience are ⬜ PLANNED.

**Important caveat on real-provider validation**: this sandboxed build environment's network egress policy blocks `api.supadata.ai`, `api.groq.com`, and `api.openai.com` (confirmed directly). Phase 2/3A ingestion, cleaning, and chunking *have* been validated against a real Supadata transcript (a real ~2h47m podcast, via `scripts/validate_real_transcript.py` run from an unrestricted environment — see that script and the README status line for the numbers). The AI pipeline (§17) has been built and thoroughly tested against a `FakeLLMProvider` (`tests/fakes.py`) exercising the real persistence path end-to-end, but **no real LLM call has been made from this environment** — `tests/integration/test_article_pipeline_smoke.py` is the explicit, skip-by-default test to run that validation from an environment that can reach a real LLM provider.

---

## 1. Guiding constraints (from the spec)

- **Modular monolith, not microservices** (§7, §91). One FastAPI backend with clear internal module boundaries; one Next.js frontend. ✅
- **No premature infrastructure**: no Kubernetes, no dedicated vector DB, no multi-provider abstraction gold-plating, no Elasticsearch (§7, §69, §92). ✅
- **Every external dependency is abstracted** behind a provider interface (§11, §64, §65, §93): `TranscriptProvider` ✅, `LLMProvider` ⬜, `EmbeddingProvider` ⬜, `StorageProvider` ⬜.
- **The raw transcript is the source of truth** (§18, §98). ✅ `TranscriptSegment.text` (the provider's raw output) is never modified by cleaning, and cleaning's output is never persisted onto the raw table at all — `clean_transcript_segments()` returns transient in-memory `CleanedSegment`s consumed directly by the chunker (§5, §7). Nothing invented: speaker labels are never fabricated, and cleaning is deterministic, rule-based text normalization only — no LLM (§7).
- **Timestamps and speaker labels survive the entire pipeline** (§14, §19, §98). ✅ `start_ms`/`duration_ms`/`speaker` persisted per segment; every `Chunk` carries `start_ms`/`end_ms` derived from real segment boundaries (never interpolated) and a full list of the segment IDs it was built from (§7). The YouTube-timestamp-link UI itself is still ⬜ (no reading UI yet).
- **Everything long-running is async and resumable** (§57, §75, §76). ✅ for ingestion and processing: both are background jobs (§6), each a single transaction on success (no partial writes) with a clear FAILED state on failure, never silently stuck.
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
│   ├── alembic.ini, alembic/          ✅ two migrations (see §5)
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
│   │   │   ├── exceptions.py          ✅ AppError hierarchy (see §9)
│   │   │   └── error_handlers.py      ✅ maps AppError -> consistent JSON error body
│   │   ├── config/settings.py         ✅ pydantic-settings, fails fast outside development (see §13)
│   │   ├── models/                    ✅ Episode, Transcript, TranscriptSegment, Chunk, ProcessingJob (see §5)
│   │   ├── schemas/                   ✅ transcript.py (provider-agnostic normalized shape), episode.py (API I/O)
│   │   ├── repositories/              ✅ Episode/Transcript/Chunk/ProcessingJob repositories
│   │   ├── services/                  ✅ EpisodeService, IngestionService, TranscriptProcessingService, CleaningService (module), ChunkingService (module), JobQueue
│   │   ├── providers/
│   │   │   ├── transcript/            ✅ TranscriptProvider ABC, SupadataTranscriptProvider
│   │   │   ├── llm/                   ⬜
│   │   │   ├── embedding/             ⬜ deliberately not introduced yet — see §8
│   │   │   └── storage/               ⬜
│   │   ├── worker/                    ✅ arq WorkerSettings + ingest_episode_transcript, process_transcript (see §6)
│   │   ├── workflows/                 ⬜ LangGraph graphs (Phase 4+)
│   │   ├── prompts/                   ⬜
│   │   ├── evaluation/                ⬜
│   │   └── utils/youtube.py           ✅ extract_video_id + InvalidYouTubeURLError
│   │
│   ├── scripts/
│   │   └── inspect_chunks.py          ✅ dev/debug CLI — prints a transcript's chunks (§7)
│   │
│   └── tests/
│       ├── unit/                      ✅ pure-function tests (URL parsing, Settings, cleaner, chunker)
│       ├── integration/               ✅ real Postgres (local/CI, not mocks) + respx-mocked Supadata
│       ├── fixtures/podcast_transcript.py  ✅ realistic synthetic long-form transcript fixture (§7)
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
└── scripts/                           ⬜ (repo-root automation — distinct from backend/scripts/, still empty)
```

This matches §73 with the admin note from the Phase 2 revision superseded: `/admin` (the real, styled admin dashboard) is still Phase 6 and doesn't exist. `frontend/app/dev/ingest` is a deliberately minimal, separate, temporary tool — it should be deleted or folded into the real `/admin` once Phase 6 builds it, not grown in place. `backend/scripts/inspect_chunks.py` is the Phase 3A equivalent for chunk inspection: a script, not an API endpoint, since chunk inspection is a development/QA activity, not a product feature yet.

---

## 3. Backend module responsibilities

| Module | Responsibility | Status | Must NOT contain |
|---|---|---|---|
| `api/` | HTTP request/response, validation, status codes | ✅ (episodes only) | business logic, direct DB/provider calls |
| `core/` | logging, exceptions, error handling, DB engine | ✅ | feature-specific logic |
| `config/` | typed `Settings` object loaded from env vars | ✅ | secrets committed as literals |
| `models/` | SQLAlchemy table definitions | ✅ (5 of ~12 eventual entities) | query logic beyond relationships |
| `schemas/` | Pydantic I/O and provider-normalized schemas (§63) | ✅ | ORM concerns |
| `repositories/` | CRUD + queries per entity | ✅ | business rules |
| `services/` | use-case orchestration + the cleaning/chunking algorithms | ✅ | raw SQL, raw HTTP calls to providers |
| `providers/` | adapters implementing the abstract provider interfaces (§11) | 🚧 (transcript only) | business rules |
| `worker/` | arq task definitions + WorkerSettings | ✅ | HTTP concerns |
| `workflows/` | LangGraph graph definitions + per-stage node functions (§38–39) | ⬜ | HTTP concerns |
| `prompts/` | prompt templates, one subfolder per stage, versioned (§62) | ⬜ | logic |
| `evaluation/` | fidelity/coverage/compression metrics (§60) | ⬜ | — |

The cleaner and chunker (`app/services/cleaning_service.py`, `app/services/chunking_service.py`) live in `services/` rather than a new top-level module: they're pure algorithms over already-loaded data with no I/O of their own, the same shape as the rest of `services/`, and inventing a `processing/` package for two files would be exactly the kind of premature structure §98 warns against.

Rule of thumb from §94, still honored: LangGraph is used only for the stateful AI pipeline, never for CRUD, auth, or plain algorithmic steps. Cleaning + chunking is deterministic, local, synchronous-within-the-job code — deliberately not a graph.

---

## 4. Provider abstractions (§11, §64, §65, §93)

```python
class TranscriptProvider(ABC):                              # ✅ IMPLEMENTED
    async def get_metadata(self, video_url: str) -> EpisodeMetadata: ...
    async def get_transcript(self, video_url: str) -> NormalizedTranscript: ...

class LLMProvider(ABC):                                       # ✅ IMPLEMENTED (§17.2)
    async def generate(self, *, messages, system=None, temperature=0.2, max_tokens=None) -> LLMTextResponse: ...
    async def generate_structured(self, *, messages, response_model, system=None, temperature=0.2, max_tokens=None): ...

class EmbeddingProvider(ABC): ...                             # ⬜ PLANNED (deliberately not introduced yet — see §17.7)
class StorageProvider(ABC): ...                                # ⬜ PLANNED (not yet needed — nothing binary to store)
```

`TranscriptProvider` differs from the original sketch in one way: it has two methods (`get_metadata` + `get_transcript`), not one — Supadata exposes metadata and transcript as separate endpoints, and the Episode-metadata requirement (spec §15/task item 8) needs somewhere to live that isn't transcript-shaped.

`SupadataTranscriptProvider` (`app/providers/transcript/supadata.py`) — ✅ implemented against Supadata's documented `/transcript` and `/metadata` endpoints (`x-api-key` auth, transparent 202/job-polling for large videos). **Correction history**: initially implemented with `mode=native` (only return transcripts that already exist as real captions, reasoned as closer to §88 "never invent"); corrected to `mode=auto` per explicit direction — `mode=auto` asks for native captions first and falls back to a generated transcript when a video has none, trading a small amount of that purity for actually getting a transcript on videos without official captions. `text=false` is non-negotiable either way — ingestion needs timestamped segments, not a plain string. One caveat, stated plainly in the module docstring: Supadata's own docs site was not directly fetchable from this build environment (network policy blocked it), so the exact response shape was assembled from third-party summaries of the documented behavior, not verified against a real call (see the top-of-document caveat). Parsing is defensive (`MalformedProviderResponseError` on anything unexpected) precisely because of that uncertainty.

`services/` and `worker/` depend only on the abstract `TranscriptProvider` interface — `IngestionService` takes a `TranscriptProvider` in its constructor and has never imported `SupadataTranscriptProvider` or `httpx` directly; tests inject a `StubProvider` (`tests/fakes.py`) instead.

`LLMProvider` (`app/providers/llm/`) follows the identical pattern for inference — see §17.2 for the full writeup (three concrete providers, structured-output handling, exception mapping, and why `app/ai/` never imports `groq`/`openai` directly).

---

## 5. Data model (§16)

```text
Episode                                            ✅
   ├── Transcript (1:1)                            ✅
   │       ├── TranscriptSegment (ordered)         ✅  text (raw only — cleaning is transient, not persisted)
   │       └── Chunk (ordered)                     ✅
   └── ProcessingJob (1:many)                      ✅

--- everything below is ⬜ PLANNED, Phase 3B+ ---

Podcast → Episode   (see note below — not yet introduced)
Chunk
   └── ChunkAnalysis
Episode
   ├── Topic (via EpisodeTopic)
   ├── EvaluationResult
   └── Article
           ├── ArticleSection
           └── ArticleCitation → TranscriptSegment
```

### Deviations from the original sketch (and why)

- **No `Podcast` entity yet.** Per explicit instruction when Phase 2 was scoped ("a Podcast entity is optional at this stage; do not introduce it unless it is genuinely useful for the current ingestion flow"), it's been deferred — nothing in ingestion or processing needs to group episodes by podcast/channel yet. Channel metadata lives directly on `Episode`. Introduce `Podcast` when something actually needs it — most likely Phase 6 (browsing/filtering by podcast).
- **No persisted cleaned representation at all — not a `cleaned_text` column, not a separate table.** The original sketch anticipated a raw/cleaned pair per §18. An earlier version of this migration added a nullable `TranscriptSegment.cleaned_text` column; on review, that mixed raw source data with derived processing state on the canonical raw-transcript table for no concrete benefit — cleaning is cheap, deterministic, and pure, so recomputing it costs nothing, while persisting it risks silent staleness if the cleaning rules ever change without a backfill. Instead, `clean_transcript_segments()` (`app/services/cleaning_service.py`) returns a list of transient, in-memory `CleanedSegment` objects (`segment_id`, `sequence_number`, `text`, `start_ms`, `duration_ms`) that the chunker consumes directly and that are never written to the database on their own — only the chunker's output (`Chunk`, with `source_segment_ids` pointing back to the raw segments) is persisted. This keeps `TranscriptSegment` write-once raw provider output, full stop.
- **`Chunk` has no join table for source-segment traceability.** `Chunk.source_segment_ids` is a plain `ARRAY(UUID)` column, not a `ChunkSegment` association table. Chunks are normally built from a contiguous range of segments, which a join table would model more "properly," but a single oversized segment can legitimately be split across two adjacent chunks (§8), so the same segment ID can appear in two chunks' arrays — an explicit array handles that with zero extra schema, a strict range (`first_segment_id`, `last_segment_id`) would not.

### IDs and timestamps (locked decisions)

- **Every table uses a UUID primary key** (`uuid.uuid4()`, assigned application-side so it's available before flush — see `app/models/base.py::UUIDPrimaryKeyMixin`). Never an auto-increment integer, and never a natural/external identifier as the PK.
- **`Episode.youtube_video_id`** is the YouTube video ID, stored as a separate, uniquely-indexed column — never the primary key. It's what idempotency is keyed on (§51; see §6).
- **`created_at`/`updated_at`** (`app/models/base.py::TimestampMixin`) are always `timestamptz`, `server_default=now()`, UTC. Transcript timing (`start_ms`, `duration_ms`, and `Chunk.start_ms`/`end_ms`) is a completely separate concept — position within the source media — and is never coerced into a datetime column.

### Current fields

- **Episode**: `id`, `youtube_video_id` (unique, indexed), `youtube_url`, `title`/`description`/`channel_name`/`channel_id`/`thumbnail_url`/`duration_seconds`/`published_at`/`language` (all nullable — populated from provider metadata, never fabricated), `status` (`ProcessingStatus` enum — see below), `last_error`, `created_at`, `updated_at`.
- **Transcript**: `id`, `episode_id` (FK, unique — enforces 1:1), `language`, `created_at`, `updated_at`.
- **TranscriptSegment**: `id`, `transcript_id` (FK, indexed), `sequence_number`, `text` (raw, write-once — no cleaned/derived column; see "Deviations" above), `start_ms`, `duration_ms`, `speaker` (nullable, never invented), `created_at`. Unique constraint on `(transcript_id, sequence_number)` — ordering guarantee + query index.
- **Chunk**: `id`, `transcript_id` (FK, indexed), `episode_id` (FK, indexed — denormalized from transcript, avoids a join for the common "chunks for this episode" query), `sequence_number`, `text`, `start_ms`, `end_ms`, `source_segment_ids` (`ARRAY(UUID)`), `token_count` (approximate — see §8), `created_at`. Unique constraint on `(transcript_id, sequence_number)`.
- **ProcessingJob**: `id`, `episode_id` (FK, indexed), `job_type` (`TRANSCRIPT_INGESTION` | `TRANSCRIPT_PROCESSING`), `status` (`PENDING`/`RUNNING`/`COMPLETED`/`FAILED`, indexed), `started_at`, `completed_at`, `error_message` (safe/sanitized — see §9), `created_at`, `updated_at`.

### Migrations

1. `78e3d5ee683d` — initial schema (Episode, Transcript, TranscriptSegment, ProcessingJob).
2. `9d7cf2188f5a` — adds `Chunk` and the `TRANSCRIPT_PROCESSING` job type (originally also added `TranscriptSegment.cleaned_text`; removed from this same unmerged migration on review — see "Deviations" above — rather than left in place and reverted by a third migration). Two things found by actually running `alembic downgrade`/`upgrade` cycles against a real Postgres, not just the forward migration: autogenerate doesn't detect new values on an existing Postgres native `ENUM` type (had to hand-add `ALTER TYPE job_type ADD VALUE`), and since there's no `ALTER TYPE ... DROP VALUE`, the add had to be written as `ADD VALUE IF NOT EXISTS` — otherwise a downgrade-then-upgrade cycle fails with "enum label already exists" on the second upgrade (the same class of bug as migration 1's enum-type-drop issue, different manifestation). Re-verified after the `cleaned_text` removal: downgrade→upgrade and a repeated downgrade/upgrade cycle both leave a clean schema with no enum corruption.

### Processing status enum (§17)

The full spec vocabulary is declared in `app/models/episode.py::ProcessingStatus` up front (so the Postgres enum type never needs an `ALTER TYPE` migration per future phase). Phase 2 + 3A produce:

```
INGESTING → TRANSCRIPT_FETCHED → CHUNKING
          ↘ FAILED (from either stage)
```

`CHUNKING` is used as the resting/terminal status once cleaning+chunking succeeds — there's no intermediate `CLEANING` write (cleaning and chunking happen in one job, one commit; see §6) and no `ANALYZING` yet since Phase 3B doesn't exist. `CLEANING`, `ANALYZING` through `PUBLISHED` are ⬜, reserved for the phases that set them.

---

## 6. Async job architecture (locked decision)

This is the concrete implementation of §57 for the two pipelines that exist today. Ingestion and processing are two separate jobs (two `ProcessingJob` rows, two arq tasks) chained automatically — ingestion enqueues processing on success — rather than one combined job, matching §94's "one node, one responsibility" even without LangGraph.

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
        │     └─ terminal (TRANSCRIPT_FETCHED/CHUNKING/FAILED) → return existing episode as-is (200)
        │
        └─ new: create Episode(status=INGESTING) + ProcessingJob(TRANSCRIPT_INGESTION, PENDING)
              in one transaction, commit
                  │
                  ├─ IntegrityError (lost a race on the unique constraint)?
                  │     → rollback, re-fetch the winner, return it (still no duplicate)
                  │
                  └─ enqueue_transcript_ingestion(episode_id, job_id) via arq/Redis
                        → API returns 201 immediately (never waits for Supadata)


arq worker (app/worker/tasks.py::ingest_episode_transcript)
        │
        ▼
mark job RUNNING
        │
        ▼
IngestionService.run
        │
        ├─ provider.get_transcript(youtube_url)   -- the load-bearing call
        ├─ provider.get_metadata(youtube_url)      -- best-effort; failure here doesn't fail the job
        │
        ▼
persist Transcript + all TranscriptSegments + Episode metadata + Episode.status=TRANSCRIPT_FETCHED
+ ProcessingJob.status=COMPLETED, all in ONE transaction/commit
        │
        ├─ success ─▶ create ProcessingJob(TRANSCRIPT_PROCESSING, PENDING), commit,
        │             enqueue_transcript_processing(episode_id, new_job_id)
        │
        ▼ (on any exception instead)
rollback, then:
  - non-retryable (bad URL/auth/not-found/malformed) → mark FAILED immediately, don't re-raise
  - retryable, attempts remain → re-raise so arq retries (max 3 attempts total)
  - retryable, attempts exhausted → mark FAILED, don't re-raise


arq worker (app/worker/tasks.py::process_transcript), auto-enqueued above
        │
        ▼
mark job RUNNING
        │
        ▼
TranscriptProcessingService.run  -- app/services/transcript_processing_service.py
        │
        ├─ clean_transcript_segments(transcript.segments)     -- returns list[CleanedSegment], no DB writes
        ├─ chunk_transcript_segments(cleaned_segments, config)  -- see §7/§8
        ├─ ChunkRepository.replace_all(...)                    -- delete-then-insert (idempotency, see §7)
        │
        ▼
Episode.status=CHUNKING + ProcessingJob.status=COMPLETED, one transaction/commit
        │
        ▼ (on any exception: same non-retryable/retryable split as ingestion)
```

Why no separate `POST /episodes/{id}/process` endpoint (differs from the original §7/§56 sketch): the Phase 2 task specified create-and-enqueue as a single step, and Phase 3A specified reusing the existing job infrastructure rather than adding a new trigger — chunking has exactly one thing to chain into after ingestion, so an API-level trigger would be a distinction without a difference. Reintroduce an explicit endpoint when Phase 3B+ needs to re-run a *specific* stage on an already-processed episode without redoing everything before it (§76).

**Queue**: arq — async-native, Redis-backed, matches the stack already in docker-compose. `app/worker/settings.py::WorkerSettings` registers both tasks; `on_startup` builds one shared `SupadataTranscriptProvider` (in `ctx["provider"]`) and reuses arq's own Redis connection for a `JobQueue` (`ctx["job_queue"]`) rather than opening a second pool. If `SUPADATA_API_KEY` is unset, `startup` logs a warning and leaves `ctx["provider"] = None` rather than crashing the worker process — every ingestion job then fails fast and individually instead of crash-looping the whole worker (verified live).

**Retry**: `MAX_TRIES = 3` for both tasks (`app/worker/tasks.py`). Retryability is a property on the exception itself — `AppError.retryable` defaults to `False` (most domain errors, like "not found," describe state that won't change on retry) and `TranscriptProviderError` flips it back to `True` for genuinely transient provider failures, with its own subclasses (auth, not-found, malformed) overriding back to `False`. Found via a real bug while adding `process_transcript`: `TranscriptNotFoundError` had no `retryable` attribute at all before this default existed, so `getattr(exc, "retryable", True)` silently treated "no transcript exists for this episode" as worth retrying — which can never succeed. Fixed by moving the default onto `AppError` itself instead of leaving it unset for every non-provider exception.

**Transaction boundaries**: exactly one commit for each job's success path and exactly one commit for its terminal-failure path (rollback first, discarding partial work). Verified with dedicated tests and live.

---

## 7. Transcript processing: cleaning + source traceability (Phase 3A)

`app/services/cleaning_service.py` — ✅ deterministic, rule-based, no LLM (§3A.3). `clean_segment_text()` handles, per segment, in isolation: whitespace/newline/tab normalization, removing space-before-punctuation, and stripping a small explicit allowlist of non-speech caption markers (`[Music]`, `[Applause]`, `[Laughter]`, `[Inaudible]`, `[Silence]`, `[Crosstalk]`) — deliberately *not* a blanket "strip anything bracketed," since arbitrary bracketed text could be real spoken/quoted content. A second function, `_dedupe_cross_segment_overlap()`, handles the one cross-segment case: auto-captions are often produced from a rolling audio window, so the tail of one caption and the head of the next can be the literal same words re-transcribed — an exact ≥3-word match is trimmed from the *following* segment. Within-segment word repetition ("no no", "very very") is deliberately left untouched — it could be real disfluency or emphasis, not an artifact, and removing it would risk exactly the "don't remove substantive content" violation the task warns against.

`clean_transcript_segments()` returns `list[CleanedSegment]` — a transient, in-memory dataclass (`segment_id`, `sequence_number`, `text`, `start_ms`, `duration_ms`) — rather than mutating or persisting anything on the `TranscriptSegment` ORM model. `chunk_transcript_segments()` (`app/services/chunking_service.py`) then consumes that list directly: `Raw TranscriptSegment → clean_transcript_segments() → CleanedSegment (in memory) → chunk_transcript_segments() → persisted Chunk`. This was a deliberate correction during Phase 3A review: cleaning is cheap and pure, so there's no benefit to persisting its output (and a real risk of it going stale relative to `text` if the cleaning rules change), so nothing sits between the raw segment and the chunker except an in-memory value.

Source traceability (§3A.2): every `Chunk.source_segment_ids` is the literal list of `TranscriptSegment.id`s its text was built from, in order, with duplicates only when a single long segment was split across two chunks (§8). This directly answers "which transcript segments produced this chunk" without a join — `backend/scripts/inspect_chunks.py` demonstrates it by mapping each chunk's IDs back to `sequence_number`s and printing them as a range (e.g. `[1..37]`).

**Idempotency** (§3A.8): `ChunkRepository.replace_all()` deletes a transcript's existing chunks and inserts the freshly computed set, in the same transaction the caller commits. Chunking is a pure function of `(cleaned segments, config)`, so there's nothing worth preserving across a rerun — this guarantees the same logical chunk set every time without ever risking a partial mix of old and new rows or a unique-constraint collision from a naive re-insert. Verified directly: running the pipeline twice against the same transcript produces the same chunk count/text/ordering (different row IDs, since they're new rows, but the same logical content).

---

## 8. Chunking strategy (Phase 3A)

`app/services/chunking_service.py::chunk_transcript_segments()` — ✅. No naive fixed-token splitting (§3A.4 explicitly forbids it) and no embeddings (§3A.9 explicitly forbids it for this phase — see below). The algorithm:

1. **Units.** Each cleaned segment becomes one "unit" of chunkable text, *except* a segment whose own text alone exceeds `CHUNK_MAX_TOKENS` — that gets pre-split (`_split_long_text`) into several units, preferring sentence boundaries (`(?<=[.!?])\s+`) and falling back to word boundaries, **never splitting a word**. All pieces of a split segment share that segment's ID and timing.
2. **Accumulate toward the target.** Walk units in order, adding one at a time to the current chunk (peeking each addition's actual joined-text token estimate first — see the "token estimation" note below — and refusing to add a unit that would push the running total over `CHUNK_MAX_TOKENS`).
3. **Boundary preference, once at or past `CHUNK_TARGET_TOKENS`:** a **pause boundary** (the gap to the next segment is ≥ `CHUNK_PAUSE_THRESHOLD_MS`) is taken immediately — it's the closest thing to a real topic/thought boundary this heuristic has. A **sentence boundary** (the unit's text ends with `.`/`!`/`?`) is only *remembered* as a fallback, and scanning keeps going, hoping for a pause boundary, until `CHUNK_MAX_TOKENS` forces a decision — at which point the remembered sentence boundary is used, or (if there wasn't one either) the chunk is cut wherever accumulation had to stop.
4. **Forced splits always end the chunk immediately** — a pre-split piece was already sized up to `CHUNK_MAX_TOKENS` on its own, so continuing past it risks exceeding the ceiling.
5. **Minimum size** (`CHUNK_MIN_TOKENS`) is never checked directly — it falls out of the above: nothing closes a chunk before `CHUNK_TARGET_TOKENS` except end-of-transcript or a forced split, and `CHUNK_MIN_TOKENS < CHUNK_TARGET_TOKENS` is enforced by `ChunkingConfig` validation. Only the transcript's trailing chunk (or a forced-split remainder) may end up smaller than `CHUNK_MIN_TOKENS`, which is expected and allowed.

### A real bug found and fixed by manual quality inspection

The first working version took the *first* sentence boundary once past target, treating it as equally good as a pause boundary. Since nearly every transcript segment ends with terminal punctuation, this meant the algorithm almost always cut at the very next period — ignoring a real pause-based topic transition that might be only a segment or two further ahead. This was caught by literally reading chunk output from `tests/fixtures/podcast_transcript.py` (§3A.14), not by a unit test — the unit tests at the time were all consistent with the buggy behavior because they were written to match it. Fixed as described in step 3 above, and now pinned by `test_prefers_pause_boundary_over_nearer_sentence_boundary` plus a dedicated fixture-based regression test. Verified against the fixture directly: a genuine 1900ms gap at the right spot is now correctly preferred over four weaker, closer sentence-only candidates.

### Token estimation

No concrete LLM/tokenizer has been chosen yet (that's Phase 4), so `estimate_tokens()` is a documented approximation (`~4 characters/token`), not a real tokenizer. It's used consistently for both the chunk-sizing decisions and the stored `token_count` field — **always measured on the actual joined text**, never summed from per-piece estimates. Summing was tried first and was a real bug: per-piece estimates miss the join-space characters and each piece's own `max(1, ...)` floor compounds across many small pieces, so a chunk built from several individually-safe pieces could still exceed `CHUNK_MAX_TOKENS` in aggregate. Caught by a unit test splitting one very long, unpunctuated segment into words and asserting every resulting piece actually stayed under the limit.

### Overlap (§3A.7)

Fully implemented and tested (`CHUNK_OVERLAP_TOKENS`, segment-granular: when a chunk closes, trailing units worth up to `CHUNK_OVERLAP_TOKENS` are re-included at the start of the next chunk), but **defaults to 0 (disabled)** — an explicit decision, not an oversight. Nothing in Phase 3A consumes overlap; its only real purpose is avoiding context loss at chunk boundaries during embedding-based similarity search, which doesn't exist until Phase 3C. Shipping it non-zero by default would just duplicate text (and duplicate `source_segment_ids` entries between adjacent chunks) for no current benefit. Set `CHUNK_OVERLAP_TOKENS` above 0 once a real consumer needs it — the mechanism doesn't need to change, only the default.

### Embeddings (§3A.9)

**Deliberately not used.** The "semantic" boundary signal here is sentence-end + pause-gap heuristics, not embedding similarity — no `EmbeddingProvider` exists (§4), no embedding calls happen anywhere in this phase, and none of the `pyproject.toml` dependencies changed to support one. This is both because no embedding model/provider has been chosen yet, and because the task explicitly scoped embeddings to Phase 3C ("do not add an embedding dependency merely because the product will eventually use RAG"). If evaluation of the heuristic approach against real content later shows it's insufficient, embedding-based boundary detection can be added behind `chunk_transcript_segments()`'s existing signature without changing callers.

### Configuration

All four values live in `Settings`/`.env.example`, never hard-coded:

| Variable | Default | Controls |
|---|---|---|
| `CHUNK_TARGET_TOKENS` | 800 | Size a chunk tries to reach before looking for a boundary. Not a hard limit. |
| `CHUNK_MIN_TOKENS` | 300 | Below this, no boundary is taken (except end-of-transcript/forced split). |
| `CHUNK_MAX_TOKENS` | 1200 | Hard ceiling — enforced even if it means splitting a single segment's text. |
| `CHUNK_OVERLAP_TOKENS` | 0 | Segment-granular overlap between adjacent chunks. See "Overlap" above for why this defaults off. |
| `CHUNK_PAUSE_THRESHOLD_MS` | 1500 | A gap at least this long between two segments is a candidate pause boundary. |

### Realistic fixture (§3A.12)

`backend/tests/fixtures/podcast_transcript.py` — ✅ a hand-written, clearly-labeled-synthetic ~78-segment, ~9-minute transcript (not real Supadata output — see the top-of-document caveat) covering: short and long segments, a run of near-zero-gap back-and-forth exchange, several large topic-transition pauses (1.5–3s), multiple distinct topic sections, and one deliberately long uninterrupted 143-word segment with no internal punctuation. Used both as a pytest fixture (`tests/unit/test_chunking_realistic_fixture.py` — no lost/duplicated words, no oversized chunks, deterministic, valid source segments) and manually, via `scripts/inspect_chunks.py`, for the quality pass that found the boundary-preference bug above.

---

## 9. Error model (§23 of the Phase 2 task)

`app/core/exceptions.py` defines one hierarchy (`AppError` base, `code: str` + `status_code: int` + `retryable: bool` on every subclass) shared by the API and both workers; `app/core/error_handlers.py` registers a single FastAPI exception handler that turns any `AppError` into `{"code": ..., "message": ...}` at the right status code, and a catch-all handler that turns anything else into a generic `500 INTERNAL_ERROR` (never leaking a stack trace or exception internals to the client).

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

## 10. API surface (§55–56)

```
POST   /api/v1/episodes                       ✅ create/idempotently return an episode; enqueues ingestion
GET    /api/v1/episodes/{id}                   ✅
GET    /api/v1/episodes/{id}/transcript        ✅
POST   /api/v1/episodes/{id}/process           ⬜ (see §6 — not needed until explicit reprocessing exists)
GET    /api/v1/episodes/{id}/chunks            ⬜ (chunk inspection is currently `scripts/inspect_chunks.py`, not an API — Phase 3A didn't call for one; add when something other than a developer needs to read chunks)
GET    /api/v1/episodes/{id}/article           ⬜
POST   /api/v1/episodes/{id}/qa                ⬜
```

Admin auth (§66) is still ⬜ — every endpoint above is currently unauthenticated. `ADMIN_AUTH_SECRET` is declared in `Settings`/`.env.example` and enforced as a *required, non-empty config value outside development* (see §13), but nothing reads it yet to actually gate a request. This is safe today only because there's no admin-only mutation distinct from the public surface (there is no public surface yet either) — closing this gap is a prerequisite for Phase 6, not before.

---

## 11. Frontend structure

- Next.js App Router, TypeScript, Tailwind (§9). ✅ scaffolding, 🚧 content.
- `/` — placeholder homepage. ✅
- `/dev/ingest` — tiny internal tool (URL input, Ingest button, polls episode status) to exercise the Phase 2 API without curl. ✅ Explicitly not the Phase 6 admin dashboard — see §2.
- Public routes (`/articles/[slug]`, library/search) — ⬜, nothing to render (no Article model yet).
- Real admin routes (`/admin/*`, §48) — ⬜.
- Shared components (§45) — ⬜.

---

## 12. Local development

```bash
cp .env.example .env
docker compose up      # postgres (pgvector), redis, backend, worker, frontend
```

`docker-compose.yml` includes a `worker` service (`arq app.worker.settings.WorkerSettings`) alongside `backend`/`frontend`/`postgres`/`redis`. Backend/worker share the same image and bind mount; the backend container runs `uvicorn --reload`.

Outside Docker:

```bash
cd backend && pip install -e ".[dev]"
alembic upgrade head
uvicorn app.main:app --reload      # API
arq app.worker.settings.WorkerSettings   # worker, separate process
pytest                             # needs a running Postgres — see tests/conftest.py
python -m scripts.inspect_chunks <episode_id> [--full-text] [--segments]   # dev/debug chunk viewer
```

---

## 13. Configuration (§53–54)

See [`.env.example`](./.env.example) for the authoritative list. `app/config/settings.py` (`Settings`, pydantic-settings) now does more than load values:

- **Fails fast outside development.** A `model_validator` requires `SUPADATA_API_KEY`, `LLM_API_KEY`, and `ADMIN_AUTH_SECRET` to be non-empty whenever `APP_ENV != "development"` — a misconfigured non-dev deploy refuses to boot instead of silently running with empty secrets. `APP_ENV=development` (the default) never requires them, so the dev inner loop stays credential-free.
- **Never exposes secrets to the frontend.** The only backend-configured value the frontend ever sees is `NEXT_PUBLIC_API_BASE_URL` (not a secret) — verified structurally: nothing in `frontend/` imports or proxies any other env var, and Next.js itself only ever bundles `NEXT_PUBLIC_*` into client code.
- **Chunking configuration** (`CHUNK_TARGET_TOKENS`, `CHUNK_MIN_TOKENS`, `CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP_TOKENS`, `CHUNK_PAUSE_THRESHOLD_MS`) — see §8 for what each controls and why `CHUNK_OVERLAP_TOKENS` defaults to 0.

---

## 14. Core dependencies

**Backend** (`backend/pyproject.toml`): `fastapi`, `uvicorn`, `sqlalchemy[asyncio]`, `asyncpg`, `alembic` ✅, `pydantic`/`pydantic-settings`, `redis`, `arq` ✅, `httpx` (also the Supadata client). Dev-only: `pytest`, `pytest-asyncio`, `respx` ✅ (HTTP mocking for the provider tests). Not yet added: `pgvector` (Python client — needed once embeddings exist, Phase 3C), `langgraph`/`langchain`, an LLM SDK, an embedding SDK. Phase 3A's cleaner/chunker added **no new dependencies** — both are pure Python + regex over data already loaded via SQLAlchemy.

**Frontend** (`frontend/package.json`): `next@16`, `react@19`/`react-dom@19`, `typescript`, `tailwindcss` — all still auditing clean (`npm audit` → 0 vulnerabilities is a bar for landing any dependency change, not just a nice-to-have). A typed API client layer and a test runner are still ⬜.

**Infra**: `pgvector/pgvector:pg16`, `redis:7-alpine`, via `docker-compose.yml`. The pgvector extension itself is still not `CREATE EXTENSION`'d anywhere (no migration needs it yet — no vector columns exist until Phase 3C).

**CI**: `.github/workflows/ci.yml` ✅ — backend job runs migrations against a real Postgres service container then `pytest`; frontend job runs `npm ci && npm run build && npm audit`. Both must pass for a PR to be considered green.

---

## 15. MVP build order

This mirrors §97 and §101–105; each milestone is only started once the previous one is reliable.

1. **Phase 1 — Foundation** ✅: repo scaffolding, Docker Compose, FastAPI skeleton with health check, Next.js skeleton, env config, CI, fail-fast settings.
2. **Phase 2 — Ingestion** ✅ (§101, first functional milestone): YouTube URL validation → `SupadataTranscriptProvider` → metadata + transcript normalization → persisted to Postgres via an async job → queryable through the API. No AI processing.
3. **Phase 3A — Cleaning + Chunking** ✅ (§102, first half): deterministic per-segment cleaning, hybrid pause/sentence/size chunking, chunk persistence with source traceability, auto-chained after ingestion. No embeddings, no chunk analysis. Validated against a real ~2h47m podcast transcript.
4. **V1 AI pipeline — Topic analysis through a reviewable article** ✅ (§17 below, covers what the original sketch's Phases 3B-6 described, built as one V1 pass rather than staged phases): `LLMProvider` abstraction (Groq/OpenAI/self-hosted), topic analysis, article planning, section-by-section generation, deterministic validation, a bounded revision loop, and a minimal review API/UI. Explicitly **not** including pgvector/embeddings-based retrieval (§17.7 — not needed at this scale) or fidelity verification beyond the deterministic checks (LLM-based validation exists as an optional, off-by-default supplementary signal only).
5. **Real end-to-end validation against a real LLM + real Supadata data** ⬜: next step — this sandbox cannot reach either provider (see the top-of-document caveat); `tests/integration/test_article_pipeline_smoke.py` and the real worker/API path are ready for it.
6. **The real editorial CMS + public reading experience** ⬜: homepage content, library, article page, the real admin dashboard. `frontend/app/dev/review/[episodeId]` gets replaced, not extended.
7. **Semantic retrieval / Q&A** ⬜: only once a concrete need for it is demonstrated against real generated articles (§17.7).
8. **Evaluation** ⬜: evaluation dataset, metrics, regression tests, LLM cost tracking.

Definition of done for the MVP as a whole is the checklist in §106 of the product spec.

---

## 16. Open architectural decisions (to resolve when their phase starts)

- **Object storage implementation**: local filesystem for development vs an S3-compatible provider — `StorageProvider` still doesn't exist because nothing binary needs storing yet (transcripts are text, in Postgres).
- **Admin auth mechanism**: simplest viable (a single admin credential + session/JWT) vs a fuller auth provider. `ADMIN_AUTH_SECRET` is reserved and required outside dev, but unused — needs an actual decision before Phase 6 adds admin-only mutations.
- **`Podcast` entity**: deferred (see §5) until something needs to group episodes by channel/podcast.
- **Real tokenizer**: `estimate_tokens()` (§8) is a `chars/4` approximation, still true after §17 — every prompt/structured-output size decision in `app/ai/` reuses it rather than a real tokenizer, since no concrete model/tokenizer has been benchmarked against real generation yet. Replace once one has; nothing else in the pipeline's structure needs to change.
- **Chunk-level embeddings and `EmbeddingProvider`**: still deliberately not introduced (§17.7) — add if/when semantic retrieval is actually needed against real generated articles, not preemptively.
- **`Chunk` vector column / pgvector**: deferred alongside embeddings.
- **Per-task model assignment**: V1 uses one `LLM_MODEL` for topic analysis, planning, generation, and validation (§17.4) — assign different models per task once there's real quality/cost/latency data to base that on, not before.

---

## 17. AI pipeline: article generation (Phases C-H)

```
Chunk (Phase 3A, unchanged)
   |
   v
topic_analysis  --LLM, structured, batched-->  Topic (persisted, separate from Chunk)
   |
   v
planning        --LLM, structured-->           ArticlePlan (persisted independently)
   |
   v
section_generation --LLM, structured, per-section--> Article + ArticleSection (persisted)
   |
   v
validation      --deterministic (+ optional LLM)--> ValidationResult (persisted)
   |
   +--fails, revisions remain--> revision (regenerate only the flagged sections) --> back to validation
   |
   +--passes, or revisions exhausted--> Episode.status = READY_FOR_REVIEW
```

This is `app/ai/graph.py` — the one place in the codebase LangGraph is used (§1: "AI workflow → LangGraph; CRUD/DB workflows → normal services"). Ingestion, transcript processing, and the review API are all plain services/repositories, unchanged by this section.

### 17.1 Why canonical chunks aren't article sections

A `Chunk` (§8) is a *source-oriented* unit: sized to a token budget, cut at whichever pause/sentence boundary is nearest, with no notion of what it's "about". An article section is a *meaning-oriented* unit: however many chunks it takes to cover one coherent idea. Collapsing the two would mean either chunk sizes dictate article structure (arbitrary) or article structure dictates chunk sizes (breaks Phase 3A's token-budget guarantees). `Topic` (§17.3) is the explicit middle layer: several chunks group into one topic, several topics group into one planned section — both mappings are `ARRAY(UUID)` columns, the same "array, not join table" pattern as `Chunk.source_segment_ids` (§5), for the same reason: a chunk or topic near a boundary can legitimately belong to more than one grouping.

### 17.2 `LLMProvider` (`app/providers/llm/`)

```python
class LLMProvider(ABC):                                                    # ✅
    async def generate(self, *, messages, system=None, temperature=0.2, max_tokens=None) -> LLMTextResponse: ...
    async def generate_structured(self, *, messages, response_model: type[T], system=None, temperature=0.2, max_tokens=None) -> T: ...

GroqProvider(ChatCompletionsProvider)        # ✅ AsyncGroq
OpenAIProvider(ChatCompletionsProvider)      # ✅ AsyncOpenAI
OpenSourceProvider(ChatCompletionsProvider)  # ✅ AsyncOpenAI pointed at a custom base_url
```

All three subclass `_chat_completions.ChatCompletionsProvider`, which implements `generate`/`generate_structured` once against any client exposing an OpenAI-style `client.chat.completions.create(...)` method — Groq's Python SDK uses that exact shape (its API is OpenAI-compatible), and a self-hosted OpenAI-compatible server (vLLM, Ollama, TGI) needs no fourth SDK, just the `openai` client with a different `base_url`. Each subclass only supplies client construction and exception-class mapping (its own `AuthenticationError`/`RateLimitError`/`APIConnectionError` → `LLMProviderAuthError`/`LLMProviderRateLimitError`/`LLMProviderError`, mirroring `TranscriptProviderError`'s hierarchy in `app/core/exceptions.py`). This was a deliberate late simplification: an earlier version had each provider re-implement `generate`/`generate_structured` (three near-identical copies) — collapsed into the shared base once three real, concrete duplicates existed, not preemptively.

**Structured output** is JSON-mode (`response_format={"type": "json_object"}`) plus the target Pydantic model's JSON Schema embedded in the system prompt, parsed with `model_validate_json`, with **one** good-faith retry (the bad output + validation error fed back to the model) before raising `LLMStructuredOutputError`. This was chosen over OpenAI's newer strict `json_schema` mode specifically because it works identically across all three providers — Groq doesn't support that stricter mode on every model, and the self-hosted case can be any server — "provider/model agnostic" (the task's own requirement) ruled out anything provider-specific.

**Selection** (`app/providers/llm/factory.py`): `get_llm_provider(settings)` reads `LLM_PROVIDER` (`groq` | `openai` | `opensource`) and constructs the matching concrete provider with `LLM_MODEL` and that provider's own key — `Settings._require_secrets_outside_development` only requires the *selected* provider's key outside development (`_LLM_PROVIDER_KEY_FIELD` in `app/config/settings.py`), never all three. `app/ai/` and `app/worker/` depend on `LLMProvider` and the factory only; no `groq`/`openai` import outside `app/providers/llm/`.

**Chunk/topic references in prompts**: every structured schema in `app/ai/schemas.py` (`TopicAnalysisResult`, `ArticlePlanResult`, `GeneratedSection`) refers to chunks/topics by small integers (a chunk's position in the batch, a topic's own `sequence_number`) — models reliably mangle or hallucinate UUIDs but handle small integers correctly. `app/ai/nodes/` resolves every integer back to a real database UUID before anything is persisted; nothing is ever written to the database on the strength of a model-generated ID.

### 17.3 Data model additions

```
Episode
   ├── Transcript (1:1)
   │       ├── TranscriptSegment (ordered)                     ✅
   │       ├── Chunk (ordered)                                 ✅
   │       └── Topic (ordered)                                 ✅  chunk_ids: ARRAY(UUID), key_claims: JSONB
   ├── ArticlePlan (1:1)                                        ✅  sections: JSONB (see below)
   ├── Article (1:1)                                            ✅
   │       ├── ArticleSection (ordered)                         ✅  supporting_chunk_ids/supporting_topic_ids: ARRAY(UUID)
   │       └── ValidationResult (many, kept — not replaced)     ✅  checks: JSONB
   └── ProcessingJob (1:many)                                   ✅  + JobType.ARTICLE_GENERATION
```

- **Topic**: `id`, `transcript_id`/`episode_id` (FK, indexed — same denormalization as `Chunk.episode_id`), `sequence_number`, `title`, `summary`, `chunk_ids` (`ARRAY(UUID)`), `key_claims` (`JSONB` — list of `{text, speaker, claim_type}`), `subtopics` (`ARRAY(Text)`), `created_at`. Regenerated whole (delete + reinsert) each time topic analysis runs, same idempotency strategy as `Chunk` (`TopicRepository.replace_all`).
- **ArticlePlan**: `id`, `episode_id` (FK, unique — 1:1, like `Transcript`), `title`, `introduction_summary`, `conclusion_summary`, `sections` (`JSONB` list of planned sections). `JSONB`, not a child table, for the same reason as `Topic.key_claims`: only ever read/written as a whole alongside its parent, never queried independently in V1.
- **Article**: `id`, `episode_id` (FK, unique), `article_plan_id` (FK), `title`, `revision_count`. Deleting `ArticlePlan` cascades to `Article` (`ondelete='CASCADE'` on `article_plan_id`) — deliberate: a plan regeneration invalidates whatever article was built from the old plan, and the pipeline regenerates the article from the new plan immediately after anyway.
- **ArticleSection**: `id`, `article_id` (FK, indexed), `sequence_number`, `heading`, `content`, `supporting_chunk_ids`/`supporting_topic_ids` (`ARRAY(UUID)`, unique constraint on `(article_id, sequence_number)`). This is Phase D's evidence-grounding layer made concrete: "article section → supporting chunks → transcript segments → timestamps" is a direct field lookup, no join, no retrieval step.
- **ValidationResult**: `id`, `article_id` (FK, indexed), `passed` (bool, denormalized AND of `checks`), `checks` (`JSONB`). **Not** delete-then-replace like every other repository here — each validation run (including the ones between revision attempts) is kept as its own row, so a reviewer or the revision node itself can see the before/after history across a revision loop.
- **`JobType.ARTICLE_GENERATION`**: added via `ALTER TYPE job_type ADD VALUE IF NOT EXISTS` (same migration pattern as every prior `JobType`/enum addition — see §6/migration history). One job type covers the whole graph (topic analysis through validation/revision), not one job per stage, matching how `TRANSCRIPT_PROCESSING` already covers cleaning + chunking as a single job.

Migration: `6c806f44935d_add_article_generation_pipeline_models` — verified with the same downgrade/upgrade round-trip discipline as every prior migration (full cycle + a `-1`/upgrade cycle, no enum corruption).

### 17.4 Model selection

One `LLM_MODEL` drives topic analysis, planning, section generation, and (if enabled) validation in V1 — not because the architecture can't support per-task models (`PipelineDeps`/`LLMProvider` are already per-call, so a future `deps.llm_provider_for(task)` is a small change, not a redesign), but because assigning different models per task before there's real quality/cost/latency data to base it on would be premature optimization. Benchmark against real generated articles first (§16), then split if the data supports it.

### 17.5 LangGraph design

`app/ai/state.py::ArticlePipelineState` (a `TypedDict`) is the graph's state; `PipelineDeps` (session, `LLMProvider`, `Settings` — plus the repositories built from them) is threaded through every node via closures (`nodes/*.py::build(deps) -> node_fn`), not through graph state itself, since it's infrastructure, not pipeline data. **No checkpointer is configured** — this compiles to a plain in-process graph for one arq job invocation; Postgres (via each node's own repository writes, one commit per node) is already the durability layer, and a crash mid-run is handled by arq's existing retry policy (`MAX_TRIES`, §6), the same as every other job in this codebase. Wiring up LangGraph's own persistence/checkpointing would duplicate that for no benefit — the "don't overbuild" instruction this pipeline was built under ruled it out explicitly.

Each node commits its own work before returning (topic analysis persists `Topic` rows, planning persists `ArticlePlan`, section generation persists `Article`+`ArticleSection`, validation persists `ValidationResult`) and updates `Episode.status` at the point it starts (`ANALYZING` → `PLANNING` → `GENERATING` → `VERIFYING` → `REVISING` if a revision loop triggers) — so `GET /episodes/{id}` reflects real progress mid-run, not just a single opaque "processing" state.

**Topic-analysis batching** (`app/ai/nodes/topic_analysis.py`): chunks are grouped into batches under `TOPIC_ANALYSIS_TOKEN_BUDGET` (`estimate_tokens()`-based, default 12000), each batch analyzed independently with chunk indices numbered globally (not reset per batch, so no renumbering is needed afterward), and — only if there was more than one batch — one additional merge call reconciles topics that may have split across a batch boundary. A single-batch transcript skips the merge call entirely.

**Revision loop** (`app/ai/nodes/revision.py`): bounded by `MAX_REVISION_ATTEMPTS` (existing setting, §13). On a failed validation, a regex over each failing check's own `details` text (`"section N"`) identifies which section(s) to regenerate — every `article_validation.py` check that can name a specific section does so in that exact phrasing precisely so this targeting works; a check with no section-specific phrasing (e.g. overall article length) becomes shared feedback appended to whichever sections *do* get regenerated, and if no check names any section at all, every section is regenerated rather than guessing. Regeneration reuses `section_generation.generate_section()` directly — the revision path is not a second implementation of section generation, only a different call site with extra feedback in the prompt.

### 17.6 Deterministic validation (`app/services/article_validation.py`)

Ten pure-Python, LLM-free checks (no DB session, no network) — this is Phase H's reliability backbone, independently unit-tested (`tests/unit/test_article_validation.py`) without needing a database or a model call:

1. **source_coverage** — what fraction of the transcript's chunks are cited by at least one section (reported always; fails only on zero coverage — not every chunk must be cited).
2. **source_traceability** — every `supporting_chunk_id` resolves to a real `Chunk`.
3. **no_empty_sections** — heading and content (≥ a minimum length) are actually present.
4. **no_duplicate_sections** — no two sections share a heading or content.
5. **no_duplicate_paragraphs** — no paragraph (above a trivial-length floor) repeats across sections.
6. **no_missing_planned_sections** — every `ArticlePlan` section produced a generated `ArticleSection`.
7. **article_length** — generated word count vs. `ARTICLE_MAX_LENGTH_RATIO` of the transcript's word count (the product goal — "substantially shorter" — as a soft, reported ratio, plus a hard floor against an essentially-empty generation).
8. **unsupported_content** — flags a section with *no* `supporting_chunk_ids` or `supporting_topic_ids` at all; a structural proxy for "unsupported claims" (true claim-level fidelity checking needs an LLM/NLP judge, explicitly out of scope for a deterministic check).
9. **invalid_source_references** — every `supporting_topic_id`, on sections and on the plan itself, resolves to a real `Topic`.
10. **broken_timestamp_references** — every cited chunk's `start_ms`/`end_ms` are non-negative and ordered correctly.

An **optional** LLM-based coherence review (`ENABLE_LLM_VALIDATION`, off by default) is appended to the persisted `checks` list as pure supplementary information — it never contributes to `ValidationReport.passed`, and a failure to even run it (provider error) is swallowed, never fails the pipeline. "Never a replacement for the deterministic checks" (the task's own requirement) is enforced structurally: `article_validation.py` has no LLM import at all.

### 17.7 Why no pgvector / embeddings / RAG in V1

Evidence grounding (Phase D) is satisfied entirely by direct ID-array references — `ArticleSection.supporting_chunk_ids`/`supporting_topic_ids` — resolved by lookup, not by similarity search. At V1's scale (39 chunks for a 2h47m podcast, per the real validation run), an LLM reading a compact topic list or a handful of chunks per section has no retrieval problem to solve; semantic search would add infrastructure (an embedding model choice, a vector index, a retrieval step with its own failure modes) to solve a problem this pipeline doesn't have yet. Introduce `EmbeddingProvider`/pgvector when a concrete need appears against real generated articles (e.g. cross-episode Q&A, §16) — not preemptively because the original architecture sketch mentioned RAG.

### 17.8 API surface additions

```
POST   /api/v1/episodes/{id}/generate-article   ✅  202, enqueues ARTICLE_GENERATION — never auto-chained
                                                      after chunking (unlike ingestion -> processing): this
                                                      makes paid LLM calls, so it's explicit-only.
GET    /api/v1/episodes/{id}/article             ✅  article + sections (with resolved source chunks/timestamps)
                                                      + latest validation result + current episode status
```

`frontend/app/dev/review/[episodeId]/page.tsx` is Phase I's minimal reviewable surface over this: the article, its sections, each section's supporting source timestamps, processing status, and validation pass/fail per check — explicitly not a styled editorial CMS (§16's "real admin dashboard" is still later).

These are called out rather than pre-decided because §98 instructs explaining *why* before making an architectural change, and none of them are forced by Phase 3A.
