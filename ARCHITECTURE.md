# Architecture

This document translates [`PRODUCT_SPEC.md`](./PRODUCT_SPEC.md) into concrete engineering decisions: the repository layout, the module boundaries, the data model, the processing pipeline, required configuration, and the build order. It is derived from the spec and should stay in sync with it — if they disagree, the spec is the source of truth for product intent and this file should be updated to match.

## Status convention

Every section below is written in the target/final tense (this is architecture, not a changelog), but each component is explicitly tagged so this document can never be mistaken for describing more than actually exists:

- **✅ IMPLEMENTED** — exists in the repo today, has tests, has been run against a real database/queue (not just unit-mocked).
- **🚧 PARTIAL** — some of it exists (e.g. the interface but not every implementation).
- **⬜ PLANNED** — described here for context on where it's headed; no code yet.

As of this revision: **Phase 1 (Foundation)**, **Phase 2 (Transcript Ingestion)**, and **Phase 3A (Transcript Cleaning + Chunking)** are ✅ IMPLEMENTED. Everything from Phase 3B onward (embeddings, retrieval, article generation, verification, RAG, LangGraph, the public reading UI, the real admin dashboard) is ⬜ PLANNED.

**Important caveat on Phase 2/3A validation**: ingestion has been exercised against a *real* arq worker, *real* Redis, and *real* Postgres, but never against the *real* Supadata API — this sandboxed build environment's network egress policy blocks `api.supadata.ai` (confirmed directly: the proxy returns 403 on `CONNECT`). `SupadataTranscriptProvider` is fully implemented and thoroughly tested against a mocked HTTP layer (`respx`), and Phase 3A's cleaner/chunker have been run against a realistic hand-built fixture transcript, but **no real Supadata transcript content has been retrieved or validated from this environment.** That validation is expected to happen separately, from an environment that can reach `api.supadata.ai`.

---

## 1. Guiding constraints (from the spec)

- **Modular monolith, not microservices** (§7, §91). One FastAPI backend with clear internal module boundaries; one Next.js frontend. ✅
- **No premature infrastructure**: no Kubernetes, no dedicated vector DB, no multi-provider abstraction gold-plating, no Elasticsearch (§7, §69, §92). ✅
- **Every external dependency is abstracted** behind a provider interface (§11, §64, §65, §93): `TranscriptProvider` ✅, `LLMProvider` ⬜, `EmbeddingProvider` ⬜, `StorageProvider` ⬜.
- **The raw transcript is the source of truth** (§18, §98). ✅ `TranscriptSegment.text` (the provider's raw output) is never modified by cleaning — cleaning writes to a separate `cleaned_text` column (§5, §7). Nothing invented: speaker labels are never fabricated, and cleaning is deterministic, rule-based text normalization only — no LLM (§7).
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

class LLMProvider(ABC): ...                                  # ⬜ PLANNED (Phase 4)
class EmbeddingProvider(ABC): ...                             # ⬜ PLANNED (Phase 3C — deliberately not introduced in 3A, see §8)
class StorageProvider(ABC): ...                                # ⬜ PLANNED (not yet needed — nothing binary to store)
```

`TranscriptProvider` differs from the original sketch in one way: it has two methods (`get_metadata` + `get_transcript`), not one — Supadata exposes metadata and transcript as separate endpoints, and the Episode-metadata requirement (spec §15/task item 8) needs somewhere to live that isn't transcript-shaped.

`SupadataTranscriptProvider` (`app/providers/transcript/supadata.py`) — ✅ implemented against Supadata's documented `/transcript` and `/metadata` endpoints (`x-api-key` auth, transparent 202/job-polling for large videos). **Correction history**: initially implemented with `mode=native` (only return transcripts that already exist as real captions, reasoned as closer to §88 "never invent"); corrected to `mode=auto` per explicit direction — `mode=auto` asks for native captions first and falls back to a generated transcript when a video has none, trading a small amount of that purity for actually getting a transcript on videos without official captions. `text=false` is non-negotiable either way — ingestion needs timestamped segments, not a plain string. One caveat, stated plainly in the module docstring: Supadata's own docs site was not directly fetchable from this build environment (network policy blocked it), so the exact response shape was assembled from third-party summaries of the documented behavior, not verified against a real call (see the top-of-document caveat). Parsing is defensive (`MalformedProviderResponseError` on anything unexpected) precisely because of that uncertainty.

`services/` and `worker/` depend only on the abstract `TranscriptProvider` interface — `IngestionService` takes a `TranscriptProvider` in its constructor and has never imported `SupadataTranscriptProvider` or `httpx` directly; tests inject a `StubProvider` (`tests/fakes.py`) instead.

---

## 5. Data model (§16)

```text
Episode                                            ✅
   ├── Transcript (1:1)                            ✅
   │       ├── TranscriptSegment (ordered)         ✅  text (raw) + cleaned_text (nullable, derived)
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
- **No separate raw/cleaned `Transcript` rows.** The original sketch anticipated a raw/cleaned pair per §18. Instead, cleaning is applied at the *segment* level: `TranscriptSegment.text` (raw, immutable) and `TranscriptSegment.cleaned_text` (nullable, populated by `clean_transcript_segments`). This is the minimum schema addition that satisfies "raw remains recoverable, clean is a derived representation" (§3A.2/§3A.4 of the Phase 3A task) — one nullable column, not a second parallel table hierarchy. `cleaned_text` is safe to recompute/overwrite on a rerun because cleaning is a pure function of `text`.
- **`Chunk` has no join table for source-segment traceability.** `Chunk.source_segment_ids` is a plain `ARRAY(UUID)` column, not a `ChunkSegment` association table. Chunks are normally built from a contiguous range of segments, which a join table would model more "properly," but a single oversized segment can legitimately be split across two adjacent chunks (§8), so the same segment ID can appear in two chunks' arrays — an explicit array handles that with zero extra schema, a strict range (`first_segment_id`, `last_segment_id`) would not.

### IDs and timestamps (locked decisions)

- **Every table uses a UUID primary key** (`uuid.uuid4()`, assigned application-side so it's available before flush — see `app/models/base.py::UUIDPrimaryKeyMixin`). Never an auto-increment integer, and never a natural/external identifier as the PK.
- **`Episode.youtube_video_id`** is the YouTube video ID, stored as a separate, uniquely-indexed column — never the primary key. It's what idempotency is keyed on (§51; see §6).
- **`created_at`/`updated_at`** (`app/models/base.py::TimestampMixin`) are always `timestamptz`, `server_default=now()`, UTC. Transcript timing (`start_ms`, `duration_ms`, and `Chunk.start_ms`/`end_ms`) is a completely separate concept — position within the source media — and is never coerced into a datetime column.

### Current fields

- **Episode**: `id`, `youtube_video_id` (unique, indexed), `youtube_url`, `title`/`description`/`channel_name`/`channel_id`/`thumbnail_url`/`duration_seconds`/`published_at`/`language` (all nullable — populated from provider metadata, never fabricated), `status` (`ProcessingStatus` enum — see below), `last_error`, `created_at`, `updated_at`.
- **Transcript**: `id`, `episode_id` (FK, unique — enforces 1:1), `language`, `created_at`, `updated_at`.
- **TranscriptSegment**: `id`, `transcript_id` (FK, indexed), `sequence_number`, `text` (raw), `cleaned_text` (nullable), `start_ms`, `duration_ms`, `speaker` (nullable, never invented), `created_at`. Unique constraint on `(transcript_id, sequence_number)` — ordering guarantee + query index.
- **Chunk**: `id`, `transcript_id` (FK, indexed), `episode_id` (FK, indexed — denormalized from transcript, avoids a join for the common "chunks for this episode" query), `sequence_number`, `text`, `start_ms`, `end_ms`, `source_segment_ids` (`ARRAY(UUID)`), `token_count` (approximate — see §8), `created_at`. Unique constraint on `(transcript_id, sequence_number)`.
- **ProcessingJob**: `id`, `episode_id` (FK, indexed), `job_type` (`TRANSCRIPT_INGESTION` | `TRANSCRIPT_PROCESSING`), `status` (`PENDING`/`RUNNING`/`COMPLETED`/`FAILED`, indexed), `started_at`, `completed_at`, `error_message` (safe/sanitized — see §9), `created_at`, `updated_at`.

### Migrations

1. `78e3d5ee683d` — initial schema (Episode, Transcript, TranscriptSegment, ProcessingJob).
2. `9d7cf2188f5a` — adds `Chunk`, `TranscriptSegment.cleaned_text`, and the `TRANSCRIPT_PROCESSING` job type. Two things found by actually running `alembic downgrade`/`upgrade` cycles against a real Postgres, not just the forward migration: autogenerate doesn't detect new values on an existing Postgres native `ENUM` type (had to hand-add `ALTER TYPE job_type ADD VALUE`), and since there's no `ALTER TYPE ... DROP VALUE`, the add had to be written as `ADD VALUE IF NOT EXISTS` — otherwise a downgrade-then-upgrade cycle fails with "enum label already exists" on the second upgrade (the same class of bug as migration 1's enum-type-drop issue, different manifestation).

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
        ├─ clean_transcript_segments(transcript.segments)     -- sets .cleaned_text in place
        ├─ chunk_transcript_segments(transcript.segments, config)  -- see §7/§8
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
3. **Phase 3A — Cleaning + Chunking** ✅ (§102, first half): deterministic per-segment cleaning, hybrid pause/sentence/size chunking, chunk persistence with source traceability, auto-chained after ingestion. No embeddings, no chunk analysis.
4. **Phase 3B/3C — Embeddings + Retrieval** ⬜ (§102, second half): `EmbeddingProvider`, pgvector columns, chunk-level embedding storage, semantic retrieval. First phase that needs pgvector.
5. **Phase 4 — Intelligence** ⬜ (§103): conversation map → article plan → grounded section generation. First phase that needs `LLMProvider` and LangGraph.
6. **Phase 5 — Verification** ⬜ (§104): claim extraction, fidelity verification, revision loop.
7. **Phase 6 — UI** ⬜ (§105): homepage content, library, article page, the real admin dashboard, processing status UI. `frontend/app/dev/ingest` gets replaced, not extended.
8. **Phase 7 — Q&A** ⬜: episode-filtered RAG question answering with timestamp citations.
9. **Phase 8 — Evaluation** ⬜: evaluation dataset, metrics, regression tests, LLM cost tracking.

Definition of done for the MVP as a whole is the checklist in §106 of the product spec.

---

## 16. Open architectural decisions (to resolve when their phase starts)

- **Object storage implementation**: local filesystem for development vs an S3-compatible provider — `StorageProvider` still doesn't exist because nothing binary needs storing yet (transcripts are text, in Postgres).
- **Admin auth mechanism**: simplest viable (a single admin credential + session/JWT) vs a fuller auth provider. `ADMIN_AUTH_SECRET` is reserved and required outside dev, but unused — needs an actual decision before Phase 6 adds admin-only mutations.
- **`Podcast` entity**: deferred (see §5) until something needs to group episodes by channel/podcast.
- **Real tokenizer**: `estimate_tokens()` (§8) is a `chars/4` approximation. Replace with a real tokenizer (e.g. `tiktoken` or whatever matches the chosen `LLMProvider`) once Phase 4 picks a concrete model — the chunker's structure doesn't need to change, only that one function.
- **Chunk-level embeddings and `EmbeddingProvider`**: deferred to Phase 3C by explicit instruction (§8) — not forced by anything in Phase 3A.
- **`Chunk` vector column / pgvector**: deferred alongside embeddings.

These are called out rather than pre-decided because §98 instructs explaining *why* before making an architectural change, and none of them are forced by Phase 3A.
