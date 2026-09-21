# Podcast-to-Knowledge Platform

Turn conversations into knowledge.

This platform transforms long-form podcast conversations (starting with YouTube) into high-quality, grounded, easy-to-read knowledge articles — roughly turning 2 hours of conversation into 15–30 minutes of high-quality reading. It is an AI editorial engine, not a basic summarizer: it understands the whole conversation, reorganizes its ideas into a coherent narrative, generates a readable article, and verifies that the article stays faithful to the original transcript.

> **Status: V1 end-to-end pipeline (Phases A-I).** YouTube URL → Supadata → transcript → cleaning → canonical chunks → topic analysis → article plan → section-by-section generation → deterministic validation → a minimal reviewable state, all wired end-to-end asynchronously and queryable through the API. Phase 3A (cleaning/chunking) has been validated against a real ~2h47m podcast transcript; the AI pipeline (Phases C-H) has been built and tested against a fake LLM provider but has **not yet been run against a real LLM from this sandbox** — this environment cannot reach `api.supadata.ai`, `api.groq.com`, or `api.openai.com` (see `ARCHITECTURE.md`'s top-of-document caveat). See [Current status](#current-status) below for exactly what's verified how.

## Documentation

- [`PRODUCT_SPEC.md`](./PRODUCT_SPEC.md) — the source-of-truth product and engineering specification: product vision, MVP scope, pipeline design, data model, prompt philosophy, and definition of done.
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — the engineering translation of the spec: repository structure, module responsibilities, provider abstractions, data model, API surface, environment variables, dependencies, and build order.

Read the product spec first for *what* and *why*; read the architecture doc for *how* it maps to code.

## Tech stack

- **Frontend**: Next.js, TypeScript, Tailwind CSS
- **Backend**: Python, FastAPI
- **AI orchestration**: LangGraph, with LangChain used only where it materially simplifies an integration
- **Database**: PostgreSQL (pgvector for embeddings, from Phase 3B/3C on), SQLAlchemy (async) + Alembic migrations
- **Background processing**: Redis-backed async job queue ([arq](https://github.com/python-arq/arq))
- **Transcript acquisition**: [Supadata](https://supadata.ai/) (YouTube transcripts), isolated behind a `TranscriptProvider` abstraction
- **Deployment**: Docker-first; the whole stack runs locally via `docker compose up`

Architecture is a clean modular monolith (one FastAPI backend, one Next.js frontend) — no microservices, no Kubernetes, no dedicated vector database in V1. See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the reasoning.

## Repository structure

```text
podcast-to-knowledge-platform/
├── README.md
├── PRODUCT_SPEC.md
├── ARCHITECTURE.md
├── docker-compose.yml
├── .env.example
├── .github/workflows/ci.yml   # backend pytest; frontend build + audit
├── backend/
│   ├── pyproject.toml
│   ├── alembic.ini, alembic/  # 4 migrations: initial schema, Chunk, [refactor], article pipeline models
│   ├── app/
│   │   ├── main.py
│   │   ├── api/               # health, episodes (create/get/transcript), articles (generate/get)
│   │   ├── core/               # db engine, exceptions, error handlers
│   │   ├── config/             # pydantic-settings, fails fast outside development
│   │   ├── models/             # Episode, Transcript, TranscriptSegment, Chunk, ProcessingJob,
│   │   │                       # Topic, ArticlePlan, Article, ArticleSection, ValidationResult
│   │   ├── schemas/            # provider-normalized + API I/O schemas
│   │   ├── repositories/       # DB access per entity
│   │   ├── services/           # EpisodeService, IngestionService, TranscriptProcessingService,
│   │   │                       # ArticleService, cleaning/chunking, article_validation, JobQueue
│   │   ├── providers/
│   │   │   ├── transcript/     # TranscriptProvider, SupadataTranscriptProvider
│   │   │   └── llm/            # LLMProvider, Groq/OpenAI/OpenSource providers, factory
│   │   ├── ai/                 # the AI workflow only (LangGraph) -- schemas, prompts, nodes/, graph.py
│   │   └── worker/             # arq WorkerSettings + ingestion/processing/article-generation tasks
│   ├── scripts/                # inspect_chunks.py, inspect_article.py, validate_real_transcript.py
│   └── tests/                  # unit/, integration/ (real Postgres, mocked providers), fixtures/
├── frontend/
│   ├── package.json
│   └── app/
│       ├── page.tsx            # placeholder homepage
│       └── dev/
│           ├── ingest/         # tiny internal tool to trigger/inspect ingestion (not the real admin UI)
│           └── review/[episodeId]/  # minimal article review UI (not the real editorial CMS)
├── evaluation/                # evaluation dataset — not yet added
├── docs/
│   └── adr/
└── scripts/
```

The full target layout, with backend module responsibilities for later phases and an explicit IMPLEMENTED/PLANNED status per component, is documented in [`ARCHITECTURE.md`](./ARCHITECTURE.md#2-repository-structure).

## Current status

**Phases A-I (ingestion through a reviewable article) are implemented**, per `PRODUCT_SPEC.md` and `ARCHITECTURE.md` §15:

**Ingestion + transcript processing (validated against a real ~2h47m podcast transcript):**
- [x] YouTube URL validation, idempotent episode creation, async ingestion job (arq/Redis)
- [x] `SupadataTranscriptProvider` — metadata + transcript, normalized, never fabricated (**not validated against the real Supadata API from this sandbox** — its network policy blocks `api.supadata.ai`; validated separately from an environment that can reach it, see `scripts/validate_real_transcript.py`)
- [x] Deterministic transcript cleaning (raw text never modified; cleaning's output is a transient in-memory `CleanedSegment`, never persisted) + hybrid pause/sentence-aware chunking (no embeddings, no naive fixed-token splitting), full source-segment traceability, idempotent regeneration

**AI pipeline (Phases C-H — built and tested against a `FakeLLMProvider`; not yet run against a real LLM from this sandbox, which also cannot reach `api.groq.com`/`api.openai.com`):**
- [x] `LLMProvider` abstraction (`app/providers/llm/`) — `GroqProvider`, `OpenAIProvider`, `OpenSourceProvider` (any OpenAI-compatible endpoint), selected via `LLM_PROVIDER`; only the selected provider's API key is required. Structured output via JSON-mode + Pydantic validation with a one-shot retry.
- [x] Topic analysis over canonical chunks (`app/ai/nodes/topic_analysis.py`) — token-budgeted batching + merge for long transcripts, structured output, persisted as `Topic` rows separate from `Chunk` (a topic spans several chunks)
- [x] Article planning (`ArticlePlan`, persisted independently for inspection) grounded in the identified topics
- [x] Section-by-section generation (never one call over the whole transcript) with explicit anti-hallucination/attribution constraints per section
- [x] Deterministic validation (`app/services/article_validation.py`, 10 checks, LLM-free) — source coverage/traceability, empty/duplicate sections & paragraphs, missing planned sections, article length, unsupported content, invalid references, broken timestamps
- [x] Bounded revision loop (LangGraph conditional edge, `MAX_REVISION_ATTEMPTS`) that regenerates only the sections a failing check identifies
- [x] `POST /api/v1/episodes/{id}/generate-article`, `GET /api/v1/episodes/{id}/article` + a minimal review UI (`frontend/app/dev/review/[episodeId]`) showing the article, sections, source timestamps, and validation results — Phase I's minimal reviewable state, not a full editorial CMS
- [x] LangGraph used only for this AI workflow (`app/ai/graph.py`) — ingestion/processing/review stay plain services, no checkpointer (Postgres is the durability layer, same as every other job)

**Both together:**
- [x] 204 backend tests (1 real-LLM smoke test skipped by default, run explicitly — see below) — unit, integration against real Postgres, mocked Supadata/LLM providers, live ASGI API tests
- [x] `scripts/inspect_chunks.py`, `scripts/inspect_article.py`, `scripts/validate_real_transcript.py` dev tools
- [ ] pgvector/embeddings — deliberately not introduced; 39 chunks with direct ID-array grounding doesn't need vector retrieval yet (see `ARCHITECTURE.md`)
- [ ] Real end-to-end run against a real LLM + real Supadata data from a network-unrestricted environment — next step, see "Real-data validation" below
- [ ] Public reading experience and the real admin/editorial dashboard — later phase

## Local development

```bash
cp .env.example .env   # fill in real values — never commit .env
docker compose up      # Postgres (pgvector) + Redis + backend + worker + frontend
```

- Backend health check: `http://localhost:8000/api/v1/health`
- Ingest an episode: `POST http://localhost:8000/api/v1/episodes {"youtube_url": "https://www.youtube.com/watch?v=..."}` (requires a real `SUPADATA_API_KEY` in `.env` to actually fetch a transcript — without one, the job fails clearly rather than hanging). Cleaning + chunking run automatically right after.
- Generate an article once chunking has finished: `POST http://localhost:8000/api/v1/episodes/{id}/generate-article` (requires `LLM_PROVIDER` + that provider's API key in `.env` — this is a separate, explicit step, never auto-triggered, since it makes paid LLM calls). Poll `GET /api/v1/episodes/{id}` for `status` (`ANALYZING` → `PLANNING` → `GENERATING` → `VERIFYING` → optionally `REVISING` → `READY_FOR_REVIEW`), then `GET /api/v1/episodes/{id}/article`.
- Frontend: `http://localhost:3000`, `http://localhost:3000/dev/ingest` to trigger ingestion, or `http://localhost:3000/dev/review/{episodeId}` to generate and review an article

To run services outside Docker during development:

```bash
# Backend (requires Python 3.11+ and a running Postgres + Redis)
cd backend
pip install -e ".[dev]"
alembic upgrade head
uvicorn app.main:app --reload      # API
arq app.worker.settings.WorkerSettings   # worker, run in a separate terminal
pytest                             # needs DATABASE_URL pointed at a real (test) Postgres — see tests/conftest.py
python -m scripts.inspect_chunks <episode_id> [--full-text]    # dev/debug: print a transcript's chunks
python -m scripts.inspect_article <episode_id> [--full-text]   # dev/debug: print a generated article
# From an environment that can reach api.supadata.ai (this repo's build/CI sandbox cannot,
# see the status note above) -- validates cleaning+chunking against a real transcript
# without touching Postgres/Redis/the DB, printing only statistics + truncated previews:
SUPADATA_API_KEY=... python -m scripts.validate_real_transcript "<youtube_url>" [--full-text]
# Runs the one real-LLM smoke test (skipped by default, never in CI -- see the test file):
RUN_REAL_LLM_SMOKE_TEST=1 LLM_PROVIDER=groq GROQ_API_KEY=... LLM_MODEL=... \
    pytest tests/integration/test_article_pipeline_smoke.py -v -s

# Frontend (requires Node 20+)
cd frontend
npm install
npm run dev
```

## Configuration

All configuration is via environment variables — see [`.env.example`](./.env.example) for the full list with descriptions. Secrets are never hard-coded, never logged, never committed, and never exposed to frontend code; all secret-holding calls happen on the backend.

## Contributing

Development follows the vertical-slice workflow in `PRODUCT_SPEC.md` §99: understand → design → implement → test → run → fix → document, one feature at a time. See `ARCHITECTURE.md` §15 for the current build order.
