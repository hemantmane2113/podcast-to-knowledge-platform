# Podcast-to-Knowledge Platform

Turn conversations into knowledge.

This platform transforms long-form podcast conversations (starting with YouTube) into high-quality, grounded, easy-to-read knowledge articles — roughly turning 2 hours of conversation into 15–30 minutes of high-quality reading. It is an AI editorial engine, not a basic summarizer: it understands the whole conversation, reorganizes its ideas into a coherent narrative, generates a readable article, and verifies that the article stays faithful to the original transcript.

> **Status: Phase 3A — Transcript Cleaning + Chunking.** YouTube URL → Supadata → normalized transcript → PostgreSQL is implemented end-to-end, asynchronously, and queryable through the API; transcripts are then deterministically cleaned and split into semantically-grouped chunks, automatically, with full source-segment traceability. Embeddings/retrieval, article generation, verification, and the public reading/admin UI have not been built yet — see [Current status](#current-status) below. **Note:** ingestion has not yet been validated against the real Supadata API from this repository's build environment (see `ARCHITECTURE.md`'s top-of-document caveat) — everything below is verified against a real Postgres/Redis/worker and a mocked/fixture transcript, not a live Supadata response.

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
│   ├── alembic.ini, alembic/  # two migrations: Episode/Transcript/TranscriptSegment/ProcessingJob, then Chunk + cleaned_text
│   ├── app/
│   │   ├── main.py
│   │   ├── api/               # health, episodes (create/get/transcript)
│   │   ├── core/              # db engine, exceptions, error handlers
│   │   ├── config/            # pydantic-settings, fails fast outside development
│   │   ├── models/            # Episode, Transcript, TranscriptSegment, Chunk, ProcessingJob
│   │   ├── schemas/           # provider-normalized + API I/O schemas
│   │   ├── repositories/      # DB access per entity
│   │   ├── services/          # EpisodeService, IngestionService, TranscriptProcessingService, cleaning/chunking, JobQueue
│   │   ├── providers/transcript/  # TranscriptProvider, SupadataTranscriptProvider
│   │   └── worker/            # arq WorkerSettings + ingestion/processing tasks
│   ├── scripts/inspect_chunks.py  # dev/debug: print a transcript's chunks
│   └── tests/                 # unit/, integration/ (real Postgres, mocked Supadata), fixtures/ (realistic transcript)
├── frontend/
│   ├── package.json
│   └── app/
│       ├── page.tsx           # placeholder homepage
│       └── dev/ingest/        # tiny internal tool to trigger/inspect ingestion (not the real admin UI)
├── evaluation/                # evaluation dataset — not yet added
├── docs/
│   └── adr/
└── scripts/
```

The full target layout, with backend module responsibilities for later phases and an explicit IMPLEMENTED/PLANNED status per component, is documented in [`ARCHITECTURE.md`](./ARCHITECTURE.md#2-repository-structure).

## Current status

**Phase 1 (Foundation)**, **Phase 2 (Transcript Ingestion)**, and **Phase 3A (Cleaning + Chunking)** are done, per `PRODUCT_SPEC.md` §97/§101/§102 and `ARCHITECTURE.md` §15:

- [x] Product spec, architecture doc, env var reference
- [x] Docker Compose (Postgres + pgvector, Redis, backend, worker, frontend), CI (GitHub Actions)
- [x] FastAPI skeleton with `/api/v1/health`; Next.js skeleton with a placeholder homepage
- [x] YouTube URL validation, idempotent episode creation, async ingestion job (arq/Redis)
- [x] `SupadataTranscriptProvider` behind a `TranscriptProvider` interface — metadata + transcript, normalized, never fabricated (**not yet validated against the real Supadata API** — see the status note above)
- [x] Episode/Transcript/TranscriptSegment/Chunk/ProcessingJob persisted via SQLAlchemy + Alembic
- [x] `POST /api/v1/episodes`, `GET /api/v1/episodes/{id}`, `GET /api/v1/episodes/{id}/transcript` — tested and verified against a real Postgres + Redis + live worker process, not just mocks
- [x] Consistent API error model (`{"code", "message"}` with correct HTTP status), retry policy (max 3 attempts, non-retryable errors never retried), no partial persistence on failure
- [x] Deterministic transcript cleaning (whitespace/punctuation normalization, rolling-caption-overlap dedup) — raw text never modified
- [x] Hybrid chunking (pause-gap + sentence-boundary preference, size-constrained, safe mid-segment splitting for oversized text) — no embeddings, no naive fixed-token splitting
- [x] Chunk persistence with full source-segment traceability, idempotent regeneration (delete-then-insert), auto-chained after ingestion
- [x] `scripts/inspect_chunks.py` dev tool + a realistic synthetic transcript fixture used for a manual quality-inspection pass (which found and fixed a real chunk-boundary-quality bug)
- [x] 115 backend tests — unit, integration against real Postgres, respx-mocked Supadata, live ASGI API tests
- [ ] Embeddings, semantic retrieval, RAG — Phase 3B/3C
- [ ] Conversation map, article planning/generation, verification — Phase 4/5
- [ ] Public reading experience and the real admin dashboard — Phase 6

## Local development

```bash
cp .env.example .env   # fill in real values — never commit .env
docker compose up      # Postgres (pgvector) + Redis + backend + worker + frontend
```

- Backend health check: `http://localhost:8000/api/v1/health`
- Ingest an episode: `POST http://localhost:8000/api/v1/episodes {"youtube_url": "https://www.youtube.com/watch?v=..."}` (requires a real `SUPADATA_API_KEY` in `.env` to actually fetch a transcript — without one, the job fails clearly rather than hanging)
- Frontend: `http://localhost:3000`, or `http://localhost:3000/dev/ingest` for a minimal UI over the above

To run services outside Docker during development:

```bash
# Backend (requires Python 3.11+ and a running Postgres + Redis)
cd backend
pip install -e ".[dev]"
alembic upgrade head
uvicorn app.main:app --reload      # API
arq app.worker.settings.WorkerSettings   # worker, run in a separate terminal
pytest                             # needs DATABASE_URL pointed at a real (test) Postgres — see tests/conftest.py
python -m scripts.inspect_chunks <episode_id> [--full-text]   # dev/debug: print a transcript's chunks

# Frontend (requires Node 20+)
cd frontend
npm install
npm run dev
```

## Configuration

All configuration is via environment variables — see [`.env.example`](./.env.example) for the full list with descriptions. Secrets are never hard-coded, never logged, never committed, and never exposed to frontend code; all secret-holding calls happen on the backend.

## Contributing

Development follows the vertical-slice workflow in `PRODUCT_SPEC.md` §99: understand → design → implement → test → run → fix → document, one feature at a time. See `ARCHITECTURE.md` §15 for the current build order.
