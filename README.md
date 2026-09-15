# Podcast-to-Knowledge Platform

Turn conversations into knowledge.

This platform transforms long-form podcast conversations (starting with YouTube) into high-quality, grounded, easy-to-read knowledge articles — roughly turning 2 hours of conversation into 15–30 minutes of high-quality reading. It is an AI editorial engine, not a basic summarizer: it understands the whole conversation, reorganizes its ideas into a coherent narrative, generates a readable article, and verifies that the article stays faithful to the original transcript.

> **Status: Phase 1 — Foundation.** The repo scaffolding (Docker Compose, FastAPI skeleton, Next.js skeleton) is in place; the ingestion and AI processing pipelines have not been built yet — see [Current status](#current-status) below.

## Documentation

- [`PRODUCT_SPEC.md`](./PRODUCT_SPEC.md) — the source-of-truth product and engineering specification: product vision, MVP scope, pipeline design, data model, prompt philosophy, and definition of done.
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — the engineering translation of the spec: repository structure, module responsibilities, provider abstractions, data model, API surface, environment variables, dependencies, and build order.

Read the product spec first for *what* and *why*; read the architecture doc for *how* it maps to code.

## Tech stack

- **Frontend**: Next.js, TypeScript, Tailwind CSS
- **Backend**: Python, FastAPI
- **AI orchestration**: LangGraph, with LangChain used only where it materially simplifies an integration
- **Database**: PostgreSQL with pgvector for embeddings
- **Background processing**: Redis-backed async job queue
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
├── backend/                 # FastAPI app (health check only so far)
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py
│   │   ├── api/v1/          # health router
│   │   ├── config/          # pydantic-settings
│   │   └── core/
│   └── tests/
├── frontend/                 # Next.js app (placeholder homepage only so far)
│   ├── package.json
│   └── app/
├── evaluation/               # evaluation dataset — not yet added
├── docs/
│   └── adr/
└── scripts/
```

The full target layout, with backend module responsibilities for later phases, is documented in [`ARCHITECTURE.md`](./ARCHITECTURE.md#2-repository-structure).

## Current status

This is the **Phase 1 — Foundation** stage described in `PRODUCT_SPEC.md` §97 and `ARCHITECTURE.md` §12. What exists today:

- [x] Product specification (`PRODUCT_SPEC.md`)
- [x] Architecture documentation (`ARCHITECTURE.md`)
- [x] Environment variable reference (`.env.example`)
- [x] Docker Compose (Postgres + pgvector, Redis, backend, frontend)
- [x] FastAPI skeleton with `/api/v1/health` (tested, verified to boot and respond)
- [x] Next.js skeleton with a placeholder homepage (verified to build)
- [ ] Ingestion pipeline (YouTube URL → Supadata → normalized, persisted transcript)
- [ ] AI processing pipeline (cleaning → chunking → analysis → article generation → verification)
- [ ] Public reading experience and admin dashboard

No business logic (ingestion, chunking, LLM calls, etc.) exists yet — this phase is scaffolding only, per `PRODUCT_SPEC.md` §100.

## Local development

```bash
cp .env.example .env   # fill in real values — never commit .env
docker compose up      # Postgres (pgvector) + Redis + backend + frontend
```

- Backend health check: `http://localhost:8000/api/v1/health`
- Frontend: `http://localhost:3000`

To run services outside Docker during development:

```bash
# Backend (requires Python 3.11+)
cd backend
pip install -e ".[dev]"
uvicorn app.main:app --reload
pytest

# Frontend (requires Node 20+)
cd frontend
npm install
npm run dev
```

## Configuration

All configuration is via environment variables — see [`.env.example`](./.env.example) for the full list with descriptions. Secrets are never hard-coded, never logged, never committed, and never exposed to frontend code; all secret-holding calls happen on the backend.

## Contributing

Development follows the vertical-slice workflow in `PRODUCT_SPEC.md` §99: understand → design → implement → test → run → fix → document, one feature at a time. See `ARCHITECTURE.md` §12 for the current build order.
