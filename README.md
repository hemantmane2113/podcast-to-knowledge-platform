# Podcast-to-Knowledge Platform

Turn conversations into knowledge.

This platform transforms long-form podcast conversations (starting with YouTube) into high-quality, grounded, easy-to-read knowledge articles — roughly turning 2 hours of conversation into 15–30 minutes of high-quality reading. It is an AI editorial engine, not a basic summarizer: it understands the whole conversation, reorganizes its ideas into a coherent narrative, generates a readable article, and verifies that the article stays faithful to the original transcript.

> **Status: pre-implementation.** This repository currently contains product/engineering documentation only. No application code has been written yet — see [Current status](#current-status) below.

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

## Repository structure (target)

```text
podcast-to-knowledge-platform/
├── README.md
├── PRODUCT_SPEC.md
├── ARCHITECTURE.md
├── docker-compose.yml       # not yet added
├── .env.example
├── backend/                 # FastAPI app — not yet added
├── frontend/                # Next.js app — not yet added
├── evaluation/               # evaluation dataset — not yet added
├── docs/
│   └── adr/
└── scripts/
```

The full target layout, with backend module responsibilities, is documented in [`ARCHITECTURE.md`](./ARCHITECTURE.md#2-repository-structure).

## Current status

This is the **Phase 1 — Foundation** stage described in `PRODUCT_SPEC.md` §97 and `ARCHITECTURE.md` §12. What exists today:

- [x] Product specification (`PRODUCT_SPEC.md`)
- [x] Architecture documentation (`ARCHITECTURE.md`)
- [x] Environment variable reference (`.env.example`)
- [ ] Docker Compose (Postgres + pgvector, Redis)
- [ ] FastAPI skeleton with health check
- [ ] Next.js skeleton
- [ ] Ingestion pipeline (YouTube URL → Supadata → normalized, persisted transcript)
- [ ] AI processing pipeline (cleaning → chunking → analysis → article generation → verification)
- [ ] Public reading experience and admin dashboard

## Local development

Once the Phase 1 scaffolding lands, local development will be:

```bash
cp .env.example .env   # fill in real values — never commit .env
docker compose up      # PostgreSQL (pgvector) + Redis
```

with the backend and frontend run locally against those services during development. This section will be filled in with concrete run instructions as `backend/` and `frontend/` are added.

## Configuration

All configuration is via environment variables — see [`.env.example`](./.env.example) for the full list with descriptions. Secrets are never hard-coded, never logged, never committed, and never exposed to frontend code; all secret-holding calls happen on the backend.

## Contributing

Development follows the vertical-slice workflow in `PRODUCT_SPEC.md` §99: understand → design → implement → test → run → fix → document, one feature at a time. See `ARCHITECTURE.md` §12 for the current build order.
