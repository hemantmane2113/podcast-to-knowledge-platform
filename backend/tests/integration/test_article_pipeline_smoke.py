"""The one REAL end-to-end smoke test for the article pipeline -- a real
LLM call (via the configured LLM_PROVIDER), against the real Phase 3A
cleaning/chunking output for the synthetic podcast fixture, through the
whole LangGraph pipeline, persisted to a real Postgres test DB.

Skipped by default (never runs in CI -- "do not make CI depend on paid
external LLM APIs"). Run it explicitly, from an environment that can
reach your LLM provider, with:

    export RUN_REAL_LLM_SMOKE_TEST=1
    export LLM_PROVIDER=groq            # or openai / opensource
    export GROQ_API_KEY=<your real key> # or OPENAI_API_KEY / OPENSOURCE_BASE_URL
    export LLM_MODEL=<a real model id>
    pytest tests/integration/test_article_pipeline_smoke.py -v -s
"""

import os
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.graph import run_article_pipeline
from app.ai.state import ArticlePipelineState, PipelineDeps
from app.config.settings import Settings
from app.models.episode import Episode, ProcessingStatus
from app.models.transcript import Transcript
from app.providers.llm.factory import get_llm_provider
from app.repositories.chunk_repository import ChunkRepository
from app.services.chunking_service import ChunkingConfig, chunk_transcript_segments
from app.services.cleaning_service import clean_transcript_segments
from tests.fixtures.podcast_transcript import build_transcript_segments

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_LLM_SMOKE_TEST") != "1",
    reason="Real-LLM smoke test -- set RUN_REAL_LLM_SMOKE_TEST=1 and real provider "
    "credentials to run it explicitly. Never runs in CI.",
)


async def test_real_llm_produces_a_reviewable_article(db_session: AsyncSession) -> None:
    settings = Settings()  # picks up real env vars, unlike every other test's explicit Settings(...)
    llm_provider = get_llm_provider(settings)

    episode = Episode(
        id=uuid.uuid4(),
        youtube_video_id="smoketest001",
        youtube_url="https://www.youtube.com/watch?v=smoketest001",
        status=ProcessingStatus.CHUNKING,
    )
    db_session.add(episode)
    transcript = Transcript(id=uuid.uuid4(), episode_id=episode.id, language="en")
    db_session.add(transcript)
    await db_session.commit()

    raw_segments = build_transcript_segments(transcript_id=transcript.id)
    cleaned = clean_transcript_segments(raw_segments)
    chunk_config = ChunkingConfig.from_settings(settings)
    candidates = chunk_transcript_segments(cleaned, chunk_config)
    chunks = await ChunkRepository(db_session).replace_all(
        transcript_id=transcript.id, episode_id=episode.id, candidates=candidates
    )
    await db_session.commit()

    deps = PipelineDeps(session=db_session, llm_provider=llm_provider, settings=settings)
    initial_state: ArticlePipelineState = {
        "episode_id": episode.id,
        "transcript_id": transcript.id,
        "chunks": chunks,
        "transcript_word_count": sum(len(c.text.split()) for c in chunks),
        "revision_count": 0,
        "max_revision_attempts": settings.max_revision_attempts,
    }

    final_state = await run_article_pipeline(deps, initial_state)

    print(f"\nGenerated article: {final_state['plan'].title}")
    print(f"Sections: {len(final_state['article'].sections)}")
    print(f"Validation passed: {final_state['validation_report'].passed}")
    for check in final_state["validation_report"].checks:
        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.details}")
    for section in final_state["article"].sections:
        print(f"\n## {section.heading}\n{section.content[:300]}...")

    assert len(final_state["topics"]) > 0
    assert len(final_state["article"].sections) > 0
    assert final_state["validation_report"] is not None
