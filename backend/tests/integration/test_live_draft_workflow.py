"""Live/Draft Article Workflow + Episode.status Decoupling: end-to-end
coverage that isn't already exercised by the pipeline-level (is_draft
threading), worker-level (Episode.status no longer written by article
generation), or article-API-level (regeneration only touches a stale
draft) test files. This file covers:

- public retrieval/listing never returns a draft, only ever the live
  article (Live/Draft Article Workflow point 6)
- the reviewer's draft-retrieval and generation-status endpoints (points
  5 and 7)
- a currently-PUBLISHED episode's Episode.status and live
  Article/ArticlePlan/ArticleSections/ValidationResults survive a draft
  generation run untouched (points 2-4)
- ArticleService.promote_draft's full transactional behavior: all five
  rejection cases, promoting with no prior live article, the swap's
  invariants (exactly one live row, zero stale rows afterward), a
  rejected promotion leaving the original live state intact, and two
  concurrent promote_draft calls for the same episode being serialized
  rather than racing (points 8-9)
"""

import asyncio
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.api.deps import get_article_service
from app.core.exceptions import (
    ArticleNotFoundError,
    ArticleValidationNotPassedError,
    DraftNotReadyError,
    EpisodeNotFoundError,
)
from app.main import app
from app.models.article import Article
from app.models.article_plan import ArticlePlan
from app.models.episode import Episode, ProcessingStatus
from app.models.processing_job import JobStatus, JobType
from app.models.transcript import Transcript
from app.repositories.article_plan_repository import ArticlePlanRepository
from app.repositories.article_repository import ArticleRepository, ArticleSectionCandidate
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.episode_repository import EpisodeRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.repositories.validation_result_repository import ValidationResultRepository
from app.services.article_service import ArticleService
from app.services.chunking_service import ChunkCandidate
from app.worker.tasks import generate_article
from tests.conftest import TEST_DATABASE_URL
from tests.fakes import FakeJobQueue, FakeLLMProvider


async def _client(db_session: AsyncSession, queue: FakeJobQueue | None = None) -> AsyncClient:
    queue = queue or FakeJobQueue()

    async def override_get_article_service():
        yield ArticleService(db_session, queue)

    app.dependency_overrides[get_article_service] = override_get_article_service
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _seed_episode_and_transcript(
    session: AsyncSession, *, status: ProcessingStatus = ProcessingStatus.CHUNKING
) -> tuple[Episode, Transcript]:
    video_id = f"vid{uuid.uuid4().hex[:8]}"
    episode = Episode(
        id=uuid.uuid4(),
        youtube_video_id=video_id,
        youtube_url=f"https://www.youtube.com/watch?v={video_id}",
        status=status,
    )
    session.add(episode)
    transcript = Transcript(id=uuid.uuid4(), episode_id=episode.id, language="en")
    session.add(transcript)
    await session.commit()
    return episode, transcript


async def _seed_article(
    session: AsyncSession,
    episode: Episode,
    transcript: Transcript,
    *,
    is_draft: bool,
    title: str = "The Article",
    validation_passed: bool | None = True,
) -> uuid.UUID:
    """Seeds a plan + single-section article at the given is_draft, and
    (unless validation_passed is None, meaning "no ValidationResult at
    all") a ValidationResult. Reuses an already-seeded chunk for this
    transcript if one exists, so a live+draft pair for the same episode
    can share the same source chunk. Returns the new article's id."""
    existing_chunks = await ChunkRepository(session).get_by_transcript_id(transcript.id)
    if existing_chunks:
        chunk = existing_chunks[0]
    else:
        chunk = (
            await ChunkRepository(session).replace_all(
                transcript_id=transcript.id,
                episode_id=episode.id,
                candidates=[
                    ChunkCandidate(
                        sequence_number=0,
                        text="Some source text.",
                        start_ms=0,
                        end_ms=1_000,
                        source_segment_ids=[],
                        token_count=5,
                    )
                ],
            )
        )[0]
        await session.commit()

    plan = await ArticlePlanRepository(session).replace(
        episode_id=episode.id,
        title=title,
        introduction_summary="i",
        conclusion_summary="c",
        sections=[],
        is_draft=is_draft,
    )
    await session.commit()
    article = await ArticleRepository(session).replace(
        episode_id=episode.id,
        article_plan_id=plan.id,
        title=title,
        revision_count=0,
        sections=[
            ArticleSectionCandidate(
                sequence_number=0, heading="Intro", content="Some prose.", supporting_chunk_ids=[chunk.id]
            )
        ],
        is_draft=is_draft,
    )
    await session.commit()
    if validation_passed is not None:
        ValidationResultRepository(session).create(article_id=article.id, passed=validation_passed, checks=[])
        await session.commit()
    return article.id


async def _seed_generation_job(session: AsyncSession, episode: Episode, *, status: JobStatus) -> None:
    jobs = ProcessingJobRepository(session)
    job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    await session.commit()
    if status == JobStatus.RUNNING:
        jobs.mark_running(job)
    elif status == JobStatus.COMPLETED:
        jobs.mark_completed(job)
    elif status == JobStatus.FAILED:
        jobs.mark_failed(job, "some failure")
    await session.commit()


# --- G: public retrieval/listing never returns a draft --------------------------------


async def test_public_article_with_only_a_draft_is_not_accessible(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    await _seed_article(db_session, episode, transcript, is_draft=True, title="Draft Only")

    client = await _client(db_session)
    try:
        detail_response = await client.get(f"/api/v1/articles/{episode.id}")
        list_response = await client.get("/api/v1/articles")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert detail_response.status_code == 404
    assert detail_response.json()["code"] == "ARTICLE_NOT_FOUND"
    assert list_response.json()["articles"] == []


async def test_public_article_returns_only_live_content_when_a_draft_also_exists(
    db_session: AsyncSession,
) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    await _seed_article(db_session, episode, transcript, is_draft=False, title="Live Title")
    await ArticleService(db_session, FakeJobQueue()).publish_article(episode.id)
    # A draft sitting alongside the now-published live article (e.g. an
    # in-progress regeneration) must never leak into the public response.
    await _seed_article(db_session, episode, transcript, is_draft=True, title="Draft Title")

    client = await _client(db_session)
    try:
        detail_response = await client.get(f"/api/v1/articles/{episode.id}")
        list_response = await client.get("/api/v1/articles")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert detail_response.status_code == 200
    assert detail_response.json()["title"] == "Live Title"
    articles = list_response.json()["articles"]
    assert len(articles) == 1
    assert articles[0]["title"] == "Live Title"


# --- H: reviewer draft retrieval (GET .../article?draft=true) -------------------------


async def test_get_article_default_returns_live_not_draft(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    await _seed_article(db_session, episode, transcript, is_draft=False, title="Live Title")
    await _seed_article(db_session, episode, transcript, is_draft=True, title="Draft Title")

    client = await _client(db_session)
    try:
        response = await client.get(f"/api/v1/episodes/{episode.id}/article")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 200
    assert response.json()["title"] == "Live Title"


async def test_get_article_draft_query_param_returns_draft(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    await _seed_article(db_session, episode, transcript, is_draft=False, title="Live Title")
    await _seed_article(db_session, episode, transcript, is_draft=True, title="Draft Title")

    client = await _client(db_session)
    try:
        response = await client.get(f"/api/v1/episodes/{episode.id}/article?draft=true")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 200
    assert response.json()["title"] == "Draft Title"


async def test_get_article_draft_query_param_404s_when_no_draft_exists(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    await _seed_article(db_session, episode, transcript, is_draft=False, title="Live Title")

    client = await _client(db_session)
    try:
        response = await client.get(f"/api/v1/episodes/{episode.id}/article?draft=true")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 404
    assert response.json()["code"] == "ARTICLE_NOT_FOUND"


# --- generation-status endpoint --------------------------------------------------------


async def test_generation_status_is_null_when_no_job_has_ever_been_created(db_session: AsyncSession) -> None:
    episode, _ = await _seed_episode_and_transcript(db_session)

    client = await _client(db_session)
    try:
        response = await client.get(f"/api/v1/episodes/{episode.id}/generation-status")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 200
    assert response.json() == {"status": None, "error_message": None}


async def test_generation_status_reflects_the_latest_job(db_session: AsyncSession) -> None:
    episode, _ = await _seed_episode_and_transcript(db_session)
    jobs = ProcessingJobRepository(db_session)
    older_job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    await db_session.commit()
    jobs.mark_failed(older_job, "first attempt failed")
    await db_session.commit()
    newer_job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    await db_session.commit()
    jobs.mark_running(newer_job)
    await db_session.commit()

    client = await _client(db_session)
    try:
        response = await client.get(f"/api/v1/episodes/{episode.id}/generation-status")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 200
    assert response.json() == {"status": "RUNNING", "error_message": None}


# --- B/C: a PUBLISHED episode's live content survives a draft generation run ----------


def _fake_llm_provider() -> FakeLLMProvider:
    from app.ai.schemas import ArticlePlanResult, GeneratedSection, PlannedSection, TopicAnalysisResult, TopicItem

    return FakeLLMProvider(
        structured_responses=[
            TopicAnalysisResult(topics=[TopicItem(title="Topic A", summary="s", chunk_sequence_numbers=[0])]),
            ArticlePlanResult(
                title="Newly Generated Draft",
                introduction_summary="i",
                sections=[
                    PlannedSection(heading="Intro", supporting_topic_sequence_numbers=[0]),
                    PlannedSection(heading="Body", supporting_topic_sequence_numbers=[0]),
                    PlannedSection(heading="Conclusion", supporting_topic_sequence_numbers=[0]),
                ],
                conclusion_summary="c",
            ),
            GeneratedSection(heading="Intro", paragraphs=[" ".join(["intro"] + ["word"] * 149)]),
            GeneratedSection(heading="Body", paragraphs=[" ".join(["body"] + ["word"] * 149)]),
            GeneratedSection(heading="Conclusion", paragraphs=[" ".join(["conclusion"] + ["word"] * 149)]),
        ]
    )


async def test_draft_generation_on_a_published_episode_leaves_live_content_and_status_untouched(
    db_session: AsyncSession,
) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    live_article_id = await _seed_article(db_session, episode, transcript, is_draft=False, title="Live Title")
    await ArticleService(db_session, FakeJobQueue()).publish_article(episode.id)

    # A long enough chunk that a 150-word generated section comfortably
    # satisfies article_max_length_ratio, same as test_worker_tasks.py's
    # own _seed_episode_with_chunks -- avoids an unprogrammed revision
    # round unrelated to what this test is checking.
    await ChunkRepository(db_session).replace_all(
        transcript_id=transcript.id,
        episode_id=episode.id,
        candidates=[
            ChunkCandidate(
                sequence_number=0,
                text=" ".join(["word"] * 2000),
                start_ms=0,
                end_ms=10_000,
                source_segment_ids=[],
                token_count=2000,
            )
        ],
    )
    await db_session.commit()

    jobs = ProcessingJobRepository(db_session)
    job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    await db_session.commit()

    ctx = {"job_try": 1, "llm_provider": _fake_llm_provider()}
    await generate_article(ctx, str(episode.id), str(job.id))

    # Read back through a fresh engine/session -- generate_article writes
    # through its own session (a separate connection), same reasoning as
    # tests/integration/test_worker_tasks.py's _reload.
    engine = create_async_engine(TEST_DATABASE_URL)
    try:
        async with AsyncSession(bind=engine) as fresh_session:
            refreshed_episode = await EpisodeRepository(fresh_session).get_by_id(episode.id)
            assert refreshed_episode.status == ProcessingStatus.PUBLISHED

            still_live = await ArticleRepository(fresh_session).get_by_episode_id(episode.id, is_draft=False)
            assert still_live is not None
            assert still_live.id == live_article_id
            assert still_live.title == "Live Title"

            draft = await ArticleRepository(fresh_session).get_by_episode_id(episode.id, is_draft=True)
            assert draft is not None
            assert draft.id != live_article_id
            assert draft.title == "Newly Generated Draft"

            refreshed_job = await ProcessingJobRepository(fresh_session).get_by_id(job.id)
            assert refreshed_job.status == JobStatus.COMPLETED
    finally:
        await engine.dispose()


# --- I-M: promote_draft's five rejection cases -----------------------------------------


async def test_promote_draft_rejects_when_no_draft_exists(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    live_article_id = await _seed_article(db_session, episode, transcript, is_draft=False, title="Live Title")

    with pytest.raises(ArticleNotFoundError):
        await ArticleService(db_session, FakeJobQueue()).promote_draft(episode.id)

    still_live = await ArticleRepository(db_session).get_by_episode_id(episode.id, is_draft=False)
    assert still_live is not None
    assert still_live.id == live_article_id


async def test_promote_draft_rejects_when_generation_still_running(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    await _seed_article(db_session, episode, transcript, is_draft=True, title="Draft Title")
    await _seed_generation_job(db_session, episode, status=JobStatus.RUNNING)

    with pytest.raises(DraftNotReadyError):
        await ArticleService(db_session, FakeJobQueue()).promote_draft(episode.id)


async def test_promote_draft_rejects_when_draft_generation_failed(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    live_article_id = await _seed_article(db_session, episode, transcript, is_draft=False, title="Live Title")
    draft_article_id = await _seed_article(db_session, episode, transcript, is_draft=True, title="Draft Title")
    await _seed_generation_job(db_session, episode, status=JobStatus.FAILED)

    with pytest.raises(DraftNotReadyError):
        await ArticleService(db_session, FakeJobQueue()).promote_draft(episode.id)

    # P: a rejected promotion leaves the original live state (and the
    # untouched draft) exactly as they were -- neither is_draft flag was
    # ever flipped.
    still_live = await ArticleRepository(db_session).get_by_episode_id(episode.id, is_draft=False)
    assert still_live is not None
    assert still_live.id == live_article_id
    still_draft = await ArticleRepository(db_session).get_by_episode_id(episode.id, is_draft=True)
    assert still_draft is not None
    assert still_draft.id == draft_article_id


async def test_promote_draft_rejects_when_no_validation_result_exists(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    await _seed_article(
        db_session, episode, transcript, is_draft=True, title="Draft Title", validation_passed=None
    )
    await _seed_generation_job(db_session, episode, status=JobStatus.COMPLETED)

    with pytest.raises(ArticleValidationNotPassedError):
        await ArticleService(db_session, FakeJobQueue()).promote_draft(episode.id)


async def test_promote_draft_rejects_when_latest_validation_failed(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    await _seed_article(
        db_session, episode, transcript, is_draft=True, title="Draft Title", validation_passed=False
    )
    await _seed_generation_job(db_session, episode, status=JobStatus.COMPLETED)

    with pytest.raises(ArticleValidationNotPassedError):
        await ArticleService(db_session, FakeJobQueue()).promote_draft(episode.id)


async def test_promote_draft_unknown_episode_raises_episode_not_found(db_session: AsyncSession) -> None:
    with pytest.raises(EpisodeNotFoundError):
        await ArticleService(db_session, FakeJobQueue()).promote_draft(uuid.uuid4())


# --- N/O: successful promotion -----------------------------------------------------------


async def test_promote_draft_succeeds_with_no_prior_live_article(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    draft_article_id = await _seed_article(db_session, episode, transcript, is_draft=True, title="Draft Title")
    await _seed_generation_job(db_session, episode, status=JobStatus.COMPLETED)

    result_episode = await ArticleService(db_session, FakeJobQueue()).promote_draft(episode.id)

    assert result_episode.status == ProcessingStatus.PUBLISHED
    assert result_episode.article_published_at is not None

    live = await ArticleRepository(db_session).get_by_episode_id(episode.id, is_draft=False)
    assert live is not None
    # The exact same row, just promoted -- not a copy.
    assert live.id == draft_article_id
    assert live.title == "Draft Title"
    assert await ArticleRepository(db_session).get_by_episode_id(episode.id, is_draft=True) is None


async def test_promote_draft_replaces_the_live_article_and_leaves_no_stale_rows(
    db_session: AsyncSession,
) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    old_live_article_id = await _seed_article(
        db_session, episode, transcript, is_draft=False, title="Old Live Title"
    )
    await ArticleService(db_session, FakeJobQueue()).publish_article(episode.id)
    new_draft_article_id = await _seed_article(
        db_session, episode, transcript, is_draft=True, title="New Draft Title"
    )
    await _seed_generation_job(db_session, episode, status=JobStatus.COMPLETED)

    result_episode = await ArticleService(db_session, FakeJobQueue()).promote_draft(episode.id)

    assert result_episode.status == ProcessingStatus.PUBLISHED

    # Exactly one live article afterward: the promoted draft, under its
    # own original id (not a new row).
    live = await ArticleRepository(db_session).get_by_episode_id(episode.id, is_draft=False)
    assert live is not None
    assert live.id == new_draft_article_id
    assert live.title == "New Draft Title"

    # Zero draft rows left over -- no leftover draft, and the old live
    # row was never left behind demoted to draft either.
    assert await ArticleRepository(db_session).get_by_episode_id(episode.id, is_draft=True) is None

    # The old live row is gone ENTIRELY (not just demoted) -- a raw count
    # confirms there is exactly one Article row for this episode at all.
    article_count = await db_session.scalar(
        select(func.count()).select_from(Article).where(Article.episode_id == episode.id)
    )
    assert article_count == 1
    assert await db_session.get(Article, old_live_article_id) is None

    # Same invariants for ArticlePlan.
    plan_count = await db_session.scalar(
        select(func.count()).select_from(ArticlePlan).where(ArticlePlan.episode_id == episode.id)
    )
    assert plan_count == 1
    live_plan = await ArticlePlanRepository(db_session).get_by_episode_id(episode.id, is_draft=False)
    assert live_plan is not None
    assert await ArticlePlanRepository(db_session).get_by_episode_id(episode.id, is_draft=True) is None


async def test_promote_draft_api_endpoint(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    await _seed_article(db_session, episode, transcript, is_draft=True, title="Draft Title")
    await _seed_generation_job(db_session, episode, status=JobStatus.COMPLETED)

    client = await _client(db_session)
    try:
        response = await client.post(f"/api/v1/episodes/{episode.id}/promote-draft")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 200
    assert response.json()["status"] == "PUBLISHED"


# --- Q: concurrent promote_draft calls are serialized, never racing -------------------


async def test_promote_draft_serializes_concurrent_calls_for_the_same_episode(
    db_session: AsyncSession,
) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    await _seed_article(db_session, episode, transcript, is_draft=True, title="Draft Title")
    await _seed_generation_job(db_session, episode, status=JobStatus.COMPLETED)

    async def _promote() -> str:
        engine = create_async_engine(TEST_DATABASE_URL)
        try:
            async with AsyncSession(bind=engine) as session:
                try:
                    await ArticleService(session, FakeJobQueue()).promote_draft(episode.id)
                    return "promoted"
                except ArticleNotFoundError:
                    # The loser: by the time it acquired the episode row
                    # lock (SELECT ... FOR UPDATE), the winner had already
                    # committed the promotion, so there's no draft left.
                    return "no_draft_left"
        finally:
            await engine.dispose()

    outcomes = await asyncio.gather(_promote(), _promote())

    # Exactly one call actually promoted -- never both (which would mean
    # the row lock didn't serialize them, and both winners might race the
    # demote/promote/delete swap against each other), never neither.
    assert sorted(outcomes) == ["no_draft_left", "promoted"]

    engine = create_async_engine(TEST_DATABASE_URL)
    try:
        async with AsyncSession(bind=engine) as fresh_session:
            live = await ArticleRepository(fresh_session).get_by_episode_id(episode.id, is_draft=False)
            assert live is not None
            assert await ArticleRepository(fresh_session).get_by_episode_id(episode.id, is_draft=True) is None
            article_count = await fresh_session.scalar(
                select(func.count()).select_from(Article).where(Article.episode_id == episode.id)
            )
            assert article_count == 1
    finally:
        await engine.dispose()
