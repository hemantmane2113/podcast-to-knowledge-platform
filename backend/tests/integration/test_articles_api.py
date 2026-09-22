import uuid

from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_article_service
from app.main import app
from app.models.episode import Episode, ProcessingStatus
from app.models.processing_job import JobStatus, JobType
from app.models.transcript import Transcript
from app.repositories.article_plan_repository import ArticlePlanRepository
from app.repositories.article_repository import ArticleRepository, ArticleSectionCandidate
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.repositories.validation_result_repository import ValidationResultRepository
from app.services.article_service import ArticleService
from app.services.chunking_service import ChunkCandidate
from tests.fakes import FakeJobQueue


async def _client(db_session: AsyncSession, queue: FakeJobQueue | None = None) -> AsyncClient:
    queue = queue or FakeJobQueue()

    async def override_get_article_service():
        yield ArticleService(db_session, queue)

    app.dependency_overrides[get_article_service] = override_get_article_service
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _seed_episode_and_transcript(session: AsyncSession) -> tuple[Episode, Transcript]:
    episode = Episode(
        id=uuid.uuid4(),
        youtube_video_id="abc12345678",
        youtube_url="https://www.youtube.com/watch?v=abc12345678",
        status=ProcessingStatus.CHUNKING,
    )
    session.add(episode)
    transcript = Transcript(id=uuid.uuid4(), episode_id=episode.id, language="en")
    session.add(transcript)
    await session.commit()
    return episode, transcript


async def test_generate_article_unknown_episode_returns_404(db_session: AsyncSession) -> None:
    client = await _client(db_session)
    try:
        response = await client.post(
            f"/api/v1/episodes/{uuid.uuid4()}/generate-article"
        )
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 404
    assert response.json()["code"] == "EPISODE_NOT_FOUND"


async def test_generate_article_enqueues_job_and_returns_202(db_session: AsyncSession) -> None:
    episode, _ = await _seed_episode_and_transcript(db_session)
    queue = FakeJobQueue()
    client = await _client(db_session, queue)
    try:
        response = await client.post(f"/api/v1/episodes/{episode.id}/generate-article")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 202
    body = response.json()
    assert body["episode_id"] == str(episode.id)
    assert uuid.UUID(body["job_id"])
    assert len(queue.enqueued_article_generation) == 1
    assert queue.enqueued_article_generation[0][0] == episode.id


# --- request_generation job lifecycle: active / failed / completed -----------------


async def test_generate_article_returns_existing_active_job_instead_of_duplicating(
    db_session: AsyncSession,
) -> None:
    episode, _ = await _seed_episode_and_transcript(db_session)
    existing_job = ProcessingJobRepository(db_session).create(
        episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION
    )
    await db_session.commit()
    assert existing_job.status == JobStatus.PENDING

    queue = FakeJobQueue()
    client = await _client(db_session, queue)
    try:
        response = await client.post(f"/api/v1/episodes/{episode.id}/generate-article")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 202
    assert response.json()["job_id"] == str(existing_job.id)
    assert len(queue.enqueued_article_generation) == 0  # no duplicate job enqueued


async def test_generate_article_allows_a_fresh_attempt_after_a_failed_job(
    db_session: AsyncSession,
) -> None:
    episode, _ = await _seed_episode_and_transcript(db_session)
    jobs = ProcessingJobRepository(db_session)
    failed_job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    await db_session.commit()
    jobs.mark_failed(failed_job, "LLM provider is not configured (check LLM_PROVIDER and its API key).")
    await db_session.commit()

    queue = FakeJobQueue()
    client = await _client(db_session, queue)
    try:
        response = await client.post(f"/api/v1/episodes/{episode.id}/generate-article")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 202
    new_job_id = uuid.UUID(response.json()["job_id"])
    assert new_job_id != failed_job.id  # a genuinely new job, not the failed one
    assert len(queue.enqueued_article_generation) == 1
    assert queue.enqueued_article_generation[0] == (episode.id, new_job_id)


async def test_generate_article_with_multiple_historical_failed_jobs_does_not_500(
    db_session: AsyncSession,
) -> None:
    """End-to-end version of the real production shape reported: several
    historical FAILED ARTICLE_GENERATION jobs accumulated for the same
    episode across repeated attempts. Confirms the real endpoint still
    creates a fresh job cleanly (no 500, no MultipleResultsFound) and
    leaves every historical row untouched -- the repository-level
    multiple-active-rows case is covered directly in
    test_processing_job_repository.py."""
    episode, _ = await _seed_episode_and_transcript(db_session)
    jobs = ProcessingJobRepository(db_session)
    failed_job_ids = []
    for _ in range(3):
        job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
        await db_session.commit()
        jobs.mark_failed(job, "old max_tokens=null failure")
        await db_session.commit()
        failed_job_ids.append(job.id)

    queue = FakeJobQueue()
    client = await _client(db_session, queue)
    try:
        response = await client.post(f"/api/v1/episodes/{episode.id}/generate-article")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 202
    new_job_id = uuid.UUID(response.json()["job_id"])
    assert new_job_id not in failed_job_ids
    assert len(queue.enqueued_article_generation) == 1

    # All three historical FAILED jobs are still present, untouched.
    for job_id in failed_job_ids:
        historical = await jobs.get_by_id(job_id)
        assert historical is not None
        assert historical.status == JobStatus.FAILED


async def _seed_completed_article(session: AsyncSession, episode: Episode, transcript: Transcript) -> None:
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
    await session.commit()
    plan = await ArticlePlanRepository(session).replace(
        episode_id=episode.id, title="p", introduction_summary="i", conclusion_summary="c", sections=[]
    )
    await session.commit()
    await ArticleRepository(session).replace(
        episode_id=episode.id, article_plan_id=plan.id, title="The Article", revision_count=0, sections=[]
    )
    await session.commit()


async def test_generate_article_does_not_regenerate_when_an_article_already_exists(
    db_session: AsyncSession,
) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    await _seed_completed_article(db_session, episode, transcript)

    jobs = ProcessingJobRepository(db_session)
    completed_job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    await db_session.commit()
    jobs.mark_completed(completed_job)
    await db_session.commit()

    queue = FakeJobQueue()
    client = await _client(db_session, queue)
    try:
        response = await client.post(f"/api/v1/episodes/{episode.id}/generate-article")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 202
    assert response.json()["job_id"] == str(completed_job.id)
    assert len(queue.enqueued_article_generation) == 0  # not silently regenerated


async def test_get_article_unknown_episode_returns_404(db_session: AsyncSession) -> None:
    client = await _client(db_session)
    try:
        response = await client.get(f"/api/v1/episodes/{uuid.uuid4()}/article")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 404
    assert response.json()["code"] == "EPISODE_NOT_FOUND"


async def test_get_article_before_generation_returns_404(db_session: AsyncSession) -> None:
    episode, _ = await _seed_episode_and_transcript(db_session)
    client = await _client(db_session)
    try:
        response = await client.get(f"/api/v1/episodes/{episode.id}/article")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 404
    assert response.json()["code"] == "ARTICLE_NOT_FOUND"


async def test_get_article_returns_sections_sources_and_validation(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)

    chunk = (
        await ChunkRepository(db_session).replace_all(
            transcript_id=transcript.id,
            episode_id=episode.id,
            candidates=[
                ChunkCandidate(
                    sequence_number=0,
                    text="Some source text.",
                    start_ms=1_000,
                    end_ms=5_000,
                    source_segment_ids=[],
                    token_count=5,
                )
            ],
        )
    )[0]
    await db_session.commit()

    plan = await ArticlePlanRepository(db_session).replace(
        episode_id=episode.id, title="p", introduction_summary="i", conclusion_summary="c", sections=[]
    )
    await db_session.commit()

    article = await ArticleRepository(db_session).replace(
        episode_id=episode.id,
        article_plan_id=plan.id,
        title="The Article",
        revision_count=0,
        sections=[
            ArticleSectionCandidate(
                sequence_number=0,
                heading="Introduction",
                content="Generated prose.",
                supporting_chunk_ids=[chunk.id],
            )
        ],
    )
    await db_session.commit()
    ValidationResultRepository(db_session).create(
        article_id=article.id, passed=True, checks=[{"name": "x", "passed": True, "details": "ok"}]
    )
    await db_session.commit()

    episodes_status = ProcessingStatus.READY_FOR_REVIEW
    from app.repositories.episode_repository import EpisodeRepository

    ep_repo = EpisodeRepository(db_session)
    ep = await ep_repo.get_by_id(episode.id)
    ep_repo.set_status(ep, episodes_status)
    await db_session.commit()

    client = await _client(db_session)
    try:
        response = await client.get(f"/api/v1/episodes/{episode.id}/article")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "The Article"
    assert body["episode_status"] == "READY_FOR_REVIEW"
    assert len(body["sections"]) == 1
    section = body["sections"][0]
    assert section["heading"] == "Introduction"
    assert len(section["supporting_chunks"]) == 1
    assert section["supporting_chunks"][0]["start_ms"] == 1_000
    assert section["supporting_chunks"][0]["end_ms"] == 5_000
    assert body["latest_validation"]["passed"] is True
    assert body["latest_validation"]["checks"][0]["name"] == "x"
