import uuid
from datetime import UTC, datetime

from httpx import ASGITransport, AsyncClient
from sqlalchemy import inspect
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
from app.repositories.topic_repository import TopicCandidate, TopicRepository
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


async def _seed_completed_article(
    session: AsyncSession, episode: Episode, transcript: Transcript, *, is_draft: bool = False
) -> None:
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
        episode_id=episode.id,
        title="p",
        introduction_summary="i",
        conclusion_summary="c",
        sections=[],
        is_draft=is_draft,
    )
    await session.commit()
    await ArticleRepository(session).replace(
        episode_id=episode.id,
        article_plan_id=plan.id,
        title="The Article",
        revision_count=0,
        sections=[],
        is_draft=is_draft,
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


# --- request_regeneration / POST regenerate-article -----------------------------------


async def test_regenerate_article_unknown_episode_returns_404(db_session: AsyncSession) -> None:
    client = await _client(db_session)
    try:
        response = await client.post(f"/api/v1/episodes/{uuid.uuid4()}/regenerate-article")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 404
    assert response.json()["code"] == "EPISODE_NOT_FOUND"


async def test_regenerate_article_enqueues_a_fresh_job_when_nothing_exists_yet(db_session: AsyncSession) -> None:
    # Nothing to delete -- confirms request_regeneration works fine when
    # there's no ArticlePlan yet (a first-ever "generate", just via the
    # explicit endpoint).
    episode, _ = await _seed_episode_and_transcript(db_session)
    queue = FakeJobQueue()
    client = await _client(db_session, queue)
    try:
        response = await client.post(f"/api/v1/episodes/{episode.id}/regenerate-article")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 202
    body = response.json()
    assert body["episode_id"] == str(episode.id)
    assert uuid.UUID(body["job_id"])
    assert len(queue.enqueued_article_generation) == 1
    assert queue.enqueued_article_generation[0][0] == episode.id


async def test_regenerate_article_keeps_the_live_article_untouched_and_removes_only_a_stale_draft(
    db_session: AsyncSession,
) -> None:
    """Live/Draft Article Workflow: a regeneration must never touch the
    published episode's live Article/ArticlePlan/ArticleSections/
    ValidationResults -- only a stale DRAFT (left over from an earlier
    draft run) is removed, so app/ai/nodes/planning.py's resumability
    check is forced to call the planner again for the new draft run
    instead of silently reusing stale draft state. This replaces the old
    pre-Live/Draft-Workflow test of the same shape, which asserted the
    (now intentionally different) opposite: that regeneration deleted the
    live article outright."""
    episode, transcript = await _seed_episode_and_transcript(db_session)

    chunk = (
        await ChunkRepository(db_session).replace_all(
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
    await db_session.commit()

    await TopicRepository(db_session).replace_all(
        transcript_id=transcript.id,
        episode_id=episode.id,
        candidates=[TopicCandidate(sequence_number=0, title="Topic A", summary="s", chunk_ids=[chunk.id])],
    )
    await db_session.commit()

    plan_repo = ArticlePlanRepository(db_session)
    article_repo = ArticleRepository(db_session)
    validation_repo = ValidationResultRepository(db_session)

    # The published episode's LIVE article -- must survive regeneration
    # completely untouched.
    live_plan = await plan_repo.replace(
        episode_id=episode.id, title="live p", introduction_summary="i", conclusion_summary="c", sections=[]
    )
    await db_session.commit()
    live_article = await article_repo.replace(
        episode_id=episode.id,
        article_plan_id=live_plan.id,
        title="The Live Article",
        revision_count=0,
        sections=[
            ArticleSectionCandidate(
                sequence_number=0, heading="Intro", content="Some prose.", supporting_chunk_ids=[chunk.id]
            )
        ],
    )
    await db_session.commit()
    validation_repo.create(article_id=live_article.id, passed=True, checks=[])
    await db_session.commit()
    live_article_id = live_article.id

    # A stale DRAFT left over from an earlier draft run -- this is the one
    # thing request_regeneration should remove.
    stale_draft_plan = await plan_repo.replace(
        episode_id=episode.id,
        title="stale draft p",
        introduction_summary="i",
        conclusion_summary="c",
        sections=[],
        is_draft=True,
    )
    await db_session.commit()
    await article_repo.replace(
        episode_id=episode.id,
        article_plan_id=stale_draft_plan.id,
        title="Stale Draft Article",
        revision_count=0,
        sections=[],
        is_draft=True,
    )
    await db_session.commit()

    jobs = ProcessingJobRepository(db_session)
    completed_job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    await db_session.commit()
    jobs.mark_completed(completed_job)
    await db_session.commit()

    queue = FakeJobQueue()
    client = await _client(db_session, queue)
    try:
        response = await client.post(f"/api/v1/episodes/{episode.id}/regenerate-article")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 202
    new_job_id = uuid.UUID(response.json()["job_id"])
    assert new_job_id != completed_job.id
    assert len(queue.enqueued_article_generation) == 1
    assert queue.enqueued_article_generation[0] == (episode.id, new_job_id)

    # The LIVE plan/article/sections/validation are all still exactly as
    # they were -- request_regeneration never touches them.
    still_live_plan = await plan_repo.get_by_episode_id(episode.id, is_draft=False)
    assert still_live_plan is not None
    assert still_live_plan.id == live_plan.id
    still_live_article = await article_repo.get_by_episode_id(episode.id, is_draft=False)
    assert still_live_article is not None
    assert still_live_article.id == live_article_id
    assert len(still_live_article.sections) == 1
    assert await validation_repo.get_latest_by_article_id(live_article_id) is not None

    # The stale DRAFT plan/article are gone, through the FK ON DELETE
    # CASCADE triggered by deleting the draft plan.
    assert await plan_repo.get_by_episode_id(episode.id, is_draft=True) is None
    assert await article_repo.get_by_episode_id(episode.id, is_draft=True) is None

    # Topics and Chunks are untouched -- neither is reached by that cascade.
    topics = await TopicRepository(db_session).get_by_transcript_id(transcript.id)
    assert len(topics) == 1
    chunks = await ChunkRepository(db_session).get_by_transcript_id(transcript.id)
    assert len(chunks) == 1

    # The completed job stays in history; a fresh PENDING job was created.
    historical = await jobs.get_by_id(completed_job.id)
    assert historical is not None
    assert historical.status == JobStatus.COMPLETED
    fresh_job = await jobs.get_by_id(new_job_id)
    assert fresh_job is not None
    assert fresh_job.status == JobStatus.PENDING


async def test_regenerate_article_returns_active_job_without_deleting_anything(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    await _seed_completed_article(db_session, episode, transcript)

    jobs = ProcessingJobRepository(db_session)
    active_job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    await db_session.commit()
    assert active_job.status == JobStatus.PENDING

    queue = FakeJobQueue()
    client = await _client(db_session, queue)
    try:
        response = await client.post(f"/api/v1/episodes/{episode.id}/regenerate-article")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 202
    assert response.json()["job_id"] == str(active_job.id)
    assert len(queue.enqueued_article_generation) == 0  # no duplicate job enqueued

    # The active-job check must short-circuit BEFORE the plan/article are
    # ever touched -- regeneration must never race a pipeline already in
    # flight for this episode.
    assert await ArticlePlanRepository(db_session).get_by_episode_id(episode.id) is not None
    assert await ArticleRepository(db_session).get_by_episode_id(episode.id) is not None


async def test_regenerate_article_after_a_failed_job_creates_a_fresh_job_and_keeps_history(
    db_session: AsyncSession,
) -> None:
    episode, _ = await _seed_episode_and_transcript(db_session)
    jobs = ProcessingJobRepository(db_session)
    failed_job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    await db_session.commit()
    jobs.mark_failed(failed_job, "some transient failure")
    await db_session.commit()

    queue = FakeJobQueue()
    client = await _client(db_session, queue)
    try:
        response = await client.post(f"/api/v1/episodes/{episode.id}/regenerate-article")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 202
    new_job_id = uuid.UUID(response.json()["job_id"])
    assert new_job_id != failed_job.id
    assert len(queue.enqueued_article_generation) == 1

    historical = await jobs.get_by_id(failed_job.id)
    assert historical is not None
    assert historical.status == JobStatus.FAILED


async def test_request_regeneration_expires_stale_already_loaded_article_objects(db_session: AsyncSession) -> None:
    """Regression guard for the identity-map risk: ArticlePlanRepository
    .delete_by_episode_id is a Core-level DELETE, which does not update
    ORM objects already sitting in the session's identity map on its own.
    If ArticleService.request_regeneration ever stopped expiring the
    session after that delete, an already-loaded Article object like
    `loaded_article` below would keep silently returning its pre-delete
    attribute values instead of reflecting the deletion -- exactly the
    class of bug this codebase has hit before with a Core DELETE and a
    stale identity map. Seeds a DRAFT article (not live) -- Live/Draft
    Article Workflow: request_regeneration only ever deletes a stale
    draft, so that's the object whose identity-map staleness is at risk
    here."""
    episode, transcript = await _seed_episode_and_transcript(db_session)
    await _seed_completed_article(db_session, episode, transcript, is_draft=True)

    article_repo = ArticleRepository(db_session)
    loaded_article = await article_repo.get_by_episode_id(episode.id, is_draft=True)
    assert loaded_article is not None

    service = ArticleService(db_session, FakeJobQueue())
    job = await service.request_regeneration(episode.id)

    assert job.status == JobStatus.PENDING
    # The stale reference must be marked expired -- otherwise, since every
    # attribute was already populated by the earlier get_by_episode_id
    # call, a plain read like `loaded_article.title` would silently keep
    # returning its cached pre-delete value forever, with no DB round trip
    # and no error, exactly the "stale identity map" failure mode this
    # guards against.
    assert inspect(loaded_article).expired

    # And the row is genuinely gone, not just marked expired: a fresh
    # query (not a reload of this specific stale instance) confirms it.
    assert await article_repo.get_by_episode_id(episode.id, is_draft=True) is None


# --- publish_article / POST publish ----------------------------------------------------


async def test_publish_unknown_episode_returns_404(db_session: AsyncSession) -> None:
    client = await _client(db_session)
    try:
        response = await client.post(f"/api/v1/episodes/{uuid.uuid4()}/publish")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 404
    assert response.json()["code"] == "EPISODE_NOT_FOUND"


async def test_publish_episode_without_article_returns_article_not_found(db_session: AsyncSession) -> None:
    episode, _ = await _seed_episode_and_transcript(db_session)
    client = await _client(db_session)
    try:
        response = await client.post(f"/api/v1/episodes/{episode.id}/publish")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 404
    assert response.json()["code"] == "ARTICLE_NOT_FOUND"


async def test_publish_fails_when_latest_validation_has_not_passed(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    await _seed_completed_article(db_session, episode, transcript)
    article = await ArticleRepository(db_session).get_by_episode_id(episode.id)
    ValidationResultRepository(db_session).create(
        article_id=article.id, passed=False, checks=[{"name": "x", "passed": False, "details": "bad"}]
    )
    await db_session.commit()

    client = await _client(db_session)
    try:
        response = await client.post(f"/api/v1/episodes/{episode.id}/publish")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 409
    assert response.json()["code"] == "ARTICLE_VALIDATION_NOT_PASSED"

    fresh = await db_session.get(Episode, episode.id)
    assert fresh.status != ProcessingStatus.PUBLISHED
    assert fresh.article_published_at is None


async def test_publish_fails_when_no_validation_result_exists_yet(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    await _seed_completed_article(db_session, episode, transcript)
    # No ValidationResult row created at all.

    client = await _client(db_session)
    try:
        response = await client.post(f"/api/v1/episodes/{episode.id}/publish")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 409
    assert response.json()["code"] == "ARTICLE_VALIDATION_NOT_PASSED"


async def test_publish_succeeds_and_sets_status_and_published_at(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    await _seed_completed_article(db_session, episode, transcript)
    article = await ArticleRepository(db_session).get_by_episode_id(episode.id)
    ValidationResultRepository(db_session).create(
        article_id=article.id, passed=True, checks=[{"name": "x", "passed": True, "details": "ok"}]
    )
    await db_session.commit()

    before = datetime.now(UTC)
    client = await _client(db_session)
    try:
        response = await client.post(f"/api/v1/episodes/{episode.id}/publish")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "PUBLISHED"
    assert body["article_published_at"] is not None
    published_at = datetime.fromisoformat(body["article_published_at"])
    assert published_at >= before

    fresh = await db_session.get(Episode, episode.id)
    assert fresh.status == ProcessingStatus.PUBLISHED
    assert fresh.article_published_at is not None


async def test_publish_does_not_modify_article_plan_or_sections(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    chunk = (
        await ChunkRepository(db_session).replace_all(
            transcript_id=transcript.id,
            episode_id=episode.id,
            candidates=[
                ChunkCandidate(
                    sequence_number=0, text="t", start_ms=0, end_ms=1_000, source_segment_ids=[], token_count=5
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
                sequence_number=0, heading="Intro", content="Original prose.", supporting_chunk_ids=[chunk.id]
            )
        ],
    )
    await db_session.commit()
    ValidationResultRepository(db_session).create(article_id=article.id, passed=True, checks=[])
    await db_session.commit()
    original_article_id = article.id
    original_plan_id = plan.id

    client = await _client(db_session)
    try:
        response = await client.post(f"/api/v1/episodes/{episode.id}/publish")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 200

    refreshed_plan = await ArticlePlanRepository(db_session).get_by_episode_id(episode.id)
    assert refreshed_plan.id == original_plan_id  # never replaced

    refreshed_article = await ArticleRepository(db_session).get_by_episode_id(episode.id)
    assert refreshed_article.id == original_article_id  # never replaced
    assert refreshed_article.revision_count == 0
    assert len(refreshed_article.sections) == 1
    assert refreshed_article.sections[0].content == "Original prose."
    assert refreshed_article.sections[0].heading == "Intro"


async def test_publish_is_idempotent_when_already_published(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    await _seed_completed_article(db_session, episode, transcript)
    article = await ArticleRepository(db_session).get_by_episode_id(episode.id)
    ValidationResultRepository(db_session).create(article_id=article.id, passed=True, checks=[])
    await db_session.commit()

    client = await _client(db_session)
    try:
        first = await client.post(f"/api/v1/episodes/{episode.id}/publish")
        second = await client.post(f"/api/v1/episodes/{episode.id}/publish")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["status"] == "PUBLISHED"
    # Republishing must not move the timestamp forward (no second
    # "publication record" -- there is only Episode.status/article_published_at).
    assert second.json()["article_published_at"] == first.json()["article_published_at"]


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
