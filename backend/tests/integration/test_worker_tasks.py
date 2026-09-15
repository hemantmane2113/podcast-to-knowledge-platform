import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.exceptions import TranscriptProviderAuthError, TranscriptProviderError
from app.models.episode import Episode, ProcessingStatus
from app.models.processing_job import JobStatus, JobType, ProcessingJob
from app.repositories.episode_repository import EpisodeRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.schemas.transcript import NormalizedSegment, NormalizedTranscript
from app.worker.tasks import MAX_TRIES, ingest_episode_transcript
from tests.conftest import TEST_DATABASE_URL
from tests.fakes import StubProvider

VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
VIDEO_ID = "dQw4w9WgXcQ"


async def _seed_episode_and_job(session: AsyncSession):
    episodes = EpisodeRepository(session)
    jobs = ProcessingJobRepository(session)
    episode = episodes.create(youtube_video_id=VIDEO_ID, youtube_url=VIDEO_URL)
    job = jobs.create(episode_id=episode.id, job_type=JobType.TRANSCRIPT_INGESTION)
    await session.commit()
    return episode, job


async def _reload(episode_id, job_id) -> tuple[Episode, ProcessingJob]:
    """The worker task writes through its own session (a separate
    connection to the same database -- see app.worker.tasks). Read back
    through a brand-new engine/session rather than the seeding session's
    identity map, so this actually observes what the worker committed
    instead of the pre-worker in-memory objects."""
    engine = create_async_engine(TEST_DATABASE_URL)
    try:
        async with AsyncSession(bind=engine) as fresh_session:
            episode = await EpisodeRepository(fresh_session).get_by_id(episode_id)
            job = await ProcessingJobRepository(fresh_session).get_by_id(job_id)
            return episode, job
    finally:
        await engine.dispose()


async def test_successful_ingestion(db_session: AsyncSession) -> None:
    episode, job = await _seed_episode_and_job(db_session)
    transcript = NormalizedTranscript(
        language="en", segments=[NormalizedSegment(text="hi", start_ms=0, duration_ms=500)]
    )
    ctx = {"job_try": 1, "provider": StubProvider(transcript=transcript)}

    await ingest_episode_transcript(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.TRANSCRIPT_FETCHED
    assert refreshed_job.status == JobStatus.COMPLETED


async def test_non_retryable_failure_marks_failed_immediately_without_raising(
    db_session: AsyncSession,
) -> None:
    episode, job = await _seed_episode_and_job(db_session)
    ctx = {
        "job_try": 1,
        "provider": StubProvider(transcript=TranscriptProviderAuthError("bad key")),
    }

    # Must NOT raise -- an auth failure is never retried, even on the first attempt.
    await ingest_episode_transcript(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.FAILED
    assert refreshed_episode.last_error == "bad key"
    assert refreshed_job.status == JobStatus.FAILED
    assert refreshed_job.error_message == "bad key"


async def test_retryable_failure_raises_to_let_arq_retry_on_non_final_attempt(
    db_session: AsyncSession,
) -> None:
    episode, job = await _seed_episode_and_job(db_session)
    ctx = {
        "job_try": 1,  # first of MAX_TRIES attempts
        "provider": StubProvider(transcript=TranscriptProviderError("transient failure")),
    }

    with pytest.raises(TranscriptProviderError):
        await ingest_episode_transcript(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    # Not yet terminal -- arq is expected to retry this job.
    assert refreshed_episode.status == ProcessingStatus.INGESTING
    assert refreshed_job.status == JobStatus.RUNNING


async def test_missing_provider_marks_failed_without_raising(db_session: AsyncSession) -> None:
    episode, job = await _seed_episode_and_job(db_session)
    ctx = {"job_try": 1, "provider": None}  # e.g. SUPADATA_API_KEY not configured

    await ingest_episode_transcript(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.FAILED
    assert "not configured" in refreshed_episode.last_error
    assert refreshed_job.status == JobStatus.FAILED


async def test_retryable_failure_marks_failed_on_final_attempt(db_session: AsyncSession) -> None:
    episode, job = await _seed_episode_and_job(db_session)
    ctx = {
        "job_try": MAX_TRIES,  # last attempt
        "provider": StubProvider(transcript=TranscriptProviderError("still failing")),
    }

    # Must NOT raise -- attempts are exhausted, this is the terminal failure.
    await ingest_episode_transcript(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.FAILED
    assert refreshed_job.status == JobStatus.FAILED
    assert refreshed_job.error_message == "still failing"
