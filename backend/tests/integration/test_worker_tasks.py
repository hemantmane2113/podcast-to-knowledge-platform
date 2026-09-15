import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.exceptions import TranscriptProviderAuthError, TranscriptProviderError
from app.models.chunk import Chunk
from app.models.episode import Episode, ProcessingStatus
from app.models.processing_job import JobStatus, JobType, ProcessingJob
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.episode_repository import EpisodeRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.repositories.transcript_repository import TranscriptRepository
from app.schemas.transcript import NormalizedSegment, NormalizedTranscript
from app.worker.tasks import MAX_TRIES, ingest_episode_transcript, process_transcript
from tests.conftest import TEST_DATABASE_URL
from tests.fakes import FakeJobQueue, StubProvider

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
    queue = FakeJobQueue()
    ctx = {"job_try": 1, "provider": StubProvider(transcript=transcript), "job_queue": queue}

    await ingest_episode_transcript(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.TRANSCRIPT_FETCHED
    assert refreshed_job.status == JobStatus.COMPLETED

    # Auto-chains into transcript processing (§3A.10) -- a new job was
    # created and enqueued, not left for something else to trigger.
    assert len(queue.enqueued_processing) == 1
    assert queue.enqueued_processing[0][0] == episode.id


async def test_successful_ingestion_without_job_queue_does_not_crash(
    db_session: AsyncSession,
) -> None:
    # A worker ctx missing "job_queue" (shouldn't happen in practice --
    # app.worker.settings always sets it) must degrade to "ingestion
    # succeeded, chaining skipped", never raise.
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


# --- process_transcript (Phase 3A) ------------------------------------------------


async def _seed_episode_with_transcript(session: AsyncSession):
    from app.models.episode import Episode
    from app.models.transcript import Transcript
    from app.models.transcript_segment import TranscriptSegment

    episode = Episode(
        id=uuid.uuid4(),
        youtube_video_id=VIDEO_ID,
        youtube_url=VIDEO_URL,
        status=ProcessingStatus.TRANSCRIPT_FETCHED,
    )
    session.add(episode)
    transcript = Transcript(id=uuid.uuid4(), episode_id=episode.id, language="en")
    session.add(transcript)
    transcript.segments = [
        TranscriptSegment(
            id=uuid.uuid4(),
            transcript_id=transcript.id,
            sequence_number=i,
            text=f"Sentence number {i}.",
            start_ms=i * 4000,
            duration_ms=3000,
        )
        for i in range(5)
    ]
    jobs = ProcessingJobRepository(session)
    job = jobs.create(episode_id=episode.id, job_type=JobType.TRANSCRIPT_PROCESSING)
    await session.commit()
    return episode, transcript, job


async def _reload_chunks(transcript_id) -> list[Chunk]:
    engine = create_async_engine(TEST_DATABASE_URL)
    try:
        async with AsyncSession(bind=engine) as fresh_session:
            return await ChunkRepository(fresh_session).get_by_transcript_id(transcript_id)
    finally:
        await engine.dispose()


async def test_process_transcript_success_persists_chunks(db_session: AsyncSession) -> None:
    episode, transcript, job = await _seed_episode_with_transcript(db_session)
    ctx = {"job_try": 1}

    await process_transcript(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.CHUNKING
    assert refreshed_job.status == JobStatus.COMPLETED

    chunks = await _reload_chunks(transcript.id)
    assert len(chunks) >= 1
    all_segment_ids = {s.id for s in transcript.segments}
    for chunk in chunks:
        assert set(chunk.source_segment_ids) <= all_segment_ids


async def test_process_transcript_is_idempotent_on_rerun(db_session: AsyncSession) -> None:
    episode, transcript, job = await _seed_episode_with_transcript(db_session)
    ctx = {"job_try": 1}

    await process_transcript(ctx, str(episode.id), str(job.id))
    first_chunks = await _reload_chunks(transcript.id)

    # Re-run against the same transcript (e.g. a manual reprocess).
    jobs = ProcessingJobRepository(db_session)
    second_job = jobs.create(episode_id=episode.id, job_type=JobType.TRANSCRIPT_PROCESSING)
    await db_session.commit()
    await process_transcript({"job_try": 1}, str(episode.id), str(second_job.id))
    second_chunks = await _reload_chunks(transcript.id)

    assert len(second_chunks) == len(first_chunks)
    assert [c.text for c in second_chunks] == [c.text for c in first_chunks]
    assert [c.sequence_number for c in second_chunks] == [c.sequence_number for c in first_chunks]


async def test_process_transcript_missing_transcript_fails_without_raising(
    db_session: AsyncSession,
) -> None:
    episodes = EpisodeRepository(db_session)
    jobs = ProcessingJobRepository(db_session)
    episode = episodes.create(youtube_video_id=VIDEO_ID, youtube_url=VIDEO_URL)
    job = jobs.create(episode_id=episode.id, job_type=JobType.TRANSCRIPT_PROCESSING)
    await db_session.commit()

    # No transcript exists for this episode -- non-retryable (TranscriptNotFoundError).
    await process_transcript({"job_try": 1}, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.FAILED
    assert refreshed_job.status == JobStatus.FAILED
