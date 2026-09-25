"""Verifies REAL arq retry behavior -- a real arq.worker.Worker running a
real production task (ingest_episode_transcript) against a real Redis,
not a re-invocation of the task function in isolation (every other test in
tests/integration/test_worker_tasks.py does that, which is exactly why the
original retry bug -- a bare `raise` that arq never actually retries --
went unnoticed through 341 passing tests: none of them exercised arq's own
Worker.run_job, only our own function bodies).

Requires a real, reachable Redis (REDIS_URL / TEST_REDIS_URL, matching
tests/conftest.py's TEST_DATABASE_URL pattern -- the same instance CI's
`redis` service container and local dev already provide). Uses a random
queue_name per test so this never collides with anything else using the
same Redis instance.
"""

import asyncio
import os
import uuid

import pytest
from arq import create_pool
from arq.connections import RedisSettings
from arq.worker import Worker
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.models.episode import ProcessingStatus
from app.models.processing_job import JobStatus, JobType
from app.repositories.episode_repository import EpisodeRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.providers.transcript.base import TranscriptProvider
from app.schemas.transcript import EpisodeMetadata, NormalizedSegment, NormalizedTranscript
from app.worker.tasks import ingest_episode_transcript
from tests.conftest import TEST_DATABASE_URL
from tests.fakes import FakeJobQueue

TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", os.environ.get("REDIS_URL", "redis://localhost:6379/0"))

VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
VIDEO_ID = "dQw4w9WgXcQ"

# Short defer for the test's own retryable failure -- proves the mechanism,
# not the specific configured RETRY_DEFER_SECONDS value (that's asserted
# separately, at the unit level, in test_worker_tasks.py).
_TEST_DEFER_SECONDS = 1


class _FlakyOnceProvider(TranscriptProvider):
    """Fails with a retryable error on the first call, succeeds on every
    call after that -- lets a real arq retry cycle be observed end to end
    against a real production task, not a synthetic dummy function."""

    def __init__(self, transcript: NormalizedTranscript):
        self._transcript = transcript
        self.call_count = 0

    async def get_transcript(self, video_url: str) -> NormalizedTranscript:
        self.call_count += 1
        if self.call_count == 1:
            from app.core.exceptions import TranscriptProviderError

            raise TranscriptProviderError("transient failure on first attempt")
        return self._transcript

    async def get_metadata(self, video_url: str) -> EpisodeMetadata:
        return EpisodeMetadata()


async def _reload(episode_id, job_id):
    engine = create_async_engine(TEST_DATABASE_URL)
    try:
        async with AsyncSession(bind=engine) as session:
            episode = await EpisodeRepository(session).get_by_id(episode_id)
            job = await ProcessingJobRepository(session).get_by_id(job_id)
            return episode, job
    finally:
        await engine.dispose()


async def test_real_arq_worker_retries_a_transient_ingestion_failure(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The end-to-end proof this whole fix is about: a real arq.worker.Worker,
    given the real ingest_episode_transcript task and a provider that fails
    once (retryable) then succeeds, must actually re-invoke the SAME job --
    not silently drop it, which is what the pre-fix bare `raise` did (see
    job 5f27dcad-b02a-4872-8a3d-639215ac77ce)."""
    import app.worker.tasks as tasks_module

    monkeypatch.setattr(tasks_module, "RETRY_DEFER_SECONDS", _TEST_DEFER_SECONDS)

    episodes = EpisodeRepository(db_session)
    jobs = ProcessingJobRepository(db_session)
    episode = episodes.create(youtube_video_id=VIDEO_ID, youtube_url=VIDEO_URL)
    job = jobs.create(episode_id=episode.id, job_type=JobType.TRANSCRIPT_INGESTION)
    await db_session.commit()

    transcript = NormalizedTranscript(
        language="en", segments=[NormalizedSegment(text="hi", start_ms=0, duration_ms=500)]
    )
    provider = _FlakyOnceProvider(transcript)
    queue_name = f"test-retry-{uuid.uuid4().hex}"

    redis = await create_pool(RedisSettings.from_dsn(TEST_REDIS_URL), default_queue_name=queue_name)
    try:
        await redis.enqueue_job(
            "ingest_episode_transcript", str(episode.id), str(job.id), _queue_name=queue_name
        )

        worker = Worker(
            functions=[ingest_episode_transcript],
            redis_pool=redis,
            queue_name=queue_name,
            ctx={"provider": provider, "job_queue": FakeJobQueue()},
            max_tries=3,
            job_timeout=30,
            burst=False,
            poll_delay=0.1,
            handle_signals=False,
        )

        run_task = asyncio.create_task(worker.main())
        try:
            for _ in range(100):  # up to ~10s
                await asyncio.sleep(0.1)
                if worker.jobs_complete >= 1 or worker.jobs_failed >= 1:
                    break
            await asyncio.sleep(0.3)  # let the final finish_job commit land
        finally:
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass
    finally:
        await redis.aclose()

    # The provider was actually called twice -- once failing, once
    # succeeding -- proving arq genuinely re-ran the task, not that it
    # merely reported success without redoing the work.
    assert provider.call_count == 2
    assert worker.jobs_retried == 1
    assert worker.jobs_complete == 1
    assert worker.jobs_failed == 0

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.TRANSCRIPT_FETCHED
    assert refreshed_job.status == JobStatus.COMPLETED


async def test_real_arq_worker_respects_max_tries_and_never_invokes_a_4th_attempt(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirms, empirically, what actually stops a repeatedly-retryable
    failure: our own application code (job_try >= MAX_TRIES -> _mark_failed,
    no further Retry raised), not arq's own max_tries guard -- by the time
    the task itself would exceed max_tries, our code has already stopped
    raising Retry, so arq never even reaches that guard (see the audit's
    empirical finding that arq's guard only fires on a WOULD-BE 4th
    invocation, never observed here). The task is invoked exactly
    MAX_TRIES=3 times, never a 4th, and -- because the final attempt's
    coroutine returns normally after _mark_failed rather than raising --
    arq itself records this as a normal completion (jobs_complete=1), even
    though Postgres correctly holds the terminal FAILED state."""
    import app.worker.tasks as tasks_module

    monkeypatch.setattr(tasks_module, "RETRY_DEFER_SECONDS", _TEST_DEFER_SECONDS)

    episodes = EpisodeRepository(db_session)
    jobs = ProcessingJobRepository(db_session)
    episode = episodes.create(youtube_video_id=f"always-fails-{uuid.uuid4().hex[:6]}", youtube_url=VIDEO_URL)
    job = jobs.create(episode_id=episode.id, job_type=JobType.TRANSCRIPT_INGESTION)
    await db_session.commit()

    class _AlwaysFailsProvider(TranscriptProvider):
        def __init__(self) -> None:
            self.call_count = 0

        async def get_transcript(self, video_url: str) -> NormalizedTranscript:
            self.call_count += 1
            from app.core.exceptions import TranscriptProviderError

            raise TranscriptProviderError("permanently down")

        async def get_metadata(self, video_url: str) -> EpisodeMetadata:
            return EpisodeMetadata()

    provider = _AlwaysFailsProvider()
    queue_name = f"test-maxtries-{uuid.uuid4().hex}"

    redis = await create_pool(RedisSettings.from_dsn(TEST_REDIS_URL), default_queue_name=queue_name)
    try:
        await redis.enqueue_job(
            "ingest_episode_transcript", str(episode.id), str(job.id), _queue_name=queue_name
        )

        worker = Worker(
            functions=[ingest_episode_transcript],
            redis_pool=redis,
            queue_name=queue_name,
            ctx={"provider": provider, "job_queue": FakeJobQueue()},
            max_tries=3,
            job_timeout=30,
            burst=False,
            poll_delay=0.1,
            handle_signals=False,
        )

        run_task = asyncio.create_task(worker.main())
        try:
            for _ in range(100):  # up to ~10s
                await asyncio.sleep(0.1)
                if provider.call_count >= 3:
                    break
            await asyncio.sleep(0.3)  # let the 3rd attempt's finish_job commit
        finally:
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass
    finally:
        await redis.aclose()

    # Our own app-level check (job_try >= MAX_TRIES=3 -> _mark_failed, no
    # further Retry) is what stops it -- so the provider is called exactly
    # 3 times, never a 4th. By design (see app/worker/tasks.py), the final
    # attempt's task coroutine does NOT raise once it has called
    # _mark_failed -- it returns normally -- so arq itself sees this as a
    # normal completion (jobs_complete=1, jobs_failed=0), even though our
    # own system has already recorded the terminal FAILED state in
    # Postgres. arq's own "jobs_failed" counter is not the source of truth
    # for whether this job succeeded from the application's point of view.
    assert provider.call_count == 3
    assert worker.jobs_retried == 2
    assert worker.jobs_complete == 1
    assert worker.jobs_failed == 0

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.FAILED
    assert refreshed_job.status == JobStatus.FAILED
