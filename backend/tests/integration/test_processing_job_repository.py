"""Regression tests for ProcessingJobRepository.get_active_job()'s
MultipleResultsFound bug: nothing enforces uniqueness on
(episode_id, job_type) at the DB level, so more than one PENDING/RUNNING
row can genuinely exist for the same episode (e.g. a worker process dying
mid-job without going through its own except-block leaves a row stuck
active forever). The query previously had no ORDER BY/LIMIT, so
scalar_one_or_none() raised MultipleResultsFound the moment a second
active row existed -- reproduced directly here against a real Postgres,
not just reasoned about.
"""

import datetime as dt
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.episode import Episode
from app.models.processing_job import JobStatus, JobType, ProcessingJob
from app.repositories.episode_repository import EpisodeRepository
from app.repositories.processing_job_repository import ProcessingJobRepository

VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
VIDEO_ID = "dQw4w9WgXcQ"


async def _seed_episode(session: AsyncSession) -> Episode:
    episode = EpisodeRepository(session).create(youtube_video_id=VIDEO_ID, youtube_url=VIDEO_URL)
    await session.commit()
    return episode


def _job(episode_id: uuid.UUID, *, status: JobStatus, created_at: dt.datetime) -> ProcessingJob:
    return ProcessingJob(
        id=uuid.uuid4(),
        episode_id=episode_id,
        job_type=JobType.ARTICLE_GENERATION,
        status=status,
        created_at=created_at,
    )


async def test_get_active_job_returns_none_with_only_historical_failed_jobs(
    db_session: AsyncSession,
) -> None:
    """(a) Multiple historical FAILED jobs -- none are active, so this must
    return None (and, critically, must not raise)."""
    episode = await _seed_episode(db_session)
    jobs = ProcessingJobRepository(db_session)
    now = dt.datetime.now(dt.UTC)
    db_session.add_all(
        [
            _job(episode.id, status=JobStatus.FAILED, created_at=now - dt.timedelta(minutes=10)),
            _job(episode.id, status=JobStatus.FAILED, created_at=now - dt.timedelta(minutes=5)),
            _job(episode.id, status=JobStatus.FAILED, created_at=now),
        ]
    )
    await db_session.commit()

    active = await jobs.get_active_job(episode.id, JobType.ARTICLE_GENERATION)

    assert active is None


async def test_get_active_job_returns_the_newest_active_job(db_session: AsyncSession) -> None:
    """(b) Multiple active (PENDING/RUNNING) rows exist -- must not raise
    MultipleResultsFound, and must return the newest one."""
    episode = await _seed_episode(db_session)
    jobs = ProcessingJobRepository(db_session)
    now = dt.datetime.now(dt.UTC)
    older = _job(episode.id, status=JobStatus.PENDING, created_at=now - dt.timedelta(minutes=5))
    newer = _job(episode.id, status=JobStatus.RUNNING, created_at=now)
    db_session.add_all([older, newer])
    await db_session.commit()

    active = await jobs.get_active_job(episode.id, JobType.ARTICLE_GENERATION)

    assert active is not None
    assert active.id == newer.id


async def test_get_active_job_ignores_historical_failed_jobs_and_returns_the_active_one(
    db_session: AsyncSession,
) -> None:
    """(c) A mix of historical FAILED jobs and one active job -- the
    active job is returned, and the FAILED rows are neither touched nor
    cause a MultipleResultsFound (there's more than one non-active row
    too, which alone wouldn't have crashed the old query, but this
    exercises the full mixed-history shape a real production DB has)."""
    episode = await _seed_episode(db_session)
    jobs = ProcessingJobRepository(db_session)
    now = dt.datetime.now(dt.UTC)
    active_job = _job(episode.id, status=JobStatus.RUNNING, created_at=now)
    db_session.add_all(
        [
            _job(episode.id, status=JobStatus.FAILED, created_at=now - dt.timedelta(minutes=20)),
            _job(episode.id, status=JobStatus.FAILED, created_at=now - dt.timedelta(minutes=10)),
            active_job,
        ]
    )
    await db_session.commit()

    active = await jobs.get_active_job(episode.id, JobType.ARTICLE_GENERATION)

    assert active is not None
    assert active.id == active_job.id

    # Historical FAILED jobs are untouched -- still present, still FAILED.
    result = await db_session.execute(
        select(ProcessingJob).where(
            ProcessingJob.episode_id == episode.id, ProcessingJob.status == JobStatus.FAILED
        )
    )
    assert len(result.scalars().all()) == 2
