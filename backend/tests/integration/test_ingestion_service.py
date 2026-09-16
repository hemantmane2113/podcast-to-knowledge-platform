import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    TranscriptProviderAuthError,
    TranscriptProviderNotFoundError,
)
from app.models.episode import Episode, ProcessingStatus
from app.models.processing_job import JobStatus, JobType, ProcessingJob
from app.repositories.episode_repository import EpisodeRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.repositories.transcript_repository import TranscriptRepository
from app.schemas.transcript import EpisodeMetadata, NormalizedSegment, NormalizedTranscript
from app.services.ingestion_service import EpisodeOrJobNotFoundError, IngestionService
from tests.fakes import StubProvider

VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
VIDEO_ID = "dQw4w9WgXcQ"


async def _seed_episode_and_job(session: AsyncSession) -> tuple[Episode, ProcessingJob]:
    episodes = EpisodeRepository(session)
    jobs = ProcessingJobRepository(session)
    episode = episodes.create(youtube_video_id=VIDEO_ID, youtube_url=VIDEO_URL)
    job = jobs.create(episode_id=episode.id, job_type=JobType.TRANSCRIPT_INGESTION)
    await session.commit()
    return episode, job


async def test_run_persists_transcript_and_metadata(db_session: AsyncSession) -> None:
    episode, job = await _seed_episode_and_job(db_session)

    transcript = NormalizedTranscript(
        language="en",
        segments=[
            NormalizedSegment(text="first", start_ms=0, duration_ms=1000, speaker=None),
            NormalizedSegment(text="second", start_ms=1000, duration_ms=1200, speaker=None),
        ],
    )
    metadata = EpisodeMetadata(title="A Title", channel_name="A Channel", duration_seconds=120)
    provider = StubProvider(transcript=transcript, metadata=metadata)

    service = IngestionService(db_session, provider)
    await service.run(episode.id, job.id)

    episodes = EpisodeRepository(db_session)
    transcripts = TranscriptRepository(db_session)
    jobs = ProcessingJobRepository(db_session)

    refreshed_episode = await episodes.get_by_id(episode.id)
    assert refreshed_episode.status == ProcessingStatus.TRANSCRIPT_FETCHED
    assert refreshed_episode.title == "A Title"
    assert refreshed_episode.channel_name == "A Channel"
    assert refreshed_episode.duration_seconds == 120

    stored_transcript = await transcripts.get_by_episode_id(episode.id)
    assert stored_transcript is not None
    assert stored_transcript.language == "en"
    assert [s.text for s in stored_transcript.segments] == ["first", "second"]
    assert [s.sequence_number for s in stored_transcript.segments] == [0, 1]
    assert stored_transcript.segments[1].start_ms == 1000

    refreshed_job = await jobs.get_by_id(job.id)
    assert refreshed_job.status == JobStatus.COMPLETED
    assert refreshed_job.completed_at is not None


async def test_run_metadata_failure_does_not_fail_ingestion(db_session: AsyncSession) -> None:
    episode, job = await _seed_episode_and_job(db_session)

    transcript = NormalizedTranscript(
        language="en", segments=[NormalizedSegment(text="hi", start_ms=0, duration_ms=500)]
    )
    provider = StubProvider(
        transcript=transcript, metadata=TranscriptProviderNotFoundError("no metadata")
    )

    service = IngestionService(db_session, provider)
    await service.run(episode.id, job.id)

    episodes = EpisodeRepository(db_session)
    refreshed = await episodes.get_by_id(episode.id)
    assert refreshed.status == ProcessingStatus.TRANSCRIPT_FETCHED
    assert refreshed.title is None


async def test_run_transcript_failure_raises_and_leaves_no_partial_transcript(
    db_session: AsyncSession,
) -> None:
    episode, job = await _seed_episode_and_job(db_session)
    provider = StubProvider(transcript=TranscriptProviderAuthError("bad key"))

    service = IngestionService(db_session, provider)
    with pytest.raises(TranscriptProviderAuthError):
        await service.run(episode.id, job.id)

    transcripts = TranscriptRepository(db_session)
    assert await transcripts.get_by_episode_id(episode.id) is None


async def test_run_raises_for_unknown_episode_or_job(db_session: AsyncSession) -> None:
    provider = StubProvider(
        transcript=NormalizedTranscript(language="en", segments=[]),
    )
    service = IngestionService(db_session, provider)

    with pytest.raises(EpisodeOrJobNotFoundError):
        await service.run(uuid.uuid4(), uuid.uuid4())
