import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import TranscriptProviderError
from app.models.episode import ProcessingStatus
from app.providers.transcript.base import TranscriptProvider
from app.repositories.episode_repository import EpisodeRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.repositories.transcript_repository import TranscriptRepository


class EpisodeOrJobNotFoundError(Exception):
    """The episode/job row the worker was told to process doesn't exist.
    Not retried — retrying wouldn't make a missing row appear."""


class IngestionService:
    """Does one ingestion attempt: call the provider, normalize, persist.

    Only ever called from the worker (app/worker/tasks.py), never from an
    API route — this is the async side of ingestion (locked decision: the
    API returns immediately after creating the Episode + ProcessingJob).
    Raises on any failure; the caller (the worker task) decides whether
    that failure is retryable and owns the final FAILED status transition,
    so this class doesn't need to know about attempt counts.
    """

    def __init__(self, session: AsyncSession, provider: TranscriptProvider):
        self._session = session
        self._provider = provider
        self._episodes = EpisodeRepository(session)
        self._transcripts = TranscriptRepository(session)
        self._jobs = ProcessingJobRepository(session)

    async def run(self, episode_id: uuid.UUID, job_id: uuid.UUID) -> None:
        episode = await self._episodes.get_by_id(episode_id)
        job = await self._jobs.get_by_id(job_id)
        if episode is None or job is None:
            raise EpisodeOrJobNotFoundError(f"episode={episode_id} job={job_id}")

        self._jobs.mark_running(job)
        await self._session.commit()

        transcript = await self._provider.get_transcript(episode.youtube_url)
        metadata = await self._get_metadata_best_effort(episode.youtube_url)

        if metadata is not None:
            self._episodes.apply_metadata(episode, metadata)
        self._transcripts.create_with_segments(episode.id, transcript)
        self._episodes.set_status(episode, ProcessingStatus.TRANSCRIPT_FETCHED)
        self._jobs.mark_completed(job)

        # One commit for the whole outcome: transcript + all its segments +
        # episode status all land together, or (on any prior exception)
        # none of them do (PRODUCT_SPEC.md §75 — no partial persistence).
        await self._session.commit()

    async def _get_metadata_best_effort(self, youtube_url: str):
        # Metadata enriches the episode but isn't the Phase 2 deliverable
        # (the transcript is) -- a metadata-only failure shouldn't sink an
        # otherwise successful transcript fetch.
        try:
            return await self._provider.get_metadata(youtube_url)
        except TranscriptProviderError:
            return None
