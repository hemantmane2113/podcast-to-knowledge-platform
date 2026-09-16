import uuid

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import EpisodeAlreadyProcessingError, EpisodeNotFoundError, TranscriptNotFoundError
from app.models.episode import Episode, ProcessingStatus
from app.models.processing_job import JobType
from app.models.transcript import Transcript
from app.repositories.episode_repository import EpisodeRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.repositories.transcript_repository import TranscriptRepository
from app.services.job_queue import JobQueue
from app.utils.youtube import extract_video_id


class EpisodeService:
    """API -> EpisodeService -> repositories/job queue. Owns the
    idempotent-create flow (PRODUCT_SPEC.md §51) and episode/transcript
    reads; does not talk to the transcript provider itself — that's
    app.services.ingestion_service, which only the worker calls."""

    def __init__(self, session: AsyncSession, job_queue: JobQueue):
        self._session = session
        self._job_queue = job_queue
        self._episodes = EpisodeRepository(session)
        self._jobs = ProcessingJobRepository(session)
        self._transcripts = TranscriptRepository(session)

    async def get_episode(self, episode_id: uuid.UUID) -> Episode:
        episode = await self._episodes.get_by_id(episode_id)
        if episode is None:
            raise EpisodeNotFoundError(f"Episode {episode_id} not found")
        return episode

    async def get_transcript(self, episode_id: uuid.UUID) -> Transcript:
        episode = await self.get_episode(episode_id)
        transcript = await self._transcripts.get_by_episode_id(episode_id)
        if transcript is None:
            raise TranscriptNotFoundError(
                f"Episode {episode_id} has no transcript yet (status={episode.status.value})"
            )
        return transcript

    async def create_episode(self, youtube_url: str) -> tuple[Episode, bool]:
        """Validates the URL, and idempotently creates+enqueues an episode.

        Returns (episode, created). Raises InvalidYouTubeURLError (via
        extract_video_id) for a bad URL, EpisodeAlreadyProcessingError if a
        transcript-ingestion job for this video is already running.
        """
        video_id = extract_video_id(youtube_url)

        existing = await self._episodes.get_by_youtube_video_id(video_id)
        if existing is not None:
            await self._reject_if_already_processing(existing)
            return existing, False

        episode = self._episodes.create(youtube_video_id=video_id, youtube_url=youtube_url)
        job = self._jobs.create(episode_id=episode.id, job_type=JobType.TRANSCRIPT_INGESTION)

        try:
            await self._session.commit()
        except IntegrityError:
            # Lost a race with a concurrent request for the same video: the
            # DB-level unique constraint on youtube_video_id is the final
            # guard (PRODUCT_SPEC.md §51). Recover by returning whatever the
            # winner created instead of erroring.
            await self._session.rollback()
            winner = await self._episodes.get_by_youtube_video_id(video_id)
            if winner is None:
                raise
            await self._reject_if_already_processing(winner)
            return winner, False

        await self._job_queue.enqueue_transcript_ingestion(episode_id=episode.id, job_id=job.id)
        return episode, True

    async def _reject_if_already_processing(self, episode: Episode) -> None:
        if episode.status != ProcessingStatus.INGESTING:
            return
        active_job = await self._jobs.get_active_job(episode.id, JobType.TRANSCRIPT_INGESTION)
        if active_job is not None:
            raise EpisodeAlreadyProcessingError(f"Episode {episode.id} is already being processed")
