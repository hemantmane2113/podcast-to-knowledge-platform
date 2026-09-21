import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ArticleNotFoundError, EpisodeNotFoundError
from app.models.article import Article
from app.models.chunk import Chunk
from app.models.episode import Episode
from app.models.processing_job import JobType, ProcessingJob
from app.repositories.article_repository import ArticleRepository
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.episode_repository import EpisodeRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.repositories.transcript_repository import TranscriptRepository
from app.services.job_queue import JobQueue


class ArticleService:
    """API -> ArticleService -> repositories/job queue (Phase I: the
    review surface). Mirrors EpisodeService's shape -- reads assemble
    what a reviewer needs to see, `request_generation` only ever creates
    a job and enqueues it, never runs the AI pipeline inline (that's the
    worker's job, app/worker/tasks.py::generate_article).
    """

    def __init__(self, session: AsyncSession, job_queue: JobQueue):
        self._session = session
        self._job_queue = job_queue
        self._episodes = EpisodeRepository(session)
        self._articles = ArticleRepository(session)
        self._transcripts = TranscriptRepository(session)
        self._chunks = ChunkRepository(session)
        self._jobs = ProcessingJobRepository(session)

    async def get_article_with_chunks(self, episode_id: uuid.UUID) -> tuple[Episode, Article, list[Chunk]]:
        episode = await self._episodes.get_by_id(episode_id)
        if episode is None:
            raise EpisodeNotFoundError(f"Episode {episode_id} not found")

        article = await self._articles.get_by_episode_id(episode_id)
        if article is None:
            raise ArticleNotFoundError(f"Episode {episode_id} has no generated article yet")

        transcript = await self._transcripts.get_by_episode_id(episode_id)
        chunks = await self._chunks.get_by_transcript_id(transcript.id) if transcript else []
        return episode, article, chunks

    async def request_generation(self, episode_id: uuid.UUID) -> ProcessingJob:
        episode = await self._episodes.get_by_id(episode_id)
        if episode is None:
            raise EpisodeNotFoundError(f"Episode {episode_id} not found")

        job = self._jobs.create(episode_id=episode_id, job_type=JobType.ARTICLE_GENERATION)
        await self._session.commit()
        await self._job_queue.enqueue_article_generation(episode_id=episode_id, job_id=job.id)
        return job
