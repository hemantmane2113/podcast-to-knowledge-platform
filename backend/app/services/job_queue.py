import uuid

from arq import ArqRedis, create_pool
from arq.connections import RedisSettings

from app.config import get_settings

_TRANSCRIPT_INGESTION_TASK = "ingest_episode_transcript"


class JobQueue:
    """Isolates the rest of the app from arq's specific enqueue API —
    services depend on this, not on arq directly."""

    def __init__(self, redis: ArqRedis):
        self._redis = redis

    async def enqueue_transcript_ingestion(
        self, *, episode_id: uuid.UUID, job_id: uuid.UUID
    ) -> None:
        await self._redis.enqueue_job(_TRANSCRIPT_INGESTION_TASK, str(episode_id), str(job_id))


_pool: ArqRedis | None = None


async def get_job_queue() -> JobQueue:
    """FastAPI dependency: a process-wide arq redis pool, lazily created."""
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    return JobQueue(_pool)
