"""arq task(s). The worker process (`arq app.worker.settings.WorkerSettings`)
is the only thing that calls IngestionService — API routes only ever create
an Episode + ProcessingJob and enqueue this task (locked async-ingestion
decision).
"""

import logging
import uuid

from app.core.db import get_sessionmaker
from app.models.episode import ProcessingStatus
from app.repositories.episode_repository import EpisodeRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.services.ingestion_service import IngestionService

logger = logging.getLogger(__name__)

MAX_TRIES = 3
_GENERIC_FAILURE_MESSAGE = "Transcript ingestion failed. See server logs for details."


async def ingest_episode_transcript(ctx: dict, episode_id: str, job_id: str) -> None:
    """arq entrypoint. `ctx["provider"]` is a shared TranscriptProvider set
    up once per worker process by app.worker.settings.startup (not
    per-task, so HTTP connections are reused). `ctx["job_try"]` is arq's
    1-indexed attempt counter.

    Retry policy: an exception with `.retryable is False` (bad URL, auth
    failure, "no transcript for this video", malformed response) is never
    retried, even on attempt 1 -- retrying wouldn't change the outcome.
    Anything else is retried up to MAX_TRIES; only the final attempt's
    failure is persisted as the episode/job's terminal FAILED state.
    """
    episode_uuid = uuid.UUID(episode_id)
    job_uuid = uuid.UUID(job_id)

    provider = ctx.get("provider")
    session_factory = get_sessionmaker()

    async with session_factory() as session:
        if provider is None:
            await _mark_failed(
                session,
                episode_uuid,
                job_uuid,
                "Transcript provider is not configured (SUPADATA_API_KEY missing).",
            )
            return

        service = IngestionService(session, provider)
        try:
            await service.run(episode_uuid, job_uuid)
        except Exception as exc:
            await session.rollback()

            is_final_attempt = ctx.get("job_try", 1) >= MAX_TRIES
            retryable = getattr(exc, "retryable", True)

            if retryable and not is_final_attempt:
                logger.warning(
                    "Ingestion attempt %s failed for episode %s (will retry): %s",
                    ctx.get("job_try"),
                    episode_id,
                    _safe_message(exc),
                )
                raise  # arq retries the task

            logger.error(
                "Ingestion failed permanently for episode %s after %s attempt(s): %s",
                episode_id,
                ctx.get("job_try"),
                _safe_message(exc),
            )
            await _mark_failed(session, episode_uuid, job_uuid, _safe_message(exc))


async def _mark_failed(session, episode_id: uuid.UUID, job_id: uuid.UUID, message: str) -> None:
    episodes = EpisodeRepository(session)
    jobs = ProcessingJobRepository(session)

    episode = await episodes.get_by_id(episode_id)
    job = await jobs.get_by_id(job_id)

    if episode is not None:
        episodes.set_status(episode, ProcessingStatus.FAILED, last_error=message)
    if job is not None:
        jobs.mark_failed(job, message)

    await session.commit()


def _safe_message(exc: Exception) -> str:
    """Only ever surfaces our own exception `.message` text (which we
    control and never populate with a provider's raw response or a
    credential) -- anything else becomes a generic string, never str(exc).
    """
    message = getattr(exc, "message", None)
    return message if isinstance(message, str) else _GENERIC_FAILURE_MESSAGE
