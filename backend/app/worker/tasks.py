"""arq task(s). The worker process (`arq app.worker.settings.WorkerSettings`)
is the only thing that calls IngestionService/TranscriptProcessingService —
API routes only ever create an Episode + ProcessingJob and enqueue
ingest_episode_transcript (locked async-ingestion decision); chunking is
chained automatically after a successful ingestion (see bottom of
ingest_episode_transcript) rather than needing its own API trigger, since
Phase 3A has exactly one thing to do after a transcript exists.
"""

import logging
import uuid

from app.config import get_settings
from app.core.db import get_sessionmaker
from app.models.episode import ProcessingStatus
from app.models.processing_job import JobType
from app.repositories.episode_repository import EpisodeRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.services.chunking_service import ChunkingConfig
from app.services.ingestion_service import IngestionService
from app.services.transcript_processing_service import TranscriptProcessingService

logger = logging.getLogger(__name__)

MAX_TRIES = 3
_GENERIC_FAILURE_MESSAGE = "Transcript ingestion failed. See server logs for details."
_GENERIC_PROCESSING_FAILURE_MESSAGE = "Transcript processing failed. See server logs for details."


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
            return

        # Ingestion succeeded -- chain straight into cleaning + chunking,
        # reusing the same job/queue infrastructure rather than requiring
        # a separate trigger (§3A.10: "reuse the existing asynchronous job
        # infrastructure ... do not redesign the Phase 2 architecture").
        job_queue = ctx.get("job_queue")
        if job_queue is None:
            logger.error(
                "No job_queue in worker ctx -- cannot enqueue transcript processing for "
                "episode %s. Transcript was ingested successfully; processing was NOT "
                "triggered and needs to be started manually.",
                episode_id,
            )
            return

        jobs = ProcessingJobRepository(session)
        processing_job = jobs.create(episode_id=episode_uuid, job_type=JobType.TRANSCRIPT_PROCESSING)
        await session.commit()
        await job_queue.enqueue_transcript_processing(
            episode_id=episode_uuid, job_id=processing_job.id
        )


async def process_transcript(ctx: dict, episode_id: str, job_id: str) -> None:
    """arq entrypoint for Phase 3A cleaning + chunking. Enqueued
    automatically by ingest_episode_transcript above; not reachable any
    other way in Phase 3A (no API endpoint triggers it directly).

    Purely local computation (no external API calls, unlike ingestion), so
    failures are realistically bugs or bad state rather than transient
    provider errors -- still retried up to MAX_TRIES for robustness (e.g.
    a momentary DB connection issue), with the same "only the final
    attempt is a terminal FAILED" policy as ingestion.
    """
    episode_uuid = uuid.UUID(episode_id)
    job_uuid = uuid.UUID(job_id)

    settings = get_settings()
    config = ChunkingConfig.from_settings(settings)
    session_factory = get_sessionmaker()

    async with session_factory() as session:
        jobs = ProcessingJobRepository(session)
        job = await jobs.get_by_id(job_uuid)
        if job is not None:
            jobs.mark_running(job)
            await session.commit()

        service = TranscriptProcessingService(session, config)
        try:
            chunk_count = await service.run(episode_uuid)
        except Exception as exc:
            await session.rollback()

            is_final_attempt = ctx.get("job_try", 1) >= MAX_TRIES
            retryable = getattr(exc, "retryable", True)

            if retryable and not is_final_attempt:
                logger.warning(
                    "Processing attempt %s failed for episode %s (will retry): %s",
                    ctx.get("job_try"),
                    episode_id,
                    _safe_message(exc, generic=_GENERIC_PROCESSING_FAILURE_MESSAGE),
                )
                raise  # arq retries the task

            logger.error(
                "Processing failed permanently for episode %s after %s attempt(s): %s",
                episode_id,
                ctx.get("job_try"),
                _safe_message(exc, generic=_GENERIC_PROCESSING_FAILURE_MESSAGE),
            )
            await _mark_failed(
                session,
                episode_uuid,
                job_uuid,
                _safe_message(exc, generic=_GENERIC_PROCESSING_FAILURE_MESSAGE),
            )
            return

        job = await jobs.get_by_id(job_uuid)
        if job is not None:
            jobs.mark_completed(job)
            await session.commit()

        logger.info("Generated %s chunks for episode %s", chunk_count, episode_id)


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


def _safe_message(exc: Exception, generic: str = _GENERIC_FAILURE_MESSAGE) -> str:
    """Only ever surfaces our own exception `.message` text (which we
    control and never populate with a provider's raw response or a
    credential) -- anything else becomes a generic string, never str(exc).
    """
    message = getattr(exc, "message", None)
    return message if isinstance(message, str) else generic
