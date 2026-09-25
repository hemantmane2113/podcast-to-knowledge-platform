"""arq task(s). The worker process (`arq app.worker.settings.WorkerSettings`)
is the only thing that calls IngestionService/TranscriptProcessingService —
API routes only ever create an Episode + ProcessingJob and enqueue
ingest_episode_transcript (locked async-ingestion decision); chunking is
chained automatically after a successful ingestion (see bottom of
ingest_episode_transcript) rather than needing its own API trigger, since
Phase 3A has exactly one thing to do after a transcript exists.
"""

import asyncio
import logging
import uuid

from arq import Retry

from app.ai.graph import run_article_pipeline
from app.ai.state import ArticlePipelineState, PipelineDeps
from app.config import get_settings
from app.core.db import get_sessionmaker
from app.core.exceptions import ChunksNotFoundError, LLMProviderAuthError, TranscriptNotFoundError
from app.models.episode import ProcessingStatus
from app.models.processing_job import JobType
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.episode_repository import EpisodeRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.repositories.transcript_repository import TranscriptRepository
from app.services.chunking_service import ChunkingConfig
from app.services.ingestion_service import IngestionService
from app.services.transcript_processing_service import TranscriptProcessingService

logger = logging.getLogger(__name__)

MAX_TRIES = 3

# How long arq waits before re-running a job after a retryable failure.
# Verified directly against the installed arq version (0.28.0) with a real
# arq.worker.Worker against a real Redis (not assumed from arq's docs or
# our own comments): raising `arq.Retry(defer=N)` is the ONLY exception
# type arq's Worker.run_job treats as "please re-run this exact job_id" --
# a bare re-raise of an ordinary exception (which is what this module used
# to do) is indistinguishable, to arq, from any other unhandled exception:
# it's treated as a terminal failure (jobs_failed += 1) and the job_id's
# Redis bookkeeping is deleted outright, so nothing ever invokes it again.
# `job_try` (ctx["job_try"]) only increments across genuine Retry-triggered
# re-invocations -- it does NOT increment merely because a task raised.
RETRY_DEFER_SECONDS = 30

_GENERIC_FAILURE_MESSAGE = "Transcript ingestion failed. See server logs for details."
_GENERIC_PROCESSING_FAILURE_MESSAGE = "Transcript processing failed. See server logs for details."
_GENERIC_ARTICLE_FAILURE_MESSAGE = "Article generation failed. See server logs for details."


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
        except asyncio.CancelledError:
            # See generate_article's identical handler below for the full
            # rationale -- arq's own job_timeout cancellation (and a
            # worker-shutdown cancellation arq is about to retry) both
            # deliver CancelledError to this task the same way; this task
            # has no way to distinguish them, so it always marks this
            # attempt FAILED. If arq does retry it, the next invocation's
            # mark_running() (via IngestionService/its job repository)
            # simply overwrites this with RUNNING again.
            await session.rollback()
            logger.error(
                "Ingestion cancelled (job timeout) for episode %s (job try %s)",
                episode_id,
                ctx.get("job_try"),
            )
            await _mark_failed(session, episode_uuid, job_uuid, _GENERIC_FAILURE_MESSAGE)
            raise
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
                raise Retry(defer=RETRY_DEFER_SECONDS) from exc

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
        except asyncio.CancelledError:
            # See generate_article's identical handler below for the full
            # rationale. mark_running() already ran above, so this task
            # (unlike ingestion) doesn't depend on the service to have set
            # RUNNING before this can meaningfully transition to FAILED.
            await session.rollback()
            logger.error(
                "Processing cancelled (job timeout) for episode %s (job try %s)",
                episode_id,
                ctx.get("job_try"),
            )
            await _mark_failed(
                session, episode_uuid, job_uuid, _GENERIC_PROCESSING_FAILURE_MESSAGE
            )
            raise
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
                raise Retry(defer=RETRY_DEFER_SECONDS) from exc

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


async def generate_article(ctx: dict, episode_id: str, job_id: str) -> None:
    """arq entrypoint for the Phase C-H AI pipeline (app/ai/graph.py).

    Deliberately NOT auto-chained after process_transcript (unlike
    ingestion -> processing): this makes paid LLM calls, so it's only
    ever enqueued by an explicit POST /episodes/{id}/generate-article
    (see app/api/v1/articles.py).

    Unlike the transcript provider (constructed once per worker process
    in app/worker/settings.py::startup), the LLM provider is constructed
    here, lazily, on each invocation -- this is the only job type that
    needs one, so worker startup and ingestion/chunking never construct,
    import, or depend on it. `ctx["llm_provider"]` is honored first when
    present, purely as a test-injection seam (see tests/fakes.py's
    FakeLLMProvider) -- the real worker ctx never sets it.
    """
    episode_uuid = uuid.UUID(episode_id)
    job_uuid = uuid.UUID(job_id)

    settings = get_settings()
    session_factory = get_sessionmaker()

    async with session_factory() as session:
        if "llm_provider" in ctx:
            llm_provider = ctx["llm_provider"]
        else:
            # Local import: only this code path needs the groq/openai
            # SDKs importable. A module-level import here (as it used to
            # be in app/worker/settings.py) would require them just to
            # start the worker process or run an ingestion/chunking job,
            # which is the exact incident this task fixes.
            from app.providers.llm.factory import get_llm_provider

            try:
                llm_provider = get_llm_provider(settings)
            except (LLMProviderAuthError, ValueError) as exc:
                logger.warning(
                    "LLM provider (%s) is not configured: %s", settings.llm_provider, exc
                )
                llm_provider = None

        if llm_provider is None:
            # Same Episode.status Decoupling rationale as the CancelledError/
            # Exception handlers below -- this failure happens before any
            # pipeline state exists, but it's still a generate_article
            # failure path, so it must never overwrite a currently-PUBLISHED
            # episode's publication state either.
            await _mark_failed(
                session,
                episode_uuid,
                job_uuid,
                "LLM provider is not configured (check LLM_PROVIDER and its API key).",
                update_episode_status=False,
            )
            return

        jobs = ProcessingJobRepository(session)
        job = await jobs.get_by_id(job_uuid)
        if job is not None:
            jobs.mark_running(job)
            await session.commit()

        try:
            transcripts = TranscriptRepository(session)
            transcript = await transcripts.get_by_episode_id(episode_uuid)
            if transcript is None:
                raise TranscriptNotFoundError(f"Episode {episode_uuid} has no transcript")

            chunks = await ChunkRepository(session).get_by_transcript_id(transcript.id)
            if not chunks:
                raise ChunksNotFoundError(
                    f"Episode {episode_uuid} has no chunks yet -- run transcript processing first"
                )

            deps = PipelineDeps(session=session, llm_provider=llm_provider, settings=settings)
            initial_state: ArticlePipelineState = {
                "episode_id": episode_uuid,
                "transcript_id": transcript.id,
                "chunks": chunks,
                "transcript_word_count": sum(len(c.text.split()) for c in chunks),
                "revision_count": 0,
                "max_revision_attempts": settings.max_revision_attempts,
                # Live/Draft Article Workflow: every run reached through
                # this task (generate-article or regenerate-article) always
                # targets the episode's DRAFT plan/article -- the live one
                # is only ever replaced by an explicit
                # ArticleService.promote_draft call, never by generation
                # itself.
                "is_draft": True,
            }
            final_state = await run_article_pipeline(deps, initial_state)
        except asyncio.CancelledError:
            # arq's own job_timeout (WorkerSettings, default 300s) cancels
            # a job that runs too long by injecting CancelledError at the
            # current await point -- a BaseException, not an Exception, so
            # it was never caught by `except Exception` below. That left
            # ProcessingJob/Episode stuck at RUNNING/ANALYZING forever even
            # when the worker process itself was fine and had already
            # moved on to other jobs: arq's cancellation only updates its
            # own Redis-side job bookkeeping, never our processing_jobs
            # table, since our own except block simply never ran. Found
            # via a real stuck job with no code path left to recover it
            # (a later POST just kept returning the same permanently
            # "active" job -- see ArticleService.request_generation).
            #
            # Always mark this attempt failed here, unconditionally: this
            # task cannot distinguish, from inside a CancelledError handler,
            # whether arq is about to genuinely retry this job_id (a
            # worker-shutdown-triggered cancellation, which arq's Worker
            # treats as retry-eligible) or whether this is terminal (arq's
            # own job_timeout expiry, which is NOT retried -- verified
            # directly against the installed arq version: a timeout surfaces
            # to arq's own run_job as TimeoutError, not CancelledError, and
            # falls straight to its terminal-failure branch). If arq does
            # retry this job_id, the next invocation's mark_running() above
            # immediately overwrites this with RUNNING again (harmless); if
            # this was actually terminal, the job's FAILED status is correct
            # instead of a permanent, unrecoverable RUNNING (update_episode_status
            # =False below: Episode.status is no longer part of what could
            # get stuck here at all -- see _mark_failed's docstring). Never
            # swallow the
            # cancellation itself -- re-raise so arq observes it as it
            # expects.
            await session.rollback()
            logger.error(
                "Article generation cancelled (job timeout) for episode %s (job try %s)",
                episode_id,
                ctx.get("job_try"),
            )
            await _mark_failed(
                session, episode_uuid, job_uuid, _GENERIC_ARTICLE_FAILURE_MESSAGE, update_episode_status=False
            )
            raise
        except Exception as exc:
            await session.rollback()

            is_final_attempt = ctx.get("job_try", 1) >= MAX_TRIES
            retryable = getattr(exc, "retryable", True)

            if retryable and not is_final_attempt:
                logger.warning(
                    "Article generation attempt %s failed for episode %s (will retry): %s",
                    ctx.get("job_try"),
                    episode_id,
                    _safe_message(exc, generic=_GENERIC_ARTICLE_FAILURE_MESSAGE),
                )
                raise Retry(defer=RETRY_DEFER_SECONDS) from exc

            logger.error(
                "Article generation failed permanently for episode %s after %s attempt(s): %s",
                episode_id,
                ctx.get("job_try"),
                _safe_message(exc, generic=_GENERIC_ARTICLE_FAILURE_MESSAGE),
            )
            await _mark_failed(
                session,
                episode_uuid,
                job_uuid,
                _safe_message(exc, generic=_GENERIC_ARTICLE_FAILURE_MESSAGE),
                update_episode_status=False,
            )
            return

        # The draft reaching a terminal state -- regardless of whether
        # validation passed -- is recorded on ProcessingJob.status
        # (COMPLETED below), never on Episode.status (Episode.status
        # Decoupling / Live-Draft Article Workflow): a failing draft still
        # needs to reach a reviewable state (Phase I), not be silently
        # dropped, but it must never overwrite a currently-PUBLISHED
        # episode's publication state. The validation result itself
        # (persisted by the validation node) tells a reviewer which checks
        # failed; GET /episodes/{id}/article?draft=true is how a reviewer
        # retrieves the draft (app/api/v1/articles.py).
        job = await jobs.get_by_id(job_uuid)
        if job is not None:
            jobs.mark_completed(job)
        await session.commit()

        logger.info(
            "Generated article for episode %s (validation passed=%s, revisions=%s)",
            episode_id,
            final_state["validation_report"].passed,
            final_state.get("revision_count", 0),
        )


async def _mark_failed(
    session, episode_id: uuid.UUID, job_id: uuid.UUID, message: str, *, update_episode_status: bool = True
) -> None:
    """`update_episode_status` defaults to True, preserving this
    function's original behavior for ingest_episode_transcript/
    process_transcript, both of which run before any article exists and
    have no publication state to protect. generate_article's own failure
    paths (and app/worker/settings.py::_recover_stale_article_generation_jobs,
    which routes stale ARTICLE_GENERATION jobs through this same function)
    pass update_episode_status=False -- a draft-generation failure is
    recorded on ProcessingJob.status/error_message only, never on
    Episode.status, so it can never overwrite a currently-PUBLISHED
    episode's publication state (Episode.status Decoupling / Live-Draft
    Article Workflow)."""
    jobs = ProcessingJobRepository(session)
    job = await jobs.get_by_id(job_id)

    if update_episode_status:
        episodes = EpisodeRepository(session)
        episode = await episodes.get_by_id(episode_id)
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
