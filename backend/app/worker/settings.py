"""arq WorkerSettings — run with: arq app.worker.settings.WorkerSettings"""

import logging
from datetime import UTC, datetime, timedelta

from arq.connections import RedisSettings

from app.config import get_settings
from app.core.db import get_sessionmaker
from app.core.exceptions import TranscriptProviderAuthError
from app.models.processing_job import JobType
from app.providers.transcript.supadata import SupadataTranscriptProvider
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.services.job_queue import JobQueue
from app.worker.tasks import (
    MAX_TRIES,
    RETRY_DEFER_SECONDS,
    _mark_failed,
    generate_article,
    ingest_episode_transcript,
    process_transcript,
)

logger = logging.getLogger(__name__)

# arq's own default (300s) is too short for generate_article: the V1
# pipeline is a strictly sequential chain (topic analysis -- possibly
# several batches -- -> planning -> one real LLM call per planned
# section -> validation -> up to max_revision_attempts more rounds), and
# a normal, fully successful run against a real, long podcast transcript
# can legitimately take several minutes. Raised to 20 minutes so a
# correctly-running job isn't killed partway through; still bounded (not
# removed) -- a job that runs long past this is still cancelled and,
# since commit 73b6191, still cleanly marked FAILED rather than left
# stuck (app/worker/tasks.py::generate_article's
# `except asyncio.CancelledError` handling, unchanged here).
JOB_TIMEOUT_SECONDS = 1200

# A worker *process* being killed/restarted (container redeploy, OOM kill,
# `docker restart`) leaves a RUNNING/PENDING processing_jobs row exactly as
# it last committed -- unlike arq's own in-loop job_timeout cancellation
# (JOB_TIMEOUT_SECONDS above), no exception is ever raised in that case, so
# app/worker/tasks.py::generate_article's `except asyncio.CancelledError`
# handling (commit 73b6191) never runs, and nothing else in arq itself
# recovers it (its Redis in_progress_key is a concurrency lock, not a
# recovery mechanism -- it just silently expires). Found via a real
# production row stuck at RUNNING/ANALYZING after a worker restart, with
# updated_at showing it was last touched well past JOB_TIMEOUT_SECONDS ago.
#
# This grace period is added on top of the maximum legitimate multi-attempt
# sequence (MAX_LEGITIMATE_RETRY_SEQUENCE_SECONDS below) before a RUNNING
# job is considered stale: any live worker still legitimately executing --
# or waiting between retries of -- a job would fall within that window, so
# a RUNNING row older than the full window cannot possibly still be in
# progress. This is a bound on elapsed time, not a blind "mark everything
# RUNNING as failed" sweep. The grace itself only absorbs clock skew /
# startup-order noise between the row's started_at and this check.
STALE_JOB_GRACE_SECONDS = 60

# The longest a genuinely-still-in-progress ARTICLE_GENERATION job can
# possibly take, end to end, before stale recovery may safely assume it's
# abandoned rather than mid-retry. `started_at` is deliberately preserved
# across retry attempts (ProcessingJobRepository.mark_running's "preserve
# the original start time" behavior), so a job legitimately on its 2nd or
# 3rd attempt (see app/worker/tasks.py's Retry(defer=RETRY_DEFER_SECONDS))
# can have a `started_at` far older than a single JOB_TIMEOUT_SECONDS while
# still being completely legitimate. This computes the true worst case
# instead of hardcoding a number that would silently go stale (pun
# intended) if MAX_TRIES, JOB_TIMEOUT_SECONDS, or RETRY_DEFER_SECONDS ever
# change: MAX_TRIES attempts, each individually bounded by
# JOB_TIMEOUT_SECONDS, with a RETRY_DEFER_SECONDS wait between each pair of
# attempts (MAX_TRIES - 1 gaps -- no wait follows the final attempt, since
# app/worker/tasks.py never raises Retry on it).
MAX_LEGITIMATE_RETRY_SEQUENCE_SECONDS = (MAX_TRIES * JOB_TIMEOUT_SECONDS) + (
    (MAX_TRIES - 1) * RETRY_DEFER_SECONDS
)

# The actual cutoff used by _recover_stale_article_generation_jobs below --
# a RUNNING row older than this cannot possibly still be a legitimate
# in-progress attempt sequence. Also used, conservatively, for the PENDING
# check: a job that was never even picked up doesn't need multi-attempt
# accounting to be recognized as stale, but reusing the same (wider) cutoff
# for both is simpler than maintaining two separate windows and only makes
# PENDING-staleness detection slightly more patient, never wrong.
STALE_JOB_CUTOFF_SECONDS = MAX_LEGITIMATE_RETRY_SEQUENCE_SECONDS + STALE_JOB_GRACE_SECONDS


async def _recover_stale_article_generation_jobs(ctx: dict) -> None:
    """Runs once per worker process, at startup. Finds ARTICLE_GENERATION
    jobs that cannot possibly still be legitimately in progress -- RUNNING
    since before STALE_JOB_CUTOFF_SECONDS ago (the full MAX_TRIES-attempt
    sequence, including retry waits, plus grace), or PENDING for that long
    without ever having been picked up (e.g. a worker died between enqueue
    and its first mark_running) -- and marks them FAILED via the same
    `_mark_failed` helper every other failure path in app/worker/tasks.py
    uses, so the episode is unstuck from ANALYZING and a user can retry
    generation instead of the job being wedged forever.
    """
    cutoff = datetime.now(UTC) - timedelta(seconds=STALE_JOB_CUTOFF_SECONDS)
    session_factory = get_sessionmaker()

    async with session_factory() as session:
        jobs = ProcessingJobRepository(session)
        stale_jobs = await jobs.get_stale_active_jobs(
            JobType.ARTICLE_GENERATION, running_cutoff=cutoff, pending_cutoff=cutoff
        )
        for job in stale_jobs:
            logger.error(
                "Recovering stale ARTICLE_GENERATION job %s for episode %s (was %s, "
                "started_at=%s, created_at=%s) -- worker process was likely killed or "
                "restarted before the job reached a terminal status.",
                job.id,
                job.episode_id,
                job.status,
                job.started_at,
                job.created_at,
            )
            await _mark_failed(
                session,
                job.episode_id,
                job.id,
                "Article generation was interrupted by a worker restart. Please retry.",
            )


async def startup(ctx: dict) -> None:
    settings = get_settings()
    try:
        ctx["provider"] = SupadataTranscriptProvider(api_key=settings.supadata_api_key)
    except TranscriptProviderAuthError:
        # Missing key: don't crash-loop the whole worker process over a
        # config problem -- let it start, and fail each job clearly and
        # immediately instead (see app/worker/tasks.py).
        logger.warning(
            "SUPADATA_API_KEY is not configured; ingestion jobs will fail until it is set."
        )
        ctx["provider"] = None

    # Deliberately NOT constructing an LLM provider here, even with the
    # same graceful-degradation pattern as the transcript provider above:
    # ingestion and chunking (the jobs every worker process handles most
    # of the time) never use one, so this process shouldn't construct --
    # or even import the groq/openai SDKs to construct -- one it may never
    # need. generate_article (app/worker/tasks.py) is the only job type
    # that needs an LLM provider, and it builds one itself, lazily, via
    # app.providers.llm.factory.get_llm_provider, exactly when that job
    # runs (same "fail clearly at the point of use" outcome as before,
    # just moved to where the use actually is).

    # ctx["redis"] is populated by arq itself before on_startup runs --
    # reuse that connection rather than opening a second pool just to
    # enqueue the follow-up transcript-processing job.
    ctx["job_queue"] = JobQueue(ctx["redis"])

    # See STALE_JOB_GRACE_SECONDS above: recovers ARTICLE_GENERATION jobs
    # left stuck RUNNING/PENDING by a worker process that was killed or
    # restarted, not just arq's own in-loop timeout cancellation.
    await _recover_stale_article_generation_jobs(ctx)


async def shutdown(ctx: dict) -> None:
    provider = ctx.get("provider")
    if provider is not None:
        await provider.aclose()


class WorkerSettings:
    functions = [ingest_episode_transcript, process_transcript, generate_article]
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_tries = MAX_TRIES
    job_timeout = JOB_TIMEOUT_SECONDS
    on_startup = startup
    on_shutdown = shutdown
