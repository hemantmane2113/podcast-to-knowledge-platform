"""Regression tests for the lazy-LLM-provider fix: app/worker/settings.py's
startup() used to construct the LLM provider eagerly (via
app.providers.llm.factory.get_llm_provider) for every worker process, even
though only generate_article (app/worker/tasks.py) ever uses one. That
required the groq/openai SDKs to be importable just to start the worker or
run an ingestion/chunking job -- exactly the class of incident this fix
closes (see app/worker/tasks.py::generate_article's docstring).

These are structural/behavioral checks that startup() itself never
constructs -- or even imports -- an LLM provider, independent of whichever
LLM_PROVIDER/GROQ_API_KEY/LLM_MODEL happen to be set in the environment
running these tests.
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

import app.worker.settings as worker_settings
import app.worker.tasks as worker_tasks
from app.models.episode import ProcessingStatus
from app.models.processing_job import JobStatus, JobType
from app.repositories.episode_repository import EpisodeRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.worker.settings import (
    JOB_TIMEOUT_SECONDS,
    MAX_LEGITIMATE_RETRY_SEQUENCE_SECONDS,
    STALE_JOB_CUTOFF_SECONDS,
    STALE_JOB_GRACE_SECONDS,
    WorkerSettings,
    _recover_stale_article_generation_jobs,
    startup,
)
from app.worker.tasks import MAX_TRIES, RETRY_DEFER_SECONDS

VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
VIDEO_ID = "dQw4w9WgXcQ"

# Genuinely stale: past the full cutoff (the widest legitimate MAX_TRIES-
# attempt sequence, including retry waits, plus grace).
_STALE_MARGIN = timedelta(seconds=STALE_JOB_CUTOFF_SECONDS + 30)
_FRESH_MARGIN = timedelta(seconds=10)

# Older than a single JOB_TIMEOUT_SECONDS + STALE_JOB_GRACE_SECONDS (the OLD,
# pre-widening cutoff -- would have been wrongly recovered under that
# formula) but still comfortably inside the real, multi-attempt-aware
# cutoff: a job legitimately in the middle of its 2nd or 3rd retry attempt.
_LEGITIMATE_MULTI_ATTEMPT_MARGIN = timedelta(seconds=STALE_JOB_CUTOFF_SECONDS - 60)

# Boundary margins, offset a few seconds either side of the exact cutoff to
# stay robust against the small real-world gap between when a test sets
# started_at and when _recover_stale_article_generation_jobs computes
# datetime.now(UTC) -- both are well clear of that gap while still each
# being on a single, unambiguous side of the strict `<` comparison in
# ProcessingJobRepository.get_stale_active_jobs.
_JUST_INSIDE_CUTOFF_MARGIN = timedelta(seconds=STALE_JOB_CUTOFF_SECONDS - 5)
_JUST_OUTSIDE_CUTOFF_MARGIN = timedelta(seconds=STALE_JOB_CUTOFF_SECONDS + 5)


async def test_startup_does_not_populate_llm_provider_in_ctx() -> None:
    ctx: dict = {"redis": object()}

    await startup(ctx)

    assert "llm_provider" not in ctx
    # The transcript provider is still constructed (or gracefully set to
    # None) exactly as before -- only the LLM provider's construction was
    # removed from startup().
    assert "provider" in ctx


def test_worker_settings_module_does_not_import_get_llm_provider() -> None:
    """Guards against regressing back to a module-level `from
    app.providers.llm.factory import get_llm_provider` in
    app/worker/settings.py -- that import alone, regardless of whether it
    is ever called, requires the groq/openai SDKs to be installed just to
    start the worker process."""
    assert not hasattr(worker_settings, "get_llm_provider")


def test_worker_tasks_module_does_not_import_get_llm_provider_at_module_level() -> None:
    """get_llm_provider is imported lazily inside generate_article() itself
    (a local import), not at module level -- so merely importing
    app.worker.tasks (which app.worker.settings does unconditionally, to
    register its arq functions) never requires groq/openai to be
    installed. Only actually running generate_article does."""
    assert not hasattr(worker_tasks, "get_llm_provider")


def test_worker_settings_explicitly_configures_a_1200_second_job_timeout() -> None:
    """arq's own default (300s) is too short for generate_article: the V1
    pipeline is a strictly sequential chain of real LLM calls (topic
    analysis, possibly several batches -> planning -> one call per
    planned section -> validation -> up to max_revision_attempts more
    rounds), and a normal, fully successful run against a real, long
    podcast transcript can legitimately take several minutes -- a real
    run was killed mid-pipeline by the 300s default. Raised explicitly,
    in version control, rather than left at arq's implicit default."""
    assert JOB_TIMEOUT_SECONDS == 1200
    assert WorkerSettings.job_timeout == 1200


async def _seed_episode(session: AsyncSession):
    episodes = EpisodeRepository(session)
    episode = episodes.create(youtube_video_id=VIDEO_ID, youtube_url=VIDEO_URL)
    episodes.set_status(episode, ProcessingStatus.ANALYZING)
    await session.commit()
    return episode


async def test_recover_marks_stale_running_article_generation_job_failed(
    db_session: AsyncSession,
) -> None:
    """Reproduces the real production incident this fix targets: a worker
    process killed/restarted mid-job leaves the row exactly as it last
    committed (RUNNING, no exception ever raised -- unlike arq's own
    in-loop job_timeout cancellation, which app/worker/tasks.py's
    `except asyncio.CancelledError` already handles). A RUNNING row whose
    started_at is older than STALE_JOB_CUTOFF_SECONDS (the full legitimate
    multi-attempt sequence plus grace) cannot possibly still be
    legitimately in progress."""
    episode = await _seed_episode(db_session)
    jobs = ProcessingJobRepository(db_session)
    job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(UTC) - _STALE_MARGIN
    await db_session.commit()
    episode_id, job_id = episode.id, job.id  # captured before expire_all below

    await _recover_stale_article_generation_jobs({})

    db_session.expire_all()  # written through a separate session -- see below
    refreshed_episode = await EpisodeRepository(db_session).get_by_id(episode_id)
    refreshed_job = await ProcessingJobRepository(db_session).get_by_id(job_id)
    assert refreshed_episode.status == ProcessingStatus.FAILED
    assert refreshed_job.status == JobStatus.FAILED
    assert refreshed_job.error_message


async def test_recover_leaves_recent_running_article_generation_job_untouched(
    db_session: AsyncSession,
) -> None:
    """Proves this is NOT a blind "mark every RUNNING job failed" sweep
    (an explicit requirement): a job that started well within
    JOB_TIMEOUT_SECONDS could still be a live, legitimately in-progress
    job on a healthy worker, and must be left alone."""
    episode = await _seed_episode(db_session)
    jobs = ProcessingJobRepository(db_session)
    job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(UTC) - _FRESH_MARGIN
    await db_session.commit()
    episode_id, job_id = episode.id, job.id  # captured before expire_all below

    await _recover_stale_article_generation_jobs({})

    db_session.expire_all()  # written through a separate session -- see above
    refreshed_episode = await EpisodeRepository(db_session).get_by_id(episode_id)
    refreshed_job = await ProcessingJobRepository(db_session).get_by_id(job_id)
    assert refreshed_episode.status == ProcessingStatus.ANALYZING
    assert refreshed_job.status == JobStatus.RUNNING


def test_stale_cutoff_accounts_for_the_full_legitimate_retry_sequence() -> None:
    """Regression test for the D.4a decision: the cutoff must be computed
    from MAX_TRIES/JOB_TIMEOUT_SECONDS/RETRY_DEFER_SECONDS, not a hardcoded
    number, so it stays correct if any of those change. MAX_TRIES attempts,
    each individually bounded by JOB_TIMEOUT_SECONDS, with a
    RETRY_DEFER_SECONDS wait between each pair of attempts (MAX_TRIES - 1
    gaps -- no wait follows the final, non-retried attempt)."""
    expected_sequence = (MAX_TRIES * JOB_TIMEOUT_SECONDS) + ((MAX_TRIES - 1) * RETRY_DEFER_SECONDS)
    assert MAX_LEGITIMATE_RETRY_SEQUENCE_SECONDS == expected_sequence
    assert STALE_JOB_CUTOFF_SECONDS == expected_sequence + STALE_JOB_GRACE_SECONDS
    # With the current real values (MAX_TRIES=3, JOB_TIMEOUT_SECONDS=1200,
    # RETRY_DEFER_SECONDS=30, STALE_JOB_GRACE_SECONDS=60): 3*1200 + 2*30 +
    # 60 = 3720s (62 minutes) -- much wider than the old single-attempt
    # cutoff of JOB_TIMEOUT_SECONDS + STALE_JOB_GRACE_SECONDS = 1260s.
    assert STALE_JOB_CUTOFF_SECONDS == 3720
    assert STALE_JOB_CUTOFF_SECONDS > JOB_TIMEOUT_SECONDS + STALE_JOB_GRACE_SECONDS


async def test_recover_leaves_job_within_legitimate_multi_attempt_window_untouched(
    db_session: AsyncSession,
) -> None:
    """The core regression this widening exists for: once app/worker/tasks.py
    genuinely retries via arq.Retry(defer=RETRY_DEFER_SECONDS), started_at
    is preserved across attempts (ProcessingJobRepository.mark_running), so
    a job legitimately in the middle of its 2nd or 3rd attempt can have a
    started_at older than a single JOB_TIMEOUT_SECONDS + STALE_JOB_GRACE_SECONDS
    while still being completely legitimate. Recovering it anyway (the OLD
    cutoff formula) would incorrectly kill a job arq is still working
    through and free the episode for a wrongly-duplicated new submission."""
    episode = await _seed_episode(db_session)
    jobs = ProcessingJobRepository(db_session)
    job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(UTC) - _LEGITIMATE_MULTI_ATTEMPT_MARGIN
    await db_session.commit()
    episode_id, job_id = episode.id, job.id  # captured before expire_all below

    # Sanity-check the premise: this margin is older than the OLD,
    # pre-widening cutoff formula would have tolerated.
    assert _LEGITIMATE_MULTI_ATTEMPT_MARGIN > timedelta(
        seconds=JOB_TIMEOUT_SECONDS + STALE_JOB_GRACE_SECONDS
    )

    await _recover_stale_article_generation_jobs({})

    db_session.expire_all()  # written through a separate session -- see above
    refreshed_episode = await EpisodeRepository(db_session).get_by_id(episode_id)
    refreshed_job = await ProcessingJobRepository(db_session).get_by_id(job_id)
    assert refreshed_episode.status == ProcessingStatus.ANALYZING
    assert refreshed_job.status == JobStatus.RUNNING


async def test_recover_leaves_job_just_inside_the_cutoff_untouched(
    db_session: AsyncSession,
) -> None:
    """Boundary condition: a RUNNING row just younger than
    STALE_JOB_CUTOFF_SECONDS must not be recovered."""
    episode = await _seed_episode(db_session)
    jobs = ProcessingJobRepository(db_session)
    job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(UTC) - _JUST_INSIDE_CUTOFF_MARGIN
    await db_session.commit()
    episode_id, job_id = episode.id, job.id  # captured before expire_all below

    await _recover_stale_article_generation_jobs({})

    db_session.expire_all()  # written through a separate session -- see above
    refreshed_episode = await EpisodeRepository(db_session).get_by_id(episode_id)
    refreshed_job = await ProcessingJobRepository(db_session).get_by_id(job_id)
    assert refreshed_episode.status == ProcessingStatus.ANALYZING
    assert refreshed_job.status == JobStatus.RUNNING


async def test_recover_marks_job_just_outside_the_cutoff_failed(
    db_session: AsyncSession,
) -> None:
    """Boundary condition: a RUNNING row just older than
    STALE_JOB_CUTOFF_SECONDS must be recovered."""
    episode = await _seed_episode(db_session)
    jobs = ProcessingJobRepository(db_session)
    job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(UTC) - _JUST_OUTSIDE_CUTOFF_MARGIN
    await db_session.commit()
    episode_id, job_id = episode.id, job.id  # captured before expire_all below

    await _recover_stale_article_generation_jobs({})

    db_session.expire_all()  # written through a separate session -- see above
    refreshed_episode = await EpisodeRepository(db_session).get_by_id(episode_id)
    refreshed_job = await ProcessingJobRepository(db_session).get_by_id(job_id)
    assert refreshed_episode.status == ProcessingStatus.FAILED
    assert refreshed_job.status == JobStatus.FAILED


async def test_recover_marks_stale_pending_article_generation_job_failed(
    db_session: AsyncSession,
) -> None:
    """A worker can also die between enqueue and its first mark_running
    (job_try/mark_running happens at the top of generate_article) -- the
    row is left PENDING forever in that case, never RUNNING. Same
    staleness argument applies to created_at instead of started_at."""
    episode = await _seed_episode(db_session)
    jobs = ProcessingJobRepository(db_session)
    job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    job.created_at = datetime.now(UTC) - _STALE_MARGIN
    await db_session.commit()
    episode_id, job_id = episode.id, job.id  # captured before expire_all below

    await _recover_stale_article_generation_jobs({})

    db_session.expire_all()  # written through a separate session -- see above
    refreshed_episode = await EpisodeRepository(db_session).get_by_id(episode_id)
    refreshed_job = await ProcessingJobRepository(db_session).get_by_id(job_id)
    assert refreshed_episode.status == ProcessingStatus.FAILED
    assert refreshed_job.status == JobStatus.FAILED


async def test_recover_leaves_completed_article_generation_job_untouched(
    db_session: AsyncSession,
) -> None:
    """A historical COMPLETED job with an old started_at must never be
    touched -- staleness only applies to still-active (RUNNING/PENDING)
    statuses."""
    episode = await _seed_episode(db_session)
    jobs = ProcessingJobRepository(db_session)
    job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    job.status = JobStatus.COMPLETED
    job.started_at = datetime.now(UTC) - _STALE_MARGIN
    job.completed_at = datetime.now(UTC) - _STALE_MARGIN
    await db_session.commit()
    job_id = job.id  # captured before expire_all below

    await _recover_stale_article_generation_jobs({})

    db_session.expire_all()  # written through a separate session -- see above
    refreshed_job = await ProcessingJobRepository(db_session).get_by_id(job_id)
    assert refreshed_job.status == JobStatus.COMPLETED


async def test_recover_ignores_stale_running_jobs_of_other_job_types(
    db_session: AsyncSession,
) -> None:
    """Scoped to ARTICLE_GENERATION only (the smallest safe V1 fix for the
    reported incident) -- a stale TRANSCRIPT_PROCESSING row is left for a
    future, separately-scoped fix rather than silently swept up here."""
    episode = await _seed_episode(db_session)
    jobs = ProcessingJobRepository(db_session)
    job = jobs.create(episode_id=episode.id, job_type=JobType.TRANSCRIPT_PROCESSING)
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(UTC) - _STALE_MARGIN
    await db_session.commit()
    job_id = job.id  # captured before expire_all below

    await _recover_stale_article_generation_jobs({})

    db_session.expire_all()  # written through a separate session -- see above
    refreshed_job = await ProcessingJobRepository(db_session).get_by_id(job_id)
    assert refreshed_job.status == JobStatus.RUNNING


async def test_startup_calls_stale_job_recovery(db_session: AsyncSession) -> None:
    """End-to-end through the real on_startup entrypoint arq calls, not
    just the helper directly -- proves the wiring in startup() is live."""
    episode = await _seed_episode(db_session)
    jobs = ProcessingJobRepository(db_session)
    job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(UTC) - _STALE_MARGIN
    await db_session.commit()
    job_id = job.id  # captured before expire_all below

    await startup({"redis": object()})

    db_session.expire_all()  # written through a separate session -- see above
    refreshed_job = await ProcessingJobRepository(db_session).get_by_id(job_id)
    assert refreshed_job.status == JobStatus.FAILED
