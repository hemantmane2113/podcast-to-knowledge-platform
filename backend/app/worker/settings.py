"""arq WorkerSettings — run with: arq app.worker.settings.WorkerSettings"""

import logging

from arq.connections import RedisSettings

from app.config import get_settings
from app.core.exceptions import TranscriptProviderAuthError
from app.providers.transcript.supadata import SupadataTranscriptProvider
from app.services.job_queue import JobQueue
from app.worker.tasks import MAX_TRIES, generate_article, ingest_episode_transcript, process_transcript

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
