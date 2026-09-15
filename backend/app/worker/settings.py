"""arq WorkerSettings — run with: arq app.worker.settings.WorkerSettings"""

import logging

from arq.connections import RedisSettings

from app.config import get_settings
from app.core.exceptions import TranscriptProviderAuthError
from app.providers.transcript.supadata import SupadataTranscriptProvider
from app.services.job_queue import JobQueue
from app.worker.tasks import MAX_TRIES, ingest_episode_transcript, process_transcript

logger = logging.getLogger(__name__)


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

    # ctx["redis"] is populated by arq itself before on_startup runs --
    # reuse that connection rather than opening a second pool just to
    # enqueue the follow-up transcript-processing job.
    ctx["job_queue"] = JobQueue(ctx["redis"])


async def shutdown(ctx: dict) -> None:
    provider = ctx.get("provider")
    if provider is not None:
        await provider.aclose()


class WorkerSettings:
    functions = [ingest_episode_transcript, process_transcript]
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_tries = MAX_TRIES
    on_startup = startup
    on_shutdown = shutdown
