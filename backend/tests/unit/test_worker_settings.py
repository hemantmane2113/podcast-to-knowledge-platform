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

import app.worker.settings as worker_settings
import app.worker.tasks as worker_tasks
from app.worker.settings import JOB_TIMEOUT_SECONDS, WorkerSettings, startup


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
