"""Selects and constructs the configured LLMProvider. The only place in
the application that imports all three concrete providers (plus
FallbackLLMProvider) -- everything else (app/ai/, app/worker/) depends on
LLMProvider (base.py) and get_llm_provider() only, never on a concrete
provider class directly.
"""

from app.config.settings import Settings
from app.providers.llm.base import LLMProvider
from app.providers.llm.fallback_provider import FallbackLLMProvider
from app.providers.llm.groq_provider import GroqProvider
from app.providers.llm.openai_provider import OpenAIProvider
from app.providers.llm.opensource_provider import OpenSourceProvider


def get_llm_provider(settings: Settings) -> LLMProvider:
    """Raises LLMProviderAuthError (via the concrete provider's own
    constructor) if the selected provider's required config is missing --
    the same "fail clearly at the point of use" pattern as
    SupadataTranscriptProvider, not at import time. The same applies to
    the fallback provider below: if OPENAI_FALLBACK_ENABLED is true but
    OPENAI_API_KEY isn't set, constructing it raises immediately here too
    -- there's no separate "fallback misconfigured" state to degrade into.
    """
    primary = _construct(settings.llm_provider, settings)

    # Never wrap OpenAI behind an OpenAI fallback -- that would just retry
    # the same provider against itself, not a real fallback.
    if settings.llm_provider != "openai" and settings.openai_fallback_enabled:
        fallback = OpenAIProvider(api_key=settings.openai_api_key, model=settings.openai_fallback_model)
        return FallbackLLMProvider(primary=primary, fallback=fallback)

    return primary


def _construct(provider: str, settings: Settings) -> LLMProvider:
    if provider == "groq":
        return GroqProvider(api_key=settings.groq_api_key, model=settings.llm_model)
    if provider == "openai":
        return OpenAIProvider(api_key=settings.openai_api_key, model=settings.llm_model)
    if provider == "opensource":
        return OpenSourceProvider(
            base_url=settings.opensource_base_url,
            model=settings.llm_model,
            api_key=settings.opensource_api_key,
        )
    raise ValueError(
        f"LLM_PROVIDER={provider!r} is not one of 'groq', 'openai', 'opensource'"
    )
