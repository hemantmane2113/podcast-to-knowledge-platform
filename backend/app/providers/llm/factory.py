"""Selects and constructs the configured LLMProvider. The only place in
the application that imports all three concrete providers -- everything
else (app/ai/, app/worker/) depends on LLMProvider (base.py) and
get_llm_provider() only, never on a concrete provider class directly.
"""

from app.config.settings import Settings
from app.providers.llm.base import LLMProvider
from app.providers.llm.groq_provider import GroqProvider
from app.providers.llm.openai_provider import OpenAIProvider
from app.providers.llm.opensource_provider import OpenSourceProvider


def get_llm_provider(settings: Settings) -> LLMProvider:
    """Raises LLMProviderAuthError (via the concrete provider's own
    constructor) if the selected provider's required config is missing --
    the same "fail clearly at the point of use" pattern as
    SupadataTranscriptProvider, not at import time.
    """
    provider = settings.llm_provider
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
