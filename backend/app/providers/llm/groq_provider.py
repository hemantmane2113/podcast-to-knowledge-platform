"""Groq LLMProvider -- the primary inference provider for V1 (see
ARCHITECTURE.md's provider-selection notes). Only this module imports the
`groq` SDK.
"""

from groq import AsyncGroq
from groq import APIConnectionError as GroqAPIConnectionError
from groq import AuthenticationError as GroqAuthenticationError
from groq import RateLimitError as GroqRateLimitError

from app.core.exceptions import LLMProviderAuthError
from app.providers.llm._chat_completions import (
    DEFAULT_CLIENT_MAX_RETRIES,
    DEFAULT_CLIENT_TIMEOUT_SECONDS,
    ChatCompletionsProvider,
    map_sdk_exception,
)


class GroqProvider(ChatCompletionsProvider):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMProviderAuthError("GROQ_API_KEY is not configured")
        if not model:
            raise LLMProviderAuthError("LLM_MODEL is not configured")
        self._model = model
        self._client = AsyncGroq(
            api_key=api_key,
            timeout=DEFAULT_CLIENT_TIMEOUT_SECONDS,
            max_retries=DEFAULT_CLIENT_MAX_RETRIES,
        )

    def _map_exception(self, exc: Exception) -> Exception:
        return map_sdk_exception(
            exc,
            auth_error_type=GroqAuthenticationError,
            rate_limit_error_type=GroqRateLimitError,
            connection_error_type=GroqAPIConnectionError,
        )
