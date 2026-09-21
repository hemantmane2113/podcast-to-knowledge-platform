"""OpenAI LLMProvider. Only this module (and opensource_provider.py, which
reuses the same SDK against a different base_url) imports the `openai` SDK.
"""

from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import AsyncOpenAI
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import RateLimitError as OpenAIRateLimitError

from app.core.exceptions import LLMProviderAuthError
from app.providers.llm._chat_completions import ChatCompletionsProvider, map_sdk_exception


class OpenAIProvider(ChatCompletionsProvider):
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMProviderAuthError("OPENAI_API_KEY is not configured")
        if not model:
            raise LLMProviderAuthError("LLM_MODEL is not configured")
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key)

    def _map_exception(self, exc: Exception) -> Exception:
        return map_sdk_exception(
            exc,
            auth_error_type=OpenAIAuthenticationError,
            rate_limit_error_type=OpenAIRateLimitError,
            connection_error_type=OpenAIAPIConnectionError,
        )
