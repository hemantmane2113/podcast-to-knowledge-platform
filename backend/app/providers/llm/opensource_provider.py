"""Open-source / self-hosted LLMProvider -- any server exposing an
OpenAI-compatible `/v1/chat/completions` endpoint (vLLM, Ollama, TGI, ...).
Reuses the `openai` SDK pointed at a custom `base_url` rather than adding a
fourth SDK dependency for a call shape every one of these servers already
implements -- this is the de facto standard for self-hosted inference.
"""

from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import AsyncOpenAI
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import RateLimitError as OpenAIRateLimitError

from app.core.exceptions import LLMProviderAuthError
from app.providers.llm._chat_completions import ChatCompletionsProvider, map_sdk_exception

# Most self-hosted servers don't enforce a real key, but the OpenAI SDK
# requires a non-empty string to construct the client.
_PLACEHOLDER_KEY = "not-required"


class OpenSourceProvider(ChatCompletionsProvider):
    def __init__(self, base_url: str, model: str, api_key: str = ""):
        if not base_url:
            raise LLMProviderAuthError("OPENSOURCE_BASE_URL is not configured")
        if not model:
            raise LLMProviderAuthError("LLM_MODEL is not configured")
        self._model = model
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key or _PLACEHOLDER_KEY)

    def _map_exception(self, exc: Exception) -> Exception:
        return map_sdk_exception(
            exc,
            auth_error_type=OpenAIAuthenticationError,
            rate_limit_error_type=OpenAIRateLimitError,
            connection_error_type=OpenAIAPIConnectionError,
        )
