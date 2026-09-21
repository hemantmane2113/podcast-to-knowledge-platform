"""Shared implementation for any provider whose client exposes an
OpenAI-style `client.chat.completions.create(...)` method -- both the
`groq` and `openai` Python SDKs use this exact call shape (Groq's API is
itself OpenAI-compatible), so GroqProvider, OpenAIProvider, and
OpenSourceProvider (a bare AsyncOpenAI pointed at a custom base_url) all
subclass ChatCompletionsProvider below instead of duplicating this logic
three times. Each subclass still imports only its own SDK, for exception
mapping and client construction -- this module itself stays SDK-agnostic.
"""

import json
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from app.providers.llm.base import LLMMessage, LLMProvider, LLMTextResponse, LLMUsage

T = TypeVar("T", bound=BaseModel)


class ChatCompletionsClient(Protocol):
    """Structural type for the subset of the OpenAI/Groq client shape this
    module actually uses -- not the real SDK's type, just enough for
    static checking of the call below."""

    chat: Any


def _build_messages(messages: list[LLMMessage], system: str | None) -> list[dict[str, str]]:
    payload = [{"role": m.role, "content": m.content} for m in messages]
    if system is not None:
        payload = [{"role": "system", "content": system}, *payload]
    return payload


def _extract_text_and_usage(response: Any) -> tuple[str, LLMUsage | None]:
    text = response.choices[0].message.content or ""
    usage = None
    if getattr(response, "usage", None) is not None:
        usage = LLMUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
    return text, usage


def map_sdk_exception(
    exc: Exception,
    *,
    auth_error_type: type[Exception],
    rate_limit_error_type: type[Exception],
    connection_error_type: type[Exception],
) -> Exception:
    """Maps a `groq`/`openai` SDK exception to our LLMProviderError
    hierarchy. Both SDKs export an identically-shaped exception taxonomy
    (AuthenticationError, RateLimitError, APIConnectionError, ...) since
    Groq's client is generated the same way OpenAI's is -- each caller
    passes in *its own* imported exception classes so this stays correct
    without importing both SDKs here.
    """
    from app.core.exceptions import (
        LLMProviderAuthError,
        LLMProviderError,
        LLMProviderRateLimitError,
    )

    if isinstance(exc, auth_error_type):
        return LLMProviderAuthError(str(exc))
    if isinstance(exc, rate_limit_error_type):
        return LLMProviderRateLimitError(str(exc))
    if isinstance(exc, connection_error_type):
        return LLMProviderError(str(exc))  # transient, retryable by default
    status_code = getattr(exc, "status_code", None)
    if status_code == 401:
        return LLMProviderAuthError(str(exc))
    if status_code == 429:
        return LLMProviderRateLimitError(str(exc))
    return LLMProviderError(str(exc))


_STRUCTURED_OUTPUT_INSTRUCTION = (
    "Respond with ONLY a single valid JSON object matching this JSON Schema "
    "-- no prose, no markdown code fences, no explanation before or after "
    "the JSON:\n\n{schema}"
)


class ChatCompletionsProvider(LLMProvider):
    """Base for any LLMProvider backed by an OpenAI-style chat-completions
    client. Subclasses set `self._client` and `self._model` in `__init__`
    and implement `_map_exception` using their own SDK's exception
    classes; `generate`/`generate_structured` are implemented once here.
    """

    _client: ChatCompletionsClient
    _model: str

    def _map_exception(self, exc: Exception) -> Exception:
        raise NotImplementedError

    async def generate(
        self,
        *,
        messages: list[LLMMessage],
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMTextResponse:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=_build_messages(messages, system),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise self._map_exception(exc) from exc

        text, usage = _extract_text_and_usage(response)
        return LLMTextResponse(text=text, model=self._model, usage=usage)

    async def generate_structured(
        self,
        *,
        messages: list[LLMMessage],
        response_model: type[T],
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> T:
        from app.core.exceptions import LLMStructuredOutputError

        schema_instruction = _STRUCTURED_OUTPUT_INSTRUCTION.format(
            schema=json.dumps(response_model.model_json_schema())
        )
        combined_system = f"{system}\n\n{schema_instruction}" if system else schema_instruction
        payload_messages = _build_messages(messages, combined_system)

        last_error: Exception | None = None
        raw_text = ""
        for _ in range(2):  # one real attempt + one good-faith retry
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=payload_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                raise self._map_exception(exc) from exc

            raw_text, _ = _extract_text_and_usage(response)
            try:
                return response_model.model_validate_json(raw_text)
            except (ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                payload_messages = [
                    *payload_messages,
                    {"role": "assistant", "content": raw_text},
                    {
                        "role": "user",
                        "content": (
                            "That response was not valid JSON matching the schema. "
                            f"Error: {exc}. Respond again with ONLY the corrected JSON object."
                        ),
                    },
                ]

        raise LLMStructuredOutputError(
            f"Model response did not match {response_model.__name__}'s schema "
            f"after retry: {last_error}"
        ) from last_error
