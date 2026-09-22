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

# SDK-level retries are disabled (max_retries=0) and the per-request
# timeout is kept well under arq's job timeout (default 300s, unchanged --
# see app/worker/settings.py): a 429's Retry-After header can itself be
# 100s+, and the SDK default of 2 retries compounds that, which previously
# consumed the entire job timeout before our own error handling (now
# FallbackLLMProvider) ever got a chance to run. A single failed attempt
# maps to a retryable provider error below and is handled by switching
# providers, not by waiting inside the SDK.
DEFAULT_CLIENT_TIMEOUT_SECONDS = 60.0
DEFAULT_CLIENT_MAX_RETRIES = 0


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


def _build_create_kwargs(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
    response_format: dict[str, str] | None = None,
) -> dict[str, Any]:
    """`max_tokens=None` must be *omitted*, not passed through as a literal
    null -- OpenAI's API rejects `max_tokens: null` with a 400 ("Invalid
    type for 'max_tokens': expected an unsupported value, but got null
    instead"), found via a real fallback run: Groq tolerates `null` here,
    OpenAI (used as the fallback provider) does not, so the difference
    only ever surfaced once a real Groq failure actually triggered the
    OpenAI path."""
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        kwargs["response_format"] = response_format
    return kwargs


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
        LLMProviderRequestError,
        LLMProviderTransientError,
    )

    if isinstance(exc, auth_error_type):
        return LLMProviderAuthError(str(exc))
    if isinstance(exc, rate_limit_error_type):
        return LLMProviderRateLimitError(str(exc))
    if isinstance(exc, connection_error_type):
        return LLMProviderTransientError(str(exc))  # network/timeout, retryable

    status_code = getattr(exc, "status_code", None)
    if status_code == 401:
        return LLMProviderAuthError(str(exc))
    if status_code == 429:
        return LLMProviderRateLimitError(str(exc))
    if status_code == 400:
        return LLMProviderRequestError(str(exc))  # malformed request, not retryable
    if status_code is not None and 500 <= status_code < 600:
        return LLMProviderTransientError(str(exc))  # server error, retryable
    return LLMProviderError(str(exc))  # unknown -- retryable by default (existing behavior)


_STRUCTURED_OUTPUT_INSTRUCTION = (
    "Respond with ONLY a single valid JSON object matching this JSON Schema "
    "-- no prose, no markdown code fences, no explanation before or after "
    "the JSON:\n\n{schema}"
)

# Groq's `response_format={"type": "json_object"}` performs its own
# server-side check that the model's generation is syntactically valid
# JSON, and rejects the request with HTTP 400 + `error.code ==
# "json_validate_failed"` if the model failed to produce it -- this is a
# *model-generation* failure (the same class of problem `generate_structured`
# already retries locally via `_STRUCTURED_OUTPUT_INSTRUCTION`'s corrective
# feedback loop when our own `response_model.model_validate_json(raw_text)`
# fails below), not a malformed *request* from us. Found via a real run:
# it was falling into the generic `status_code == 400 ->
# LLMProviderRequestError` branch of map_sdk_exception (correct for an
# actually-malformed request, e.g. bad params) before any text was ever
# returned to retry against -- short-circuiting the local corrective-retry
# loop entirely and, since LLMProviderRequestError.retryable is False,
# also skipping FallbackLLMProvider's provider fallback. Deliberately
# scoped to this one Groq error code: every other 400 (including a
# genuinely malformed request) is unaffected, and this never triggers
# provider fallback -- only the existing local retry.
_MODEL_JSON_VALIDATION_ERROR_CODE = "json_validate_failed"


def _is_model_json_validation_failure(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) != 400:
        return False
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return False
    error = body.get("error")
    if not isinstance(error, dict):
        return False
    return error.get("code") == _MODEL_JSON_VALIDATION_ERROR_CODE


def _describe_model_json_validation_failure(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    return str(exc)


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
                **_build_create_kwargs(
                    model=self._model,
                    messages=_build_messages(messages, system),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
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
                    **_build_create_kwargs(
                        model=self._model,
                        messages=payload_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"},
                    )
                )
            except Exception as exc:
                if not _is_model_json_validation_failure(exc):
                    raise self._map_exception(exc) from exc
                # The provider rejected the request before returning any
                # content (see _is_model_json_validation_failure above) --
                # there's no raw_text to include as a prior "assistant"
                # turn, unlike the local-validation-failure branch below.
                last_error = exc
                payload_messages = [
                    *payload_messages,
                    {
                        "role": "user",
                        "content": (
                            "Your previous response failed JSON validation: "
                            f"{_describe_model_json_validation_failure(exc)}. Respond again with ONLY "
                            "a single valid JSON object matching the schema given above -- no prose, "
                            "no markdown code fences, and ensure every string is properly JSON-escaped."
                        ),
                    },
                ]
                continue

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
