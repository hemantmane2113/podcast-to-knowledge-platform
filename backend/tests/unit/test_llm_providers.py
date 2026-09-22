"""Tests for the LLMProvider abstraction (app/providers/llm/):
- factory selection + provider-specific key requirements
- structured-output JSON parsing + one-shot retry, exercised against a
  mocked chat-completions client (no real SDK network call).

app/providers/llm/groq_provider.py, openai_provider.py, and
opensource_provider.py are thin wrappers around
_chat_completions.ChatCompletionsProvider -- these tests exercise that
shared logic directly rather than duplicating it per concrete provider.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import groq
import httpx
import openai
import pytest
from pydantic import BaseModel

from app.config.settings import Settings
from app.core.exceptions import (
    LLMProviderAuthError,
    LLMProviderError,
    LLMProviderRateLimitError,
    LLMProviderRequestError,
    LLMProviderTransientError,
    LLMStructuredOutputError,
)
from app.providers.llm._chat_completions import (
    DEFAULT_CLIENT_MAX_RETRIES,
    DEFAULT_CLIENT_TIMEOUT_SECONDS,
    ChatCompletionsProvider,
    _is_model_json_validation_failure,
)
from app.providers.llm.base import LLMMessage
from app.providers.llm.factory import get_llm_provider
from app.providers.llm.fallback_provider import FallbackLLMProvider
from app.providers.llm.groq_provider import GroqProvider
from app.providers.llm.openai_provider import OpenAIProvider
from app.providers.llm.opensource_provider import OpenSourceProvider


class _Example(BaseModel):
    title: str
    count: int


def _fake_completion(content: str) -> SimpleNamespace:
    # Mimics the subset of an OpenAI/Groq ChatCompletion response object
    # that _chat_completions.py actually reads.
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class _StubChatCompletionsProvider(ChatCompletionsProvider):
    """A minimal concrete ChatCompletionsProvider for testing the shared
    generate/generate_structured logic without a real SDK client."""

    def __init__(self, client):
        self._client = client
        self._model = "stub-model"

    def _map_exception(self, exc: Exception) -> Exception:
        return exc


# --- factory: provider selection + key requirements -------------------------------


def test_factory_selects_groq_by_default() -> None:
    settings = Settings(
        _env_file=None,
        app_env="development",
        llm_provider="groq",
        groq_api_key="k",
        llm_model="m",
        openai_fallback_enabled=False,
    )
    assert isinstance(get_llm_provider(settings), GroqProvider)


def test_factory_selects_openai() -> None:
    settings = Settings(_env_file=None, app_env="development", llm_provider="openai", openai_api_key="k", llm_model="m")
    assert isinstance(get_llm_provider(settings), OpenAIProvider)


def test_factory_selects_opensource() -> None:
    settings = Settings(
        _env_file=None,
        app_env="development",
        llm_provider="opensource",
        opensource_base_url="http://localhost:8080/v1",
        llm_model="m",
        openai_fallback_enabled=False,
    )
    assert isinstance(get_llm_provider(settings), OpenSourceProvider)


def test_factory_raises_for_unknown_provider() -> None:
    settings = Settings(_env_file=None, app_env="development", llm_provider="groq", groq_api_key="k", llm_model="m")
    object.__setattr__(settings, "llm_provider", "not-a-provider")
    with pytest.raises(ValueError, match="not-a-provider"):
        get_llm_provider(settings)


# --- factory: fallback wrapping -----------------------------------------------------


def test_factory_wraps_groq_with_openai_fallback_when_enabled() -> None:
    settings = Settings(
        _env_file=None,
        app_env="development",
        llm_provider="groq",
        groq_api_key="k",
        llm_model="m",
        openai_fallback_enabled=True,
        openai_api_key="ok",
        openai_fallback_model="gpt-5.4",
    )
    provider = get_llm_provider(settings)
    assert isinstance(provider, FallbackLLMProvider)
    assert isinstance(provider._primary, GroqProvider)
    assert isinstance(provider._fallback, OpenAIProvider)


def test_factory_returns_bare_primary_when_fallback_disabled() -> None:
    settings = Settings(
        _env_file=None,
        app_env="development",
        llm_provider="groq",
        groq_api_key="k",
        llm_model="m",
        openai_fallback_enabled=False,
    )
    provider = get_llm_provider(settings)
    assert isinstance(provider, GroqProvider)
    assert not isinstance(provider, FallbackLLMProvider)


def test_factory_does_not_wrap_openai_primary_with_an_openai_fallback() -> None:
    """Primary == openai must never become FallbackLLMProvider(openai, openai)
    -- that would just retry the same provider against itself."""
    settings = Settings(
        _env_file=None,
        app_env="development",
        llm_provider="openai",
        openai_api_key="k",
        llm_model="m",
        openai_fallback_enabled=True,
    )
    provider = get_llm_provider(settings)
    assert isinstance(provider, OpenAIProvider)
    assert not isinstance(provider, FallbackLLMProvider)


def test_factory_raises_when_fallback_enabled_but_openai_key_missing() -> None:
    """Fails clearly at construction time, same pattern as a missing
    primary key -- no silent "fallback quietly unavailable" state."""
    settings = Settings(
        _env_file=None,
        app_env="development",
        llm_provider="groq",
        groq_api_key="k",
        llm_model="m",
        openai_fallback_enabled=True,
        openai_api_key="",
    )
    with pytest.raises(LLMProviderAuthError, match="OPENAI_API_KEY"):
        get_llm_provider(settings)


def test_groq_provider_requires_api_key() -> None:
    with pytest.raises(LLMProviderAuthError):
        GroqProvider(api_key="", model="m")


def test_groq_provider_requires_model() -> None:
    with pytest.raises(LLMProviderAuthError):
        GroqProvider(api_key="k", model="")


def test_opensource_provider_requires_base_url() -> None:
    with pytest.raises(LLMProviderAuthError):
        OpenSourceProvider(base_url="", model="m")


def test_opensource_provider_does_not_require_an_api_key() -> None:
    provider = OpenSourceProvider(base_url="http://localhost:8080/v1", model="m")
    assert provider is not None


# --- generate() ---------------------------------------------------------------------


async def test_generate_returns_text_and_usage() -> None:
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=_fake_completion("hello"))))
    )
    provider = _StubChatCompletionsProvider(client)

    from app.providers.llm.base import LLMMessage

    response = await provider.generate(messages=[LLMMessage(role="user", content="hi")])

    assert response.text == "hello"
    assert response.model == "stub-model"
    assert response.usage.total_tokens == 15


async def test_generate_wraps_client_exception_via_map_exception() -> None:
    class _Provider(_StubChatCompletionsProvider):
        def _map_exception(self, exc: Exception) -> Exception:
            return RuntimeError(f"mapped: {exc}")

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(side_effect=ValueError("boom")))
        )
    )
    provider = _Provider(client)

    from app.providers.llm.base import LLMMessage

    with pytest.raises(RuntimeError, match="mapped: boom"):
        await provider.generate(messages=[LLMMessage(role="user", content="hi")])


async def test_generate_omits_max_tokens_when_none() -> None:
    """max_tokens=None must be omitted from the SDK call entirely, not
    passed through as a literal null -- OpenAI's API rejects
    `max_tokens: null` with a 400 (found via a real Groq -> OpenAI
    fallback run)."""
    create = AsyncMock(return_value=_fake_completion("hello"))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider = _StubChatCompletionsProvider(client)

    await provider.generate(messages=[LLMMessage(role="user", content="hi")], max_tokens=None)

    assert "max_tokens" not in create.call_args.kwargs


async def test_generate_preserves_an_explicit_max_tokens() -> None:
    create = AsyncMock(return_value=_fake_completion("hello"))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider = _StubChatCompletionsProvider(client)

    await provider.generate(messages=[LLMMessage(role="user", content="hi")], max_tokens=256)

    assert create.call_args.kwargs["max_tokens"] == 256


# --- generate_structured(): parsing, retry, schema mismatch -------------------------


async def test_generate_structured_parses_valid_json_on_first_try() -> None:
    valid_json = json.dumps({"title": "hello", "count": 3})
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=_fake_completion(valid_json)))
        )
    )
    provider = _StubChatCompletionsProvider(client)

    from app.providers.llm.base import LLMMessage

    result = await provider.generate_structured(
        messages=[LLMMessage(role="user", content="give me json")], response_model=_Example
    )

    assert result == _Example(title="hello", count=3)
    assert client.chat.completions.create.call_count == 1


async def test_generate_structured_omits_max_tokens_when_none() -> None:
    create = AsyncMock(return_value=_fake_completion(json.dumps({"title": "x", "count": 1})))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider = _StubChatCompletionsProvider(client)

    from app.providers.llm.base import LLMMessage

    await provider.generate_structured(
        messages=[LLMMessage(role="user", content="give me json")], response_model=_Example, max_tokens=None
    )

    assert "max_tokens" not in create.call_args.kwargs


async def test_generate_structured_preserves_an_explicit_max_tokens() -> None:
    create = AsyncMock(return_value=_fake_completion(json.dumps({"title": "x", "count": 1})))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider = _StubChatCompletionsProvider(client)

    from app.providers.llm.base import LLMMessage

    await provider.generate_structured(
        messages=[LLMMessage(role="user", content="give me json")], response_model=_Example, max_tokens=512
    )

    assert create.call_args.kwargs["max_tokens"] == 512


async def test_generate_structured_retries_once_on_invalid_json_then_succeeds() -> None:
    bad_then_good = [
        _fake_completion("not json at all"),
        _fake_completion(json.dumps({"title": "fixed", "count": 1})),
    ]
    create = AsyncMock(side_effect=bad_then_good)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider = _StubChatCompletionsProvider(client)

    from app.providers.llm.base import LLMMessage

    result = await provider.generate_structured(
        messages=[LLMMessage(role="user", content="give me json")], response_model=_Example
    )

    assert result == _Example(title="fixed", count=1)
    assert create.call_count == 2
    # The retry prompt includes the bad output and an error explanation.
    second_call_messages = create.call_args_list[1].kwargs["messages"]
    assert any("not valid JSON" in m["content"] for m in second_call_messages)


async def test_generate_structured_raises_after_two_failed_attempts() -> None:
    create = AsyncMock(return_value=_fake_completion("still not json"))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider = _StubChatCompletionsProvider(client)

    from app.providers.llm.base import LLMMessage

    with pytest.raises(LLMStructuredOutputError):
        await provider.generate_structured(
            messages=[LLMMessage(role="user", content="give me json")], response_model=_Example
        )
    assert create.call_count == 2


async def test_generate_structured_rejects_json_missing_required_fields() -> None:
    incomplete = json.dumps({"title": "missing count"})
    complete = json.dumps({"title": "ok", "count": 2})
    create = AsyncMock(side_effect=[_fake_completion(incomplete), _fake_completion(complete)])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider = _StubChatCompletionsProvider(client)

    from app.providers.llm.base import LLMMessage

    result = await provider.generate_structured(
        messages=[LLMMessage(role="user", content="give me json")], response_model=_Example
    )
    assert result == _Example(title="ok", count=2)
    assert create.call_count == 2


async def test_generate_structured_includes_json_schema_in_system_prompt() -> None:
    create = AsyncMock(return_value=_fake_completion(json.dumps({"title": "x", "count": 0})))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider = _StubChatCompletionsProvider(client)

    from app.providers.llm.base import LLMMessage

    await provider.generate_structured(
        messages=[LLMMessage(role="user", content="hi")],
        response_model=_Example,
        system="Be helpful.",
    )

    sent_messages = create.call_args_list[0].kwargs["messages"]
    assert sent_messages[0]["role"] == "system"
    assert "Be helpful." in sent_messages[0]["content"]
    assert "\"title\"" in sent_messages[0]["content"]  # schema property present
    assert create.call_args_list[0].kwargs["response_format"] == {"type": "json_object"}


# --- generate_structured(): Groq's own `json_validate_failed` structured------------
# --- output failure is a model-generation problem, not a malformed request ---------
# (see app/providers/llm/_chat_completions.py's _is_model_json_validation_failure
# docstring/comment) -- it must enter the SAME local corrective-retry loop as a
# response that fails our own Pydantic validation, not be raised immediately as a
# non-retryable LLMProviderRequestError the way a genuinely malformed 400 is.


def _groq_json_validate_failed_error(message: str = "Failed to validate JSON.") -> "groq.BadRequestError":
    # Mirrors the real Groq error body shape (groq.types.shared.error_object
    # .ErrorObject): {"error": {"message": ..., "type": ..., "code":
    # "json_validate_failed", "failed_generation": ...}}.
    return groq.BadRequestError(
        message,
        response=_httpx_response(400),
        body={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": "json_validate_failed",
                "failed_generation": "",
            }
        },
    )


def test_groq_json_validate_failed_is_recognized_as_model_generation_failure() -> None:
    # (A) The specific Groq structured-output validation failure is
    # distinguished from a generic malformed-request 400.
    exc = _groq_json_validate_failed_error("Failed to validate JSON. Please adjust your prompt.")
    assert _is_model_json_validation_failure(exc) is True


def test_generic_bad_request_is_not_a_model_json_validation_failure() -> None:
    # A 400 with no body, or a body/code that isn't json_validate_failed,
    # must never be misclassified as this specific failure.
    assert _is_model_json_validation_failure(groq.BadRequestError("bad", response=_httpx_response(400), body=None)) is False
    other_code = groq.BadRequestError(
        "bad model",
        response=_httpx_response(400),
        body={"error": {"message": "bad model", "type": "invalid_request_error", "code": "model_not_found"}},
    )
    assert _is_model_json_validation_failure(other_code) is False


async def test_generate_structured_enters_corrective_retry_on_groq_json_validate_failed() -> None:
    # (B) The failure must not propagate immediately -- it enters the same
    # local retry loop as an invalid-JSON response, with a corrective
    # follow-up message appended (mirroring the "not valid JSON" retry
    # path already tested above).
    create = AsyncMock(
        side_effect=[_groq_json_validate_failed_error(), _fake_completion(json.dumps({"title": "fixed", "count": 1}))]
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider = _StubChatCompletionsProvider(client)

    result = await provider.generate_structured(
        messages=[LLMMessage(role="user", content="give me json")], response_model=_Example
    )

    assert create.call_count == 2
    second_call_messages = create.call_args_list[1].kwargs["messages"]
    assert any("JSON validation" in m["content"] for m in second_call_messages)
    # (C) The subsequent valid response succeeds normally.
    assert result == _Example(title="fixed", count=1)


async def test_generate_structured_exhausts_retry_and_raises_structured_output_error_when_still_failing() -> None:
    # "if the retry still fails -> preserve the existing failure behavior":
    # two consecutive json_validate_failed responses must end in the same
    # LLMStructuredOutputError that two consecutive invalid-JSON responses
    # already end in (test_generate_structured_raises_after_two_failed_attempts).
    create = AsyncMock(side_effect=[_groq_json_validate_failed_error(), _groq_json_validate_failed_error()])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    provider = _StubChatCompletionsProvider(client)

    with pytest.raises(LLMStructuredOutputError):
        await provider.generate_structured(
            messages=[LLMMessage(role="user", content="give me json")], response_model=_Example
        )
    assert create.call_count == 2


async def test_generate_structured_still_raises_immediately_for_a_genuine_malformed_request() -> None:
    # (D) A normal HTTP 400 (no json_validate_failed code) must remain
    # non-retryable: no local retry, mapped straight through
    # _map_exception exactly as before this fix.
    create = AsyncMock(side_effect=groq.BadRequestError("malformed", response=_httpx_response(400), body=None))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    class _Provider(_StubChatCompletionsProvider):
        def _map_exception(self, exc: Exception) -> Exception:
            return LLMProviderRequestError(str(exc))

    provider = _Provider(client)

    with pytest.raises(LLMProviderRequestError):
        await provider.generate_structured(
            messages=[LLMMessage(role="user", content="give me json")], response_model=_Example
        )
    assert create.call_count == 1


async def test_generate_structured_fallback_to_openai_still_works_for_a_genuine_rate_limit() -> None:
    # (E) Existing fallback behavior for a genuinely retryable primary
    # failure (429) is unaffected by this fix -- the new check only ever
    # looks at HTTP 400 responses, so a 429 takes the exact same path as
    # before through map_sdk_exception -> LLMProviderRateLimitError ->
    # FallbackLLMProvider.
    with (
        patch("app.providers.llm.groq_provider.AsyncGroq") as mock_groq_cls,
        patch("app.providers.llm.openai_provider.AsyncOpenAI") as mock_openai_cls,
    ):
        mock_groq_cls.return_value.chat.completions.create = AsyncMock(
            side_effect=groq.RateLimitError("slow down", response=_httpx_response(429), body=None)
        )
        mock_openai_cls.return_value.chat.completions.create = AsyncMock(
            return_value=_fake_completion(json.dumps({"title": "from openai", "count": 1}))
        )

        groq_provider = GroqProvider(api_key="k", model="m")
        openai_provider = OpenAIProvider(api_key="k", model="m")
        provider = FallbackLLMProvider(primary=groq_provider, fallback=openai_provider)

        result = await provider.generate_structured(
            messages=[LLMMessage(role="user", content="give me json")], response_model=_Example
        )

    assert result == _Example(title="from openai", count=1)
    mock_openai_cls.return_value.chat.completions.create.assert_awaited_once()


# --- concrete providers: client construction, end-to-end generate(), and -----------
# --- each provider's own _map_exception against REAL SDK exception classes ---------
# (the shared-base tests above only prove ChatCompletionsProvider itself is correct;
# these prove GroqProvider/OpenAIProvider/OpenSourceProvider wire it up correctly).


def _httpx_response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return httpx.Response(status_code=status_code, request=request)


def _httpx_request() -> httpx.Request:
    return httpx.Request("POST", "https://example.invalid/v1/chat/completions")


async def test_groq_provider_generate_end_to_end_through_real_client_construction() -> None:
    with patch("app.providers.llm.groq_provider.AsyncGroq") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=_fake_completion("hello from groq"))

        provider = GroqProvider(api_key="real-key", model="llama-3.3-70b-versatile")
        mock_client_cls.assert_called_once_with(
            api_key="real-key",
            timeout=DEFAULT_CLIENT_TIMEOUT_SECONDS,
            max_retries=DEFAULT_CLIENT_MAX_RETRIES,
        )

        response = await provider.generate(messages=[LLMMessage(role="user", content="hi")])

    assert response.text == "hello from groq"
    assert response.model == "llama-3.3-70b-versatile"
    mock_client.chat.completions.create.assert_awaited_once()
    assert mock_client.chat.completions.create.call_args.kwargs["model"] == "llama-3.3-70b-versatile"


def test_groq_provider_maps_authentication_error() -> None:
    provider = GroqProvider(api_key="k", model="m")
    exc = groq.AuthenticationError("bad key", response=_httpx_response(401), body=None)
    assert isinstance(provider._map_exception(exc), LLMProviderAuthError)


def test_groq_provider_maps_rate_limit_error() -> None:
    provider = GroqProvider(api_key="k", model="m")
    exc = groq.RateLimitError("slow down", response=_httpx_response(429), body=None)
    assert isinstance(provider._map_exception(exc), LLMProviderRateLimitError)


def test_groq_provider_maps_connection_error_to_retryable_transient() -> None:
    provider = GroqProvider(api_key="k", model="m")
    exc = groq.APIConnectionError(request=_httpx_request())
    mapped = provider._map_exception(exc)
    assert type(mapped) is LLMProviderTransientError
    assert mapped.retryable is True


def test_groq_provider_maps_server_error_to_retryable_transient() -> None:
    provider = GroqProvider(api_key="k", model="m")
    exc = groq.InternalServerError("oops", response=_httpx_response(500), body=None)
    mapped = provider._map_exception(exc)
    assert type(mapped) is LLMProviderTransientError
    assert mapped.retryable is True


def test_groq_provider_maps_bad_request_to_non_retryable_request_error() -> None:
    provider = GroqProvider(api_key="k", model="m")
    exc = groq.BadRequestError("malformed", response=_httpx_response(400), body=None)
    mapped = provider._map_exception(exc)
    assert type(mapped) is LLMProviderRequestError
    assert mapped.retryable is False


def test_groq_provider_maps_unknown_exception_to_generic_provider_error() -> None:
    provider = GroqProvider(api_key="k", model="m")
    mapped = provider._map_exception(ValueError("something else"))
    assert type(mapped) is LLMProviderError
    assert mapped.retryable is True


async def test_openai_provider_generate_end_to_end_through_real_client_construction() -> None:
    with patch("app.providers.llm.openai_provider.AsyncOpenAI") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=_fake_completion("hello from openai"))

        provider = OpenAIProvider(api_key="real-key", model="gpt-4o-mini")
        mock_client_cls.assert_called_once_with(
            api_key="real-key",
            timeout=DEFAULT_CLIENT_TIMEOUT_SECONDS,
            max_retries=DEFAULT_CLIENT_MAX_RETRIES,
        )

        response = await provider.generate(messages=[LLMMessage(role="user", content="hi")])

    assert response.text == "hello from openai"
    mock_client.chat.completions.create.assert_awaited_once()


def test_openai_provider_maps_authentication_error() -> None:
    provider = OpenAIProvider(api_key="k", model="m")
    exc = openai.AuthenticationError("bad key", response=_httpx_response(401), body=None)
    assert isinstance(provider._map_exception(exc), LLMProviderAuthError)


def test_openai_provider_maps_rate_limit_error() -> None:
    provider = OpenAIProvider(api_key="k", model="m")
    exc = openai.RateLimitError("slow down", response=_httpx_response(429), body=None)
    assert isinstance(provider._map_exception(exc), LLMProviderRateLimitError)


async def test_opensource_provider_generate_end_to_end_uses_configured_base_url() -> None:
    with patch("app.providers.llm.opensource_provider.AsyncOpenAI") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=_fake_completion("hello from vllm"))

        provider = OpenSourceProvider(base_url="http://localhost:8080/v1", model="local-model")
        mock_client_cls.assert_called_once_with(
            base_url="http://localhost:8080/v1",
            api_key="not-required",
            timeout=DEFAULT_CLIENT_TIMEOUT_SECONDS,
            max_retries=DEFAULT_CLIENT_MAX_RETRIES,
        )

        response = await provider.generate(messages=[LLMMessage(role="user", content="hi")])

    assert response.text == "hello from vllm"


def test_opensource_provider_passes_through_a_real_api_key_when_given() -> None:
    with patch("app.providers.llm.opensource_provider.AsyncOpenAI") as mock_client_cls:
        OpenSourceProvider(base_url="http://localhost:8080/v1", model="m", api_key="secret")
        mock_client_cls.assert_called_once_with(
            base_url="http://localhost:8080/v1",
            api_key="secret",
            timeout=DEFAULT_CLIENT_TIMEOUT_SECONDS,
            max_retries=DEFAULT_CLIENT_MAX_RETRIES,
        )


def test_opensource_provider_maps_authentication_error() -> None:
    provider = OpenSourceProvider(base_url="http://localhost:8080/v1", model="m")
    exc = openai.AuthenticationError("bad key", response=_httpx_response(401), body=None)
    assert isinstance(provider._map_exception(exc), LLMProviderAuthError)
