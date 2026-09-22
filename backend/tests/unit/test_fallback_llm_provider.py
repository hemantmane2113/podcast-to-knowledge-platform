"""Tests for FallbackLLMProvider (app/providers/llm/fallback_provider.py):
the decision logic that switches to the fallback provider on a retryable
primary failure (rate limit / 5xx / connection) and never does on a
non-retryable one (auth / config / malformed request). No real Groq/OpenAI
SDK calls -- both primary and fallback are fakes.
"""

import pytest
from pydantic import BaseModel

from app.core.exceptions import (
    LLMProviderAuthError,
    LLMProviderError,
    LLMProviderRateLimitError,
    LLMProviderRequestError,
    LLMProviderTransientError,
    LLMStructuredOutputError,
)
from app.providers.llm.base import LLMMessage, LLMTextResponse, LLMUsage
from app.providers.llm.fallback_provider import FallbackLLMProvider


class _Example(BaseModel):
    title: str


class _FakeProvider:
    """A bare-bones LLMProvider: raises `error` if given, otherwise
    returns a canned response tagged with `name` so tests can tell which
    provider actually produced a result."""

    def __init__(self, name: str, *, error: Exception | None = None, structured_result: BaseModel | None = None):
        self.name = name
        self.error = error
        self.structured_result = structured_result
        self.generate_calls = 0
        self.generate_structured_calls = 0

    async def generate(self, **kwargs) -> LLMTextResponse:
        self.generate_calls += 1
        if self.error is not None:
            raise self.error
        return LLMTextResponse(text=f"from {self.name}", model=self.name, usage=LLMUsage(1, 1, 2))

    async def generate_structured(self, **kwargs):
        self.generate_structured_calls += 1
        if self.error is not None:
            raise self.error
        return self.structured_result or _Example(title=self.name)


_MESSAGES = [LLMMessage(role="user", content="hi")]


async def test_groq_succeeds_openai_not_called() -> None:
    primary = _FakeProvider("groq")
    fallback = _FakeProvider("openai")
    provider = FallbackLLMProvider(primary=primary, fallback=fallback)

    response = await provider.generate(messages=_MESSAGES)

    assert response.text == "from groq"
    assert fallback.generate_calls == 0


async def test_groq_rate_limit_falls_back_to_openai() -> None:
    primary = _FakeProvider("groq", error=LLMProviderRateLimitError("429"))
    fallback = _FakeProvider("openai")
    provider = FallbackLLMProvider(primary=primary, fallback=fallback)

    response = await provider.generate(messages=_MESSAGES)

    assert response.text == "from openai"
    assert fallback.generate_calls == 1


async def test_groq_server_error_falls_back_to_openai() -> None:
    primary = _FakeProvider("groq", error=LLMProviderTransientError("500"))
    fallback = _FakeProvider("openai")
    provider = FallbackLLMProvider(primary=primary, fallback=fallback)

    response = await provider.generate(messages=_MESSAGES)

    assert response.text == "from openai"
    assert fallback.generate_calls == 1


async def test_groq_connection_error_falls_back_to_openai() -> None:
    # APIConnectionError/APITimeoutError both map to LLMProviderTransientError
    # (app/providers/llm/_chat_completions.py) -- same case as server errors.
    primary = _FakeProvider("groq", error=LLMProviderTransientError("connection reset"))
    fallback = _FakeProvider("openai")
    provider = FallbackLLMProvider(primary=primary, fallback=fallback)

    response = await provider.generate(messages=_MESSAGES)

    assert response.text == "from openai"
    assert fallback.generate_calls == 1


async def test_groq_authentication_failure_does_not_fall_back() -> None:
    primary = _FakeProvider("groq", error=LLMProviderAuthError("GROQ_API_KEY is not configured"))
    fallback = _FakeProvider("openai")
    provider = FallbackLLMProvider(primary=primary, fallback=fallback)

    with pytest.raises(LLMProviderAuthError):
        await provider.generate(messages=_MESSAGES)
    assert fallback.generate_calls == 0


async def test_groq_configuration_failure_does_not_fall_back() -> None:
    # A malformed/application-side request (HTTP 400) maps to
    # LLMProviderRequestError, retryable=False -- same "don't hide a
    # config/programming bug behind a fallback" reasoning as auth errors.
    primary = _FakeProvider("groq", error=LLMProviderRequestError("malformed request"))
    fallback = _FakeProvider("openai")
    provider = FallbackLLMProvider(primary=primary, fallback=fallback)

    with pytest.raises(LLMProviderRequestError):
        await provider.generate(messages=_MESSAGES)
    assert fallback.generate_calls == 0


async def test_structured_output_failure_does_not_fall_back() -> None:
    # LLMStructuredOutputError (retryable=False): the model's own output
    # was malformed twice in a row -- not a provider connectivity issue.
    primary = _FakeProvider("groq", error=LLMStructuredOutputError("bad json"))
    fallback = _FakeProvider("openai")
    provider = FallbackLLMProvider(primary=primary, fallback=fallback)

    with pytest.raises(LLMStructuredOutputError):
        await provider.generate_structured(messages=_MESSAGES, response_model=_Example)
    assert fallback.generate_structured_calls == 0


async def test_openai_fallback_succeeds_final_result_returned() -> None:
    primary = _FakeProvider("groq", error=LLMProviderRateLimitError("429"))
    fallback = _FakeProvider("openai")
    provider = FallbackLLMProvider(primary=primary, fallback=fallback)

    response = await provider.generate(messages=_MESSAGES)

    assert response.text == "from openai"
    assert response.model == "openai"


async def test_openai_fallback_also_fails_meaningful_exception_propagates() -> None:
    primary = _FakeProvider("groq", error=LLMProviderRateLimitError("429"))
    fallback = _FakeProvider("openai", error=LLMProviderAuthError("OPENAI_API_KEY is not configured"))
    provider = FallbackLLMProvider(primary=primary, fallback=fallback)

    with pytest.raises(LLMProviderAuthError, match="OPENAI_API_KEY"):
        await provider.generate(messages=_MESSAGES)


async def test_generate_structured_falls_back_on_retryable_primary_failure() -> None:
    primary = _FakeProvider("groq", error=LLMProviderTransientError("500"))
    fallback = _FakeProvider("openai", structured_result=_Example(title="from fallback"))
    provider = FallbackLLMProvider(primary=primary, fallback=fallback)

    result = await provider.generate_structured(messages=_MESSAGES, response_model=_Example)

    assert result.title == "from fallback"
    assert fallback.generate_structured_calls == 1


async def test_generate_structured_no_fallback_when_primary_succeeds() -> None:
    primary = _FakeProvider("groq", structured_result=_Example(title="from groq"))
    fallback = _FakeProvider("openai")
    provider = FallbackLLMProvider(primary=primary, fallback=fallback)

    result = await provider.generate_structured(messages=_MESSAGES, response_model=_Example)

    assert result.title == "from groq"
    assert fallback.generate_structured_calls == 0


async def test_unknown_provider_error_is_retryable_by_default_and_falls_back() -> None:
    # Bare LLMProviderError (unrecognized SDK exception) keeps its
    # existing retryable=True default -- fallback still gets a chance
    # rather than failing the job outright on an ambiguous error.
    primary = _FakeProvider("groq", error=LLMProviderError("something unexpected"))
    fallback = _FakeProvider("openai")
    provider = FallbackLLMProvider(primary=primary, fallback=fallback)

    response = await provider.generate(messages=_MESSAGES)

    assert response.text == "from openai"


async def test_non_provider_exception_is_never_retryable_and_propagates() -> None:
    # A plain Python exception (bug, not a mapped provider error) has no
    # .retryable attribute -- must never trigger a fallback attempt.
    primary = _FakeProvider("groq", error=RuntimeError("unexpected bug"))
    fallback = _FakeProvider("openai")
    provider = FallbackLLMProvider(primary=primary, fallback=fallback)

    with pytest.raises(RuntimeError):
        await provider.generate(messages=_MESSAGES)
    assert fallback.generate_calls == 0
