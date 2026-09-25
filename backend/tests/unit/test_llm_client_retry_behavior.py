"""Empirical proof that DEFAULT_CLIENT_MAX_RETRIES=0
(app/providers/llm/_chat_completions.py) actually stops the real groq/openai
SDK from retrying a 429 internally -- not just that we *pass* the kwarg at
construction (test_llm_providers.py's mocked-client assertions only prove
that), but that the real, unmocked SDK client *behaves* accordingly.

This is the exact incident reported from a real pipeline run: the Groq
SDK's own retry-on-429 (honoring a Retry-After header, which can itself be
100s+) consumed most of arq's 300s job timeout before FallbackLLMProvider
ever got a chance to run -- the SDK's own retry loop
(_sleep_for_retry -> anyio.sleep) intercepted the failure before it ever
reached our code.

respx intercepts the HTTP layer only -- GroqProvider/OpenAIProvider
construct real, unmocked AsyncGroq/AsyncOpenAI clients here, exercising
the actual installed SDK's own request/retry loop
(groq/openai's _base_client.py). No real network call reaches Groq's or
OpenAI's servers; respx's transport patch intercepts before any socket is
opened.
"""

import time

import httpx
import pytest
import respx

from app.core.exceptions import LLMProviderRateLimitError
from app.providers.llm.base import LLMMessage
from app.providers.llm.fallback_provider import FallbackLLMProvider
from app.providers.llm.groq_provider import GroqProvider
from app.providers.llm.openai_provider import OpenAIProvider

_GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
_OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"

# A real 429 with a long Retry-After -- exactly the shape that made the
# SDK's own retry logic previously consume most of arq's 300s job timeout.
_LONG_RETRY_AFTER_SECONDS = "120"

# Generous relative to the near-instant (sub-second) response an actually
# non-retrying client produces, but nowhere near the ~120s a single SDK-level
# retry honoring Retry-After would take -- a robust bound either way.
_MAX_ACCEPTABLE_ELAPSED_SECONDS = 5.0


def _rate_limited_response() -> httpx.Response:
    return httpx.Response(
        429,
        json={"error": {"message": "rate limited"}},
        headers={"Retry-After": _LONG_RETRY_AFTER_SECONDS},
    )


def _ok_response(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


@respx.mock
async def test_groq_client_does_not_retry_a_429_and_raises_immediately() -> None:
    route = respx.post(_GROQ_CHAT_COMPLETIONS_URL).mock(return_value=_rate_limited_response())
    provider = GroqProvider(api_key="k", model="m")

    start = time.monotonic()
    with pytest.raises(LLMProviderRateLimitError):
        await provider.generate(messages=[LLMMessage(role="user", content="hi")])
    elapsed = time.monotonic() - start

    assert route.call_count == 1  # no SDK-level retry attempted
    assert elapsed < _MAX_ACCEPTABLE_ELAPSED_SECONDS


@respx.mock
async def test_openai_client_does_not_retry_a_429_and_raises_immediately() -> None:
    route = respx.post(_OPENAI_CHAT_COMPLETIONS_URL).mock(return_value=_rate_limited_response())
    provider = OpenAIProvider(api_key="k", model="m")

    start = time.monotonic()
    with pytest.raises(LLMProviderRateLimitError):
        await provider.generate(messages=[LLMMessage(role="user", content="hi")])
    elapsed = time.monotonic() - start

    assert route.call_count == 1
    assert elapsed < _MAX_ACCEPTABLE_ELAPSED_SECONDS


@respx.mock
async def test_real_groq_429_reaches_fallback_layer_and_openai_is_called() -> None:
    """End-to-end with real (HTTP-layer-mocked) SDK clients on both sides: a
    real Groq 429 must reach FallbackLLMProvider as LLMProviderRateLimitError
    and trigger a real call to the OpenAI fallback client -- not be swallowed
    or delayed by SDK-level retries first."""
    groq_route = respx.post(_GROQ_CHAT_COMPLETIONS_URL).mock(return_value=_rate_limited_response())
    openai_route = respx.post(_OPENAI_CHAT_COMPLETIONS_URL).mock(
        return_value=_ok_response("from openai fallback")
    )

    primary = GroqProvider(api_key="k", model="m")
    fallback = OpenAIProvider(api_key="k", model="gpt-5.4")
    provider = FallbackLLMProvider(primary=primary, fallback=fallback)

    start = time.monotonic()
    response = await provider.generate(messages=[LLMMessage(role="user", content="hi")])
    elapsed = time.monotonic() - start

    assert response.text == "from openai fallback"
    assert groq_route.call_count == 1
    assert openai_route.call_count == 1
    assert elapsed < _MAX_ACCEPTABLE_ELAPSED_SECONDS
