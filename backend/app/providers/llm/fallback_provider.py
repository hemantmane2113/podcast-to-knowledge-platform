"""Wraps a primary LLMProvider with a fallback: on a *retryable* primary
failure (rate limit, 5xx, connection/timeout -- anything whose mapped
exception has `.retryable is True`, see app/core/exceptions.py), the same
call is retried once against the fallback provider instead of failing the
job. A non-retryable failure (auth, missing config, a malformed/400
request) is never retried against the fallback either -- switching
providers wouldn't fix a bad credential or a bad request, and silently
retrying would hide a real configuration problem instead of surfacing it.

AI nodes (app/ai/nodes/) never see this: FallbackLLMProvider implements
the same LLMProvider interface as GroqProvider/OpenAIProvider/
OpenSourceProvider, so it's a drop-in substitute constructed once by
app.providers.llm.factory.get_llm_provider -- nothing above the factory
knows fallback exists.
"""

import logging

from app.providers.llm.base import LLMMessage, LLMProvider, LLMTextResponse, T

logger = logging.getLogger(__name__)


def _is_retryable(exc: Exception) -> bool:
    return getattr(exc, "retryable", False) is True


class FallbackLLMProvider(LLMProvider):
    def __init__(self, primary: LLMProvider, fallback: LLMProvider):
        self._primary = primary
        self._fallback = fallback

    async def generate(
        self,
        *,
        messages: list[LLMMessage],
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMTextResponse:
        try:
            return await self._primary.generate(
                messages=messages, system=system, temperature=temperature, max_tokens=max_tokens
            )
        except Exception as exc:
            if not _is_retryable(exc):
                raise
            self._log_fallback_triggered(exc)
            try:
                result = await self._fallback.generate(
                    messages=messages, system=system, temperature=temperature, max_tokens=max_tokens
                )
            except Exception as fallback_exc:
                self._log_fallback_failed(fallback_exc)
                raise
            logger.info(
                "LLM fallback provider succeeded provider=%s model=%s",
                type(self._fallback).__name__,
                result.model,
            )
            return result

    async def generate_structured(
        self,
        *,
        messages: list[LLMMessage],
        response_model: type[T],
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> T:
        try:
            return await self._primary.generate_structured(
                messages=messages,
                response_model=response_model,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            if not _is_retryable(exc):
                raise
            self._log_fallback_triggered(exc)
            try:
                result = await self._fallback.generate_structured(
                    messages=messages,
                    response_model=response_model,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as fallback_exc:
                self._log_fallback_failed(fallback_exc)
                raise
            logger.info("LLM fallback provider succeeded provider=%s", type(self._fallback).__name__)
            return result

    def _log_fallback_triggered(self, exc: Exception) -> None:
        logger.warning(
            "LLM primary provider failed provider=%s error_type=%s fallback_provider=%s action=fallback",
            type(self._primary).__name__,
            type(exc).__name__,
            type(self._fallback).__name__,
        )

    def _log_fallback_failed(self, fallback_exc: Exception) -> None:
        logger.error(
            "LLM fallback provider failed provider=%s error_type=%s",
            type(self._fallback).__name__,
            type(fallback_exc).__name__,
        )
