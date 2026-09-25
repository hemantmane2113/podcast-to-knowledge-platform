"""Wraps a primary LLMProvider with a fallback: on a *retryable* primary
failure (rate limit, 5xx, connection/timeout -- anything whose mapped
exception has `.retryable is True`, see app/core/exceptions.py), the same
call is retried once against the fallback provider instead of failing the
job. A non-retryable failure (auth, missing config, a malformed/400
request) is never retried against the fallback either -- switching
providers wouldn't fix a bad credential or a bad request, and silently
retrying would hide a real configuration problem instead of surfacing it.

Sticky per instance: once a fallback succeeds, every later call on THIS
instance goes straight to the fallback provider -- the primary is not
tried again for the rest of this instance's life. Without this, a
sustained primary outage (e.g. a rate limit that doesn't clear for a
while) makes every single call in a job independently retry-then-fail
against the primary first, which is pure wasted latency once the first
call has already shown the primary is down. This is deliberately scoped
to *this instance*, not global/persistent state: app.providers.llm.factory
.get_llm_provider constructs a new FallbackLLMProvider for every
article-generation job (never cached/reused across jobs), so stickiness
naturally resets for the next job with no explicit reset call, no DB
write, and no change to global config.

AI nodes (app/ai/nodes/) never see any of this: FallbackLLMProvider
implements the same LLMProvider interface as GroqProvider/OpenAIProvider/
OpenSourceProvider, so it's a drop-in substitute constructed once by
app.providers.llm.factory.get_llm_provider -- nothing above the factory
knows fallback (sticky or otherwise) exists.
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
        # Flips to True only once a fallback call has actually *succeeded*
        # (never merely attempted -- see generate()/generate_structured()
        # below, both set this after the fallback call returns, inside the
        # same try block whose except re-raises on failure without setting
        # it). A local structured-output corrective retry never reaches
        # this class at all -- it's fully contained inside whichever
        # concrete provider (primary or fallback) is actually handling the
        # call, in ChatCompletionsProvider.generate_structured -- so it
        # can't observe or reset this flag either way.
        self._use_fallback = False

    async def generate(
        self,
        *,
        messages: list[LLMMessage],
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMTextResponse:
        if self._use_fallback:
            return await self._fallback.generate(
                messages=messages, system=system, temperature=temperature, max_tokens=max_tokens
            )
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
            self._use_fallback = True
            logger.info(
                "LLM fallback provider succeeded provider=%s model=%s -- sticky for the rest of this job",
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
        if self._use_fallback:
            return await self._fallback.generate_structured(
                messages=messages,
                response_model=response_model,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
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
            self._use_fallback = True
            logger.info(
                "LLM fallback provider succeeded provider=%s -- sticky for the rest of this job",
                type(self._fallback).__name__,
            )
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
