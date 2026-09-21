"""LLMProvider — isolates the rest of the application from any specific
inference provider's SDK (mirrors app/providers/transcript/base.py's
TranscriptProvider pattern for the same reason: PRODUCT_SPEC.md §11/§93,
"every external dependency is abstracted behind a provider interface").

app/ai/ (topic analysis, article planning, section generation, validation)
depends on this abstraction only — never on `groq`, `openai`, or any other
provider SDK directly. A provider-specific SDK call belongs exclusively
inside that provider's own module under app/providers/llm/.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, TypeVar

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class LLMMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class LLMTextResponse:
    text: str
    model: str
    usage: LLMUsage | None = field(default=None)


class LLMProvider(ABC):
    """A chat-style LLM backend. Implementations translate this call shape
    into their own SDK's request/response format and raise
    app.core.exceptions LLMProviderError subclasses on failure — never a
    provider-specific exception type or raw SDK error (same discipline as
    TranscriptProvider).
    """

    @abstractmethod
    async def generate(
        self,
        *,
        messages: list[LLMMessage],
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMTextResponse:
        """Plain text generation."""
        ...

    @abstractmethod
    async def generate_structured(
        self,
        *,
        messages: list[LLMMessage],
        response_model: type[T],
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> T:
        """Generation constrained to `response_model`'s JSON schema,
        parsed and validated into an instance of it. Implementations
        should retry once on a parse/validation failure (e.g. by
        appending the validation error to the prompt) before raising
        LLMStructuredOutputError — a good-faith one-shot recovery, not
        unbounded retries.
        """
        ...
