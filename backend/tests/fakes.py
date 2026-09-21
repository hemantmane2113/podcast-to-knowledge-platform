import uuid

from pydantic import BaseModel

from app.providers.llm.base import LLMMessage, LLMProvider, LLMTextResponse
from app.providers.transcript.base import TranscriptProvider
from app.schemas.transcript import EpisodeMetadata, NormalizedTranscript
from app.services.job_queue import JobQueue


class FakeJobQueue(JobQueue):
    """Records enqueue calls instead of talking to Redis/arq."""

    def __init__(self) -> None:
        self.enqueued: list[tuple[uuid.UUID, uuid.UUID]] = []
        self.enqueued_processing: list[tuple[uuid.UUID, uuid.UUID]] = []
        self.enqueued_article_generation: list[tuple[uuid.UUID, uuid.UUID]] = []

    async def enqueue_transcript_ingestion(
        self, *, episode_id: uuid.UUID, job_id: uuid.UUID
    ) -> None:
        self.enqueued.append((episode_id, job_id))

    async def enqueue_transcript_processing(
        self, *, episode_id: uuid.UUID, job_id: uuid.UUID
    ) -> None:
        self.enqueued_processing.append((episode_id, job_id))

    async def enqueue_article_generation(
        self, *, episode_id: uuid.UUID, job_id: uuid.UUID
    ) -> None:
        self.enqueued_article_generation.append((episode_id, job_id))


class StubProvider(TranscriptProvider):
    """A TranscriptProvider double whose two methods each return a fixed
    value or raise a fixed exception, for tests that don't want a real
    HTTP call (see test_supadata_provider.py for those)."""

    def __init__(
        self,
        transcript: NormalizedTranscript | Exception,
        metadata: EpisodeMetadata | Exception | None = None,
    ):
        self._transcript = transcript
        self._metadata = metadata

    async def get_transcript(self, video_url: str) -> NormalizedTranscript:
        if isinstance(self._transcript, Exception):
            raise self._transcript
        return self._transcript

    async def get_metadata(self, video_url: str) -> EpisodeMetadata:
        if isinstance(self._metadata, Exception):
            raise self._metadata
        return self._metadata or EpisodeMetadata()


class FakeLLMProvider(LLMProvider):
    """An LLMProvider double that returns pre-programmed responses instead
    of calling a real SDK. app/ai/ node and graph tests use this
    exclusively -- no test in this suite makes a real call to any LLM
    provider (see the skip-by-default real smoke test for the one
    exception, gated behind an explicit env var).

    `structured_responses` is consumed in order, one per
    generate_structured() call (so a test can program a topic-analysis
    response, then a planning response, then per-section responses, in
    the sequence the graph will actually call them).
    """

    def __init__(
        self,
        structured_responses: list[BaseModel] | None = None,
        text_responses: list[str] | None = None,
    ):
        self._structured_responses = list(structured_responses or [])
        self._text_responses = list(text_responses or [])
        self.generate_calls: list[tuple[list[LLMMessage], str | None]] = []
        self.structured_calls: list[tuple[list[LLMMessage], str | None, type]] = []

    async def generate(
        self,
        *,
        messages: list[LLMMessage],
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> LLMTextResponse:
        self.generate_calls.append((messages, system))
        text = self._text_responses.pop(0) if self._text_responses else ""
        return LLMTextResponse(text=text, model="fake-model")

    async def generate_structured(
        self,
        *,
        messages: list[LLMMessage],
        response_model: type,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ):
        self.structured_calls.append((messages, system, response_model))
        if not self._structured_responses:
            raise AssertionError("FakeLLMProvider ran out of programmed structured responses")
        response = self._structured_responses.pop(0)
        if not isinstance(response, response_model):
            raise AssertionError(
                f"programmed response {type(response).__name__} doesn't match "
                f"requested schema {response_model.__name__}"
            )
        return response
