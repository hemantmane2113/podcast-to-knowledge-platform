import uuid

from app.providers.transcript.base import TranscriptProvider
from app.schemas.transcript import EpisodeMetadata, NormalizedTranscript
from app.services.job_queue import JobQueue


class FakeJobQueue(JobQueue):
    """Records enqueue calls instead of talking to Redis/arq."""

    def __init__(self) -> None:
        self.enqueued: list[tuple[uuid.UUID, uuid.UUID]] = []

    async def enqueue_transcript_ingestion(
        self, *, episode_id: uuid.UUID, job_id: uuid.UUID
    ) -> None:
        self.enqueued.append((episode_id, job_id))


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
