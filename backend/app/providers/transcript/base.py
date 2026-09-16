from abc import ABC, abstractmethod

from app.schemas.transcript import EpisodeMetadata, NormalizedTranscript


class TranscriptProvider(ABC):
    """Isolates the rest of the application from any specific transcript
    source (PRODUCT_SPEC.md §11, §93). Implementations translate their
    provider's request/response format into the normalized schemas in
    app/schemas/transcript.py and raise app.core.exceptions subclasses of
    TranscriptProviderError on failure — never a provider-specific
    exception type or raw HTTP error.
    """

    @abstractmethod
    async def get_metadata(self, video_url: str) -> EpisodeMetadata: ...

    @abstractmethod
    async def get_transcript(self, video_url: str) -> NormalizedTranscript: ...
