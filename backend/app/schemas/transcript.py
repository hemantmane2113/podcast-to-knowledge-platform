"""The internal, provider-agnostic transcript representation.

Nothing outside app/providers/transcript/ should ever touch a provider's
raw response shape directly — every TranscriptProvider implementation
normalizes into these models (PRODUCT_SPEC.md §13).
"""

from pydantic import BaseModel, Field


class NormalizedSegment(BaseModel):
    text: str
    start_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    speaker: str | None = None


class NormalizedTranscript(BaseModel):
    language: str | None = None
    segments: list[NormalizedSegment]


class EpisodeMetadata(BaseModel):
    """All fields nullable: only what the provider actually returns is
    populated here — nothing is fabricated when a field is unavailable
    (PRODUCT_SPEC.md §88)."""

    title: str | None = None
    description: str | None = None
    channel_name: str | None = None
    channel_id: str | None = None
    thumbnail_url: str | None = None
    duration_seconds: int | None = None
    published_at: str | None = None  # ISO 8601 string as returned by the provider
    language: str | None = None
