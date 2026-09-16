import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.episode import ProcessingStatus


class EpisodeCreateRequest(BaseModel):
    youtube_url: str


class EpisodeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    youtube_video_id: str
    youtube_url: str

    title: str | None
    description: str | None
    channel_name: str | None
    channel_id: str | None
    thumbnail_url: str | None
    duration_seconds: int | None
    published_at: datetime | None
    language: str | None

    status: ProcessingStatus
    last_error: str | None

    created_at: datetime
    updated_at: datetime


class TranscriptSegmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    sequence_number: int
    text: str
    start_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    speaker: str | None


class TranscriptResponse(BaseModel):
    episode_id: uuid.UUID
    language: str | None
    segments: list[TranscriptSegmentResponse]
