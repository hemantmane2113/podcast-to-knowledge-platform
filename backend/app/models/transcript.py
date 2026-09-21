import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.chunk import Chunk
    from app.models.episode import Episode
    from app.models.topic import Topic
    from app.models.transcript_segment import TranscriptSegment


class Transcript(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """The transcript for an episode, as returned by the transcript provider.

    One episode has at most one Transcript (raw/normalized text; cleaning
    is a transient in-memory step over TranscriptSegment.text, not
    persisted at all — see app/services/cleaning_service.py and
    ARCHITECTURE.md).
    """

    __tablename__ = "transcripts"

    episode_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("episodes.id", ondelete="CASCADE"), unique=True
    )
    language: Mapped[str | None] = mapped_column(String(16))

    episode: Mapped["Episode"] = relationship(back_populates="transcript")
    segments: Mapped[list["TranscriptSegment"]] = relationship(
        back_populates="transcript",
        cascade="all, delete-orphan",
        order_by="TranscriptSegment.sequence_number",
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="transcript",
        cascade="all, delete-orphan",
        order_by="Chunk.sequence_number",
    )
    topics: Mapped[list["Topic"]] = relationship(
        back_populates="transcript",
        cascade="all, delete-orphan",
        order_by="Topic.sequence_number",
    )
