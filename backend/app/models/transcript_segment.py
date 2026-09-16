import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.transcript import Transcript


class TranscriptSegment(UUIDPrimaryKeyMixin, Base):
    """One provider-returned transcript segment — raw provider output,
    write-once, never edited in place (a reprocess replaces the rows), so
    there's no updated_at, just created_at.

    Deterministic cleaning (app/services/cleaning_service.py) never writes
    back here: it returns a transient in-memory `CleanedSegment` per
    segment, consumed directly by the chunker, so this table never mixes
    raw source data with derived processing state (PRODUCT_SPEC.md §18
    raw-vs-clean distinction — see ARCHITECTURE.md).
    """

    __tablename__ = "transcript_segments"
    __table_args__ = (
        # Enforces stable ordering per transcript and doubles as the index
        # needed for "fetch segments for a transcript in order".
        UniqueConstraint("transcript_id", "sequence_number", name="uq_transcript_segment_order"),
    )

    transcript_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("transcripts.id", ondelete="CASCADE"), index=True
    )
    sequence_number: Mapped[int] = mapped_column(Integer)

    text: Mapped[str] = mapped_column(Text)

    # Position within the source media — distinct from database timestamps,
    # never coerced into a datetime column (locked decision).
    start_ms: Mapped[int] = mapped_column(Integer)
    duration_ms: Mapped[int] = mapped_column(Integer)

    # Null when the provider doesn't return diarization; never invented.
    speaker: Mapped[str | None] = mapped_column(String(64))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    transcript: Mapped["Transcript"] = relationship(back_populates="segments")
