import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ARRAY, DateTime, ForeignKey, Integer, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.transcript import Transcript


class Chunk(UUIDPrimaryKeyMixin, Base):
    """A semantically-grouped span of a transcript, built from one or more
    (cleaned) TranscriptSegments — see app/services/chunking_service.py.

    Write-once per generation, but the *set* of chunks for a transcript is
    fully regenerated (old rows deleted, new ones inserted, one
    transaction) each time chunking runs — see ChunkRepository.replace_all
    for the idempotency strategy documented in ARCHITECTURE.md.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("transcript_id", "sequence_number", name="uq_chunk_order"),
    )

    transcript_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("transcripts.id", ondelete="CASCADE"), index=True
    )
    # Denormalized from transcript.episode_id — avoids a join for the
    # extremely common "chunks for this episode" query, per the explicit
    # field list for this model.
    episode_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("episodes.id", ondelete="CASCADE"), index=True
    )

    sequence_number: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)

    # Segment-level boundaries (locked decision — see chunking_service.py):
    # start_ms is the first source segment's start_ms, end_ms is the last
    # source segment's start_ms + duration_ms. Never interpolated to a
    # sub-segment offset, even when a single long segment's text was split
    # across two chunks (see source_segment_ids below).
    start_ms: Mapped[int] = mapped_column(Integer)
    end_ms: Mapped[int] = mapped_column(Integer)

    # Every TranscriptSegment.id this chunk's text was drawn from, in
    # order. A plain array rather than a join table: chunks are built from
    # contiguous ranges of segments, except that a single oversized segment
    # split across two chunks (§3A.4) legitimately appears in both chunks'
    # lists — an array handles that without extra schema, a strict range
    # (first_id, last_id) would not.
    source_segment_ids: Mapped[list[uuid.UUID]] = mapped_column(ARRAY(UUID(as_uuid=True)))

    # Approximate (see chunking_service.py::estimate_tokens) — no concrete
    # LLM/tokenizer has been chosen yet (Phase 4), so this is a
    # characters-per-token heuristic, not an exact count from a real
    # tokenizer. Good enough to drive chunk sizing consistently; do not
    # treat it as authoritative token usage for billing/context-window math.
    token_count: Mapped[int] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    transcript: Mapped["Transcript"] = relationship(back_populates="chunks")
