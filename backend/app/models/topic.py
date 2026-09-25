import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ARRAY, DateTime, ForeignKey, Integer, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.transcript import Transcript


class Topic(UUIDPrimaryKeyMixin, Base):
    """A semantic/topic grouping over one or more canonical Chunks (Phase
    C — see app/ai/nodes/topic_analysis.py). Deliberately a separate
    concept from Chunk: chunks are source-oriented size-constrained units,
    topics are meaning-oriented groupings that several chunks can belong
    to — a topic almost never equals exactly one chunk in practice.

    Write-once per generation; like Chunk, the full set for a transcript
    is regenerated (deleted + reinserted) each time topic analysis runs
    rather than diffed — see TopicRepository.replace_all.
    """

    __tablename__ = "topics"
    __table_args__ = (
        UniqueConstraint("transcript_id", "sequence_number", name="uq_topic_order"),
    )

    transcript_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("transcripts.id", ondelete="CASCADE"), index=True
    )
    # Denormalized from transcript.episode_id, same reasoning as Chunk.episode_id.
    episode_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("episodes.id", ondelete="CASCADE"), index=True
    )

    sequence_number: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)

    # Every Chunk.id this topic was drawn from, in order. A plain array,
    # same reasoning as Chunk.source_segment_ids: a chunk can legitimately
    # contribute to more than one topic near a boundary, so a strict range
    # would not model this correctly.
    chunk_ids: Mapped[list[uuid.UUID]] = mapped_column(ARRAY(UUID(as_uuid=True)))

    # Structured, LLM-produced claims for this topic -- list of
    # {text, speaker, claim_type}, see app/ai/schemas.py::TopicClaim.
    # JSONB (not a child table): claims are only ever read/written as a
    # whole alongside their topic, never queried independently, so a
    # normalized table would be pure overhead for V1.
    key_claims: Mapped[list[dict]] = mapped_column(JSONB, default=list)
    subtopics: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    transcript: Mapped["Transcript"] = relationship(back_populates="topics")
