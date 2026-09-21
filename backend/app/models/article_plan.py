import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.episode import Episode


class ArticlePlan(UUIDPrimaryKeyMixin, Base):
    """The explicit plan produced before article generation (Phase E) —
    persisted on its own so it can be inspected independently of the
    generated prose, per the Phase E requirement.

    One episode has at most one ArticlePlan (like Transcript); a
    regeneration replaces it via ArticlePlanRepository.replace, same
    idempotency shape as Chunk/Topic.
    """

    __tablename__ = "article_plans"

    episode_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("episodes.id", ondelete="CASCADE"), unique=True
    )

    title: Mapped[str] = mapped_column(Text)
    introduction_summary: Mapped[str] = mapped_column(Text)
    conclusion_summary: Mapped[str] = mapped_column(Text)

    # Ordered list of planned sections -- see app/ai/schemas.py::PlannedSection
    # for the shape: {sequence_number, heading, key_ideas, supporting_topic_ids,
    # supporting_chunk_ids, viewpoints, attribution_notes}. JSONB rather than
    # a child table for the same reason as Topic.key_claims: only ever read
    # as a whole with its plan, never queried independently in V1.
    sections: Mapped[list[dict]] = mapped_column(JSONB, default=list)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    episode: Mapped["Episode"] = relationship(back_populates="article_plan")
