import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.episode import Episode


class ArticlePlan(UUIDPrimaryKeyMixin, Base):
    """The explicit plan produced before article generation (Phase E) —
    persisted on its own so it can be inspected independently of the
    generated prose, per the Phase E requirement.

    An episode has at most one LIVE (is_draft=False) ArticlePlan and at
    most one DRAFT (is_draft=True) one at a time -- enforced by the
    partial unique index below, not a plain UNIQUE(episode_id) (Live/Draft
    Article Workflow). A regeneration creates/replaces the DRAFT via
    ArticlePlanRepository.replace(..., is_draft=True); the live plan is
    never touched by regeneration, only by ArticleService.promote_draft.
    """

    __tablename__ = "article_plans"
    __table_args__ = (
        Index(
            "uq_article_plans_episode_id_live",
            "episode_id",
            unique=True,
            postgresql_where=text("NOT is_draft"),
        ),
    )

    episode_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("episodes.id", ondelete="CASCADE")
    )
    is_draft: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))

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

    episode: Mapped["Episode"] = relationship(back_populates="article_plans")
