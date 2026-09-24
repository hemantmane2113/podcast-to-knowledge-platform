import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, ForeignKey, Index, Integer, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.article_plan import ArticlePlan
    from app.models.article_section import ArticleSection
    from app.models.episode import Episode
    from app.models.validation_result import ValidationResult


class Article(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """The assembled article for an episode (Phase G). Article-generation
    progress (analyzing/planning/generating/verifying/revising) is tracked
    on ProcessingJob.status, and publication is tracked on Episode.status
    (PUBLISHED) -- neither is duplicated here; this row's own fields are
    just its content plus the bounded revision counter the graph uses to
    stop retrying (app/ai/graph.py).

    An episode has at most one LIVE (is_draft=False) Article and at most
    one DRAFT (is_draft=True) one at a time -- enforced by the partial
    unique index below, not a plain UNIQUE(episode_id) (Live/Draft Article
    Workflow). A regeneration creates/replaces the DRAFT
    (ArticleRepository.get_or_create(..., is_draft=True)); the live
    article is never touched by regeneration, only by
    ArticleService.promote_draft.
    """

    __tablename__ = "articles"
    __table_args__ = (
        Index(
            "uq_articles_episode_id_live",
            "episode_id",
            unique=True,
            postgresql_where=text("NOT is_draft"),
        ),
    )

    episode_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("episodes.id", ondelete="CASCADE")
    )
    article_plan_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("article_plans.id", ondelete="CASCADE")
    )
    is_draft: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))

    title: Mapped[str] = mapped_column(Text)
    revision_count: Mapped[int] = mapped_column(Integer, default=0)

    episode: Mapped["Episode"] = relationship(back_populates="articles")
    article_plan: Mapped["ArticlePlan"] = relationship()
    sections: Mapped[list["ArticleSection"]] = relationship(
        back_populates="article",
        cascade="all, delete-orphan",
        order_by="ArticleSection.sequence_number",
    )
    validation_results: Mapped[list["ValidationResult"]] = relationship(
        back_populates="article",
        cascade="all, delete-orphan",
        order_by="ValidationResult.created_at",
    )
