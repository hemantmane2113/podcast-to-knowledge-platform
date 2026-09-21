import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.article_plan import ArticlePlan
    from app.models.article_section import ArticleSection
    from app.models.episode import Episode
    from app.models.validation_result import ValidationResult


class Article(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """The assembled article for an episode (Phase G). Pipeline lifecycle
    (generating/verifying/revising/ready for review/published) is tracked
    on Episode.status (PRODUCT_SPEC.md §17), not duplicated here — this
    row's own fields are just its content plus the bounded revision
    counter the graph uses to stop retrying (app/ai/graph.py).

    One episode has at most one Article; a regeneration replaces it
    (ArticleRepository.replace) the same way Chunk/Topic/ArticlePlan do.
    """

    __tablename__ = "articles"

    episode_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("episodes.id", ondelete="CASCADE"), unique=True
    )
    article_plan_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("article_plans.id", ondelete="CASCADE")
    )

    title: Mapped[str] = mapped_column(Text)
    revision_count: Mapped[int] = mapped_column(Integer, default=0)

    episode: Mapped["Episode"] = relationship(back_populates="article")
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
