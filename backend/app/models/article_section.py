import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ARRAY, DateTime, ForeignKey, Integer, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.article import Article


class ArticleSection(UUIDPrimaryKeyMixin, Base):
    """One generated section of an Article (Phase F). Evidence grounding
    (Phase D) lives directly on the section: supporting_chunk_ids is the
    literal list of Chunk.id this section's content was generated from,
    the same array-not-join-table pattern as Chunk.source_segment_ids and
    Topic.chunk_ids -- this is what makes
    "article section -> supporting chunks -> transcript segments ->
    timestamps" traceable without a join, per the Phase D requirement.
    """

    __tablename__ = "article_sections"
    __table_args__ = (
        UniqueConstraint("article_id", "sequence_number", name="uq_article_section_order"),
    )

    article_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("articles.id", ondelete="CASCADE"), index=True
    )

    sequence_number: Mapped[int] = mapped_column(Integer)
    heading: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)

    supporting_chunk_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), default=list
    )
    supporting_topic_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), default=list
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    article: Mapped["Article"] = relationship(back_populates="sections")
