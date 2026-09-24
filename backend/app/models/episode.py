import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.article import Article
    from app.models.article_plan import ArticlePlan


class ProcessingStatus(str, enum.Enum):
    """Full pipeline status vocabulary (PRODUCT_SPEC.md §17).

    Phase 2 (transcript ingestion) only ever sets INGESTING,
    TRANSCRIPT_FETCHED, or FAILED. The remaining values belong to later
    phases (cleaning/chunking/analysis/generation/verification/publishing)
    and are declared now only so the enum type doesn't need an ALTER TYPE
    migration every time a later phase starts using its next value.
    """

    INGESTING = "INGESTING"
    TRANSCRIPT_FETCHED = "TRANSCRIPT_FETCHED"
    CLEANING = "CLEANING"
    CHUNKING = "CHUNKING"
    ANALYZING = "ANALYZING"
    PLANNING = "PLANNING"
    GENERATING = "GENERATING"
    VERIFYING = "VERIFYING"
    REVISING = "REVISING"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"


class Episode(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "episodes"

    # External YouTube identifier — the idempotency key (PRODUCT_SPEC.md
    # §51). Deliberately NOT the primary key: the spec requires the video ID
    # be usable as a "unique logical identifier" for lookups, not that it be
    # the database identity, and app.models.base's UUID-everywhere decision
    # would otherwise be broken for this one entity.
    youtube_video_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    youtube_url: Mapped[str] = mapped_column(Text)

    # Metadata is populated by the ingestion worker after a successful
    # Supadata call; all of it is nullable until then, and stays nullable
    # afterward for any field Supadata doesn't return (never fabricated).
    title: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    channel_name: Mapped[str | None] = mapped_column(String(255))
    channel_id: Mapped[str | None] = mapped_column(String(64))
    thumbnail_url: Mapped[str | None] = mapped_column(Text)
    duration_seconds: Mapped[int | None] = mapped_column(Integer)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    language: Mapped[str | None] = mapped_column(String(16))

    status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus, name="processing_status", native_enum=True),
        default=ProcessingStatus.INGESTING,
        index=True,
    )
    last_error: Mapped[str | None] = mapped_column(Text)

    # When OUR generated article was explicitly published (status ->
    # PUBLISHED) -- deliberately a separate column from `published_at`
    # above, which is the source YouTube video's own publish date
    # (populated by ingestion from Supadata metadata, often long before
    # any article exists). Null until publish_article actually runs; set
    # exactly once (see EpisodeRepository.mark_published) -- republishing
    # an already-PUBLISHED episode never bumps it forward.
    article_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    transcript: Mapped["Transcript | None"] = relationship(
        back_populates="episode", uselist=False, cascade="all, delete-orphan"
    )
    processing_jobs: Mapped[list["ProcessingJob"]] = relationship(
        back_populates="episode", cascade="all, delete-orphan"
    )
    article_plan: Mapped["ArticlePlan | None"] = relationship(
        back_populates="episode", uselist=False, cascade="all, delete-orphan"
    )
    article: Mapped["Article | None"] = relationship(
        back_populates="episode", uselist=False, cascade="all, delete-orphan"
    )
