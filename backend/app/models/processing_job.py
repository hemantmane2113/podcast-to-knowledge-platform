import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.episode import Episode


class JobType(str, enum.Enum):
    """Job types that actually exist. Later phases (article generation,
    verification, ...) add their own values when they exist — not
    speculated here.
    """

    TRANSCRIPT_INGESTION = "TRANSCRIPT_INGESTION"
    # Cleaning + chunking together (Phase 3A) -- one job, since chunking
    # always immediately consumes cleaning's output; see
    # app/services/chunking_service.py and ARCHITECTURE.md.
    TRANSCRIPT_PROCESSING = "TRANSCRIPT_PROCESSING"
    # The whole AI pipeline (topic analysis -> planning -> section
    # generation -> assembly -> validation -> bounded revision) as one
    # LangGraph-orchestrated job -- see app/ai/graph.py. Deliberately NOT
    # auto-chained after TRANSCRIPT_PROCESSING (unlike ingestion ->
    # processing): it makes paid LLM calls, so it's triggered explicitly
    # via POST /episodes/{id}/generate-article.
    ARTICLE_GENERATION = "ARTICLE_GENERATION"


class JobStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ProcessingJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Minimal job tracking for async ingestion — not a distributed job
    framework. One row per enqueued unit of work; the actual queueing
    mechanism is Redis + arq (app/worker/), this table is purely the
    admin-visible status record (PRODUCT_SPEC.md §17, §49).
    """

    __tablename__ = "processing_jobs"

    episode_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("episodes.id", ondelete="CASCADE"), index=True
    )
    job_type: Mapped[JobType] = mapped_column(Enum(JobType, name="job_type", native_enum=True))
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status", native_enum=True),
        default=JobStatus.PENDING,
        index=True,
    )

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # A safe, human-readable message only — never the provider's raw
    # response or any credential (see app/worker/tasks.py).
    error_message: Mapped[str | None] = mapped_column(Text)

    episode: Mapped["Episode"] = relationship(back_populates="processing_jobs")
