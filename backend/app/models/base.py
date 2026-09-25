import uuid
from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class UUIDPrimaryKeyMixin:
    """UUID primary key, generated application-side.

    All entities use UUIDs rather than integer PKs (locked decision, see
    ARCHITECTURE.md). Natural/external identifiers (e.g. Episode.youtube_video_id)
    are stored as separate unique columns, never as the primary key itself.
    """

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )


class TimestampMixin:
    """created_at/updated_at, always timezone-aware UTC.

    Distinct from any transcript-timing fields (start_ms/duration_ms), which
    represent position within the source media, not wall-clock database
    events, and must never be coerced into datetime columns.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
