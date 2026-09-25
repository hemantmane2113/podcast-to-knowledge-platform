import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.article import Article


class ValidationResult(UUIDPrimaryKeyMixin, Base):
    """One validation run's outcome (Phase H) — one row per run, not
    replaced in place, so a revision's before/after validation history
    stays inspectable (a reviewer or the revision node itself can see
    which checks failed before the last regeneration).

    `checks` is the full list of individual CheckResult dicts
    ({name, passed, details} — see app/services/article_validation.py);
    `passed` is the overall AND of all of them, denormalized for a cheap
    "is this article ready" query without unpacking the JSON.
    """

    __tablename__ = "validation_results"

    article_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("articles.id", ondelete="CASCADE"), index=True
    )

    passed: Mapped[bool] = mapped_column(Boolean)
    checks: Mapped[list[dict]] = mapped_column(JSONB, default=list)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    article: Mapped["Article"] = relationship(back_populates="validation_results")
