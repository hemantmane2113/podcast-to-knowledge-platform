import uuid

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.validation_result import ValidationResult


class ValidationResultRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    def create(
        self, *, article_id: uuid.UUID, passed: bool, checks: list[dict]
    ) -> ValidationResult:
        """Not delete-then-replace like the other repositories: each
        validation run (including ones between revision attempts) is kept
        as its own row so the before/after history across a revision
        loop stays inspectable (see ValidationResult's docstring)."""
        result = ValidationResult(
            id=uuid.uuid4(), article_id=article_id, passed=passed, checks=checks
        )
        self._session.add(result)
        return result

    async def get_latest_by_article_id(self, article_id: uuid.UUID) -> ValidationResult | None:
        result = await self._session.execute(
            select(ValidationResult)
            .where(ValidationResult.article_id == article_id)
            .order_by(desc(ValidationResult.created_at))
            .limit(1)
        )
        return result.scalar_one_or_none()
