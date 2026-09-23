import uuid
from dataclasses import dataclass, field

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.article_plan import ArticlePlan


@dataclass
class PlannedSectionData:
    """One planned section's persisted shape (stored inside
    ArticlePlan.sections as JSONB) -- topic/chunk references are already
    resolved to real UUIDs (as strings, for JSON-serializability) by the
    time this is built."""

    sequence_number: int
    heading: str
    key_ideas: list[str]
    supporting_topic_ids: list[str]
    supporting_chunk_ids: list[str]
    viewpoints: list[str] = field(default_factory=list)
    attribution_notes: list[str] = field(default_factory=list)
    narrative_purpose: str = ""
    transition_from_previous: str = ""


class ArticlePlanRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_episode_id(self, episode_id: uuid.UUID) -> ArticlePlan | None:
        result = await self._session.execute(
            select(ArticlePlan).where(ArticlePlan.episode_id == episode_id)
        )
        return result.scalar_one_or_none()

    async def replace(
        self,
        *,
        episode_id: uuid.UUID,
        title: str,
        introduction_summary: str,
        conclusion_summary: str,
        sections: list[PlannedSectionData],
    ) -> ArticlePlan:
        """Deleting the existing plan (if any) cascades to the Article
        built from it (Article.article_plan_id has ondelete='CASCADE') --
        deliberate: a plan regeneration invalidates whatever article was
        generated from the old plan, and the pipeline regenerates the
        article from the new plan right after this anyway.
        """
        await self._session.execute(delete(ArticlePlan).where(ArticlePlan.episode_id == episode_id))

        plan = ArticlePlan(
            id=uuid.uuid4(),
            episode_id=episode_id,
            title=title,
            introduction_summary=introduction_summary,
            conclusion_summary=conclusion_summary,
            sections=[
                {
                    "sequence_number": s.sequence_number,
                    "heading": s.heading,
                    "key_ideas": s.key_ideas,
                    "supporting_topic_ids": s.supporting_topic_ids,
                    "supporting_chunk_ids": s.supporting_chunk_ids,
                    "viewpoints": s.viewpoints,
                    "attribution_notes": s.attribution_notes,
                    "narrative_purpose": s.narrative_purpose,
                    "transition_from_previous": s.transition_from_previous,
                }
                for s in sections
            ],
        )
        self._session.add(plan)
        return plan
