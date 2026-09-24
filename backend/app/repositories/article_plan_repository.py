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

    async def get_by_episode_id(self, episode_id: uuid.UUID, *, is_draft: bool = False) -> ArticlePlan | None:
        """`is_draft` defaults to False (the live plan) -- every existing
        call site that doesn't explicitly ask for the draft keeps
        resolving only the live row, unchanged (Live/Draft Article
        Workflow). The partial unique index guarantees at most one row
        can ever match a given (episode_id, is_draft) pair, so
        scalar_one_or_none() stays safe."""
        result = await self._session.execute(
            select(ArticlePlan).where(
                ArticlePlan.episode_id == episode_id, ArticlePlan.is_draft == is_draft
            )
        )
        return result.scalar_one_or_none()

    async def delete_by_episode_id(self, episode_id: uuid.UUID, *, is_draft: bool = False) -> None:
        """A Core-level DELETE -- deliberately not `session.get()` +
        `session.delete()`, so nothing needs to be loaded into the
        identity map first (same idiom as the delete half of replace()
        below, factored out so a caller that only wants to invalidate
        an episode's plan -- without immediately replacing it, e.g.
        ArticleService.request_regeneration -- doesn't have to duplicate
        the statement). The DB's own ON DELETE CASCADE
        (Article.article_plan_id, ArticleSection.article_id,
        ValidationResult.article_id) removes the Article/ArticleSections/
        ValidationResults generated from this plan in the same operation;
        Topic and Chunk are untouched -- neither has a FK to ArticlePlan.

        `is_draft` scopes the delete to exactly one row (Live/Draft
        Article Workflow) -- defaults to False so an unscoped call still
        means "the live plan", matching this method's pre-draft behavior,
        but ArticleService.request_regeneration must pass is_draft=True
        explicitly so it only ever clears a stale DRAFT, never the live
        plan.

        Because this bypasses the ORM's own cascade machinery, any
        Article/ArticlePlan/ArticleSection/ValidationResult objects
        already loaded into the CALLING session's identity map are not
        automatically updated to reflect the delete -- the caller is
        responsible for expiring the session afterward if that matters
        (see ArticleService.request_regeneration).
        """
        await self._session.execute(
            delete(ArticlePlan).where(ArticlePlan.episode_id == episode_id, ArticlePlan.is_draft == is_draft)
        )

    async def replace(
        self,
        *,
        episode_id: uuid.UUID,
        title: str,
        introduction_summary: str,
        conclusion_summary: str,
        sections: list[PlannedSectionData],
        is_draft: bool = False,
    ) -> ArticlePlan:
        """Deleting the existing plan for this (episode_id, is_draft) pair
        (if any) cascades to the Article built from it
        (Article.article_plan_id has ondelete='CASCADE') -- deliberate: a
        plan regeneration invalidates whatever article was generated from
        the old plan, and the pipeline regenerates the article from the
        new plan right after this anyway. `is_draft` scopes both the
        delete and the new row (Live/Draft Article Workflow) -- the
        pipeline (app/ai/nodes/planning.py) always calls this with
        is_draft=True, so a regeneration's delete-then-insert can never
        touch the live plan.
        """
        await self.delete_by_episode_id(episode_id, is_draft=is_draft)

        plan = ArticlePlan(
            id=uuid.uuid4(),
            episode_id=episode_id,
            is_draft=is_draft,
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
