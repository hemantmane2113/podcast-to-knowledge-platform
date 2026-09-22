import uuid
from dataclasses import dataclass, field

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.article import Article
from app.models.article_section import ArticleSection


@dataclass
class ArticleSectionCandidate:
    """A generated section's persisted shape -- see
    app/ai/nodes/section_generation.py, which resolves the model's topic
    sequence_number references into real Topic/Chunk UUIDs before
    constructing one of these (same "small ints in the prompt, real UUIDs
    at the persistence boundary" reasoning as TopicCandidate)."""

    sequence_number: int
    heading: str
    content: str
    supporting_chunk_ids: list[uuid.UUID] = field(default_factory=list)
    supporting_topic_ids: list[uuid.UUID] = field(default_factory=list)


class ArticleRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_episode_id(self, episode_id: uuid.UUID) -> Article | None:
        result = await self._session.execute(
            select(Article)
            .where(Article.episode_id == episode_id)
            .options(selectinload(Article.sections), selectinload(Article.validation_results))
        )
        return result.scalar_one_or_none()

    async def replace(
        self,
        *,
        episode_id: uuid.UUID,
        article_plan_id: uuid.UUID,
        title: str,
        revision_count: int,
        sections: list[ArticleSectionCandidate],
    ) -> Article:
        """Delete-then-insert, same idempotency shape as Chunk/Topic --
        deleting the existing Article cascades to its sections and
        validation_results (both FK ondelete='CASCADE'). `revision_count`
        is supplied by the caller (app/ai/graph.py's state), not computed
        here, since the repository has no notion of "this is a revision
        vs. the first generation".
        """
        await self._session.execute(delete(Article).where(Article.episode_id == episode_id))

        article = Article(
            id=uuid.uuid4(),
            episode_id=episode_id,
            article_plan_id=article_plan_id,
            title=title,
            revision_count=revision_count,
        )
        article.sections = [
            ArticleSection(
                id=uuid.uuid4(),
                sequence_number=s.sequence_number,
                heading=s.heading,
                content=s.content,
                supporting_chunk_ids=s.supporting_chunk_ids,
                supporting_topic_ids=s.supporting_topic_ids,
            )
            for s in sections
        ]
        self._session.add(article)
        return article

    async def get_or_create(
        self, *, episode_id: uuid.UUID, article_plan_id: uuid.UUID, title: str, revision_count: int = 0
    ) -> Article:
        """Resumability's entry point for the Article row itself (see
        app/ai/nodes/section_generation.py): returns the existing Article
        as-is -- sections and all -- if it already belongs to
        `article_plan_id` (the plan THIS run is using, whether freshly
        planned or reused), so a resumed job never loses already-persisted
        sections. Only creates a fresh, empty Article when there isn't one
        yet, or when the existing one belongs to a different plan -- which
        should never happen given ArticlePlanRepository.replace's cascade
        delete on regeneration, but is never trusted blindly (see the
        Batch 2B identity requirement: never silently reuse an article
        that doesn't actually belong to the current plan).
        """
        existing = await self.get_by_episode_id(episode_id)
        if existing is not None and existing.article_plan_id == article_plan_id:
            return existing
        if existing is not None:
            await self._session.execute(delete(Article).where(Article.episode_id == episode_id))

        article = Article(
            id=uuid.uuid4(),
            episode_id=episode_id,
            article_plan_id=article_plan_id,
            title=title,
            revision_count=revision_count,
        )
        # Explicitly loaded (even though empty), same as replace() below --
        # otherwise this relationship is simply unset on the new object,
        # and accessing it after this method returns (once the object is
        # persistent) triggers an async-incompatible lazy-load instead of
        # returning the empty list a brand-new Article actually has.
        article.sections = []
        self._session.add(article)
        return article

    def upsert_section(self, article: Article, candidate: ArticleSectionCandidate) -> ArticleSection:
        """Inserts one section, or replaces an existing (necessarily
        invalid/incomplete -- see article_validation.is_section_content_valid)
        row for the same sequence_number in place -- the incremental
        counterpart to `replace()`'s all-at-once write. The caller commits
        immediately after this (see section_generation_node), so a crash
        after section N leaves sections 1..N durably persisted and reusable
        by the next attempt, rather than requiring the whole article to be
        regenerated from scratch.
        """
        existing = next((s for s in article.sections if s.sequence_number == candidate.sequence_number), None)
        if existing is not None:
            existing.heading = candidate.heading
            existing.content = candidate.content
            existing.supporting_chunk_ids = candidate.supporting_chunk_ids
            existing.supporting_topic_ids = candidate.supporting_topic_ids
            return existing

        section = ArticleSection(
            id=uuid.uuid4(),
            sequence_number=candidate.sequence_number,
            heading=candidate.heading,
            content=candidate.content,
            supporting_chunk_ids=candidate.supporting_chunk_ids,
            supporting_topic_ids=candidate.supporting_topic_ids,
        )
        article.sections.append(section)
        self._session.add(section)
        return section
