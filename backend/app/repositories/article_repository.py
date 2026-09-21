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
