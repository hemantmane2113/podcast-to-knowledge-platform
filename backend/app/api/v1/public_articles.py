import uuid

from fastapi import APIRouter

from app.api.deps import ArticleServiceDep
from app.schemas.article import PublicArticleListResponse, PublicArticleResponse, PublicArticleSummaryResponse

# A separate, top-level "/articles" namespace (not under "/episodes" like
# app/api/v1/articles.py) -- deliberately: this is the public blog surface,
# distinct from the episode-scoped, reviewer-facing article endpoints.
# Always PUBLISHED-only; see ArticleService.get_published_article /
# list_published_articles.
router = APIRouter(prefix="/articles", tags=["public"])


@router.get("", response_model=PublicArticleListResponse)
async def list_articles(service: ArticleServiceDep) -> PublicArticleListResponse:
    """The public blog listing -- PUBLISHED articles only, newest first."""
    summaries = await service.list_published_articles()
    return PublicArticleListResponse(
        articles=[
            PublicArticleSummaryResponse(
                episode_id=row.episode_id, title=row.title, published_at=row.article_published_at
            )
            for row in summaries
        ]
    )


@router.get("/{episode_id}", response_model=PublicArticleResponse)
async def get_article(episode_id: uuid.UUID, service: ArticleServiceDep) -> PublicArticleResponse:
    """A single published article. Raises the same ArticleNotFoundError
    (404) for a nonexistent episode, an episode with no article, and an
    episode whose article exists but isn't published yet -- never
    distinguishable to a public caller."""
    episode, article, chunks = await service.get_published_article(episode_id)
    chunks_by_id = {c.id: c for c in chunks}
    return PublicArticleResponse.from_models(episode, article, chunks_by_id)
