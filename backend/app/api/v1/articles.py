import uuid

from fastapi import APIRouter, status

from app.api.deps import ArticleServiceDep
from app.schemas.article import ArticleResponse, GenerateArticleResponse
from app.schemas.episode import EpisodeResponse

router = APIRouter(prefix="/episodes", tags=["articles"])


@router.post(
    "/{episode_id}/generate-article",
    response_model=GenerateArticleResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def generate_article(episode_id: uuid.UUID, service: ArticleServiceDep) -> GenerateArticleResponse:
    """Enqueues the Phase C-H AI pipeline (app/ai/graph.py) and returns
    immediately -- same async-everything shape as POST /episodes, and
    same reason: this can take a while and makes paid LLM calls, so it's
    never run inline on the request. Poll GET /episodes/{id} for
    Episode.status and GET /episodes/{id}/article once it's READY_FOR_REVIEW.
    """
    job = await service.request_generation(episode_id)
    return GenerateArticleResponse(episode_id=episode_id, job_id=job.id)


@router.post(
    "/{episode_id}/regenerate-article",
    response_model=GenerateArticleResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def regenerate_article(episode_id: uuid.UUID, service: ArticleServiceDep) -> GenerateArticleResponse:
    """Explicit regeneration -- unlike generate-article, an existing
    completed article is never a reason to skip enqueueing a fresh run
    (see ArticleService.request_regeneration for the exact lifecycle:
    the episode's ArticlePlan/Article/ArticleSections/ValidationResults
    are replaced by the new pipeline run; Topics and Chunks are kept).
    An already-active job is still respected, exactly as generate-article
    does. Same async-everything shape: enqueues and returns immediately.
    """
    job = await service.request_regeneration(episode_id)
    return GenerateArticleResponse(episode_id=episode_id, job_id=job.id)


@router.get("/{episode_id}/article", response_model=ArticleResponse)
async def get_article(episode_id: uuid.UUID, service: ArticleServiceDep) -> ArticleResponse:
    episode, article, chunks = await service.get_article_with_chunks(episode_id)
    chunks_by_id = {c.id: c for c in chunks}
    return ArticleResponse.from_models(article, episode.status, chunks_by_id)


@router.post("/{episode_id}/publish", response_model=EpisodeResponse)
async def publish_article(episode_id: uuid.UUID, service: ArticleServiceDep) -> EpisodeResponse:
    """Explicit publish -- the deliberate final step of generate ->
    validate -> human review -> explicit publish (see
    ArticleService.publish_article for the exact lifecycle: requires an
    existing article with a passing latest validation result, and never
    modifies ArticlePlan/Article/ArticleSections/Topics/Chunks). Never
    triggered automatically after generation. Idempotent: publishing an
    already-published episode returns its current state unchanged.
    """
    episode = await service.publish_article(episode_id)
    return EpisodeResponse.model_validate(episode)
