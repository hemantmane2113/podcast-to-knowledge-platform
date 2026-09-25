import uuid

from fastapi import APIRouter, status

from app.api.deps import ArticleServiceDep
from app.schemas.article import ArticleGenerationStatusResponse, ArticleResponse, GenerateArticleResponse
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


@router.get("/{episode_id}/generation-status", response_model=ArticleGenerationStatusResponse)
async def get_generation_status(
    episode_id: uuid.UUID, service: ArticleServiceDep
) -> ArticleGenerationStatusResponse:
    """The reviewer UI's poll target while a draft is generating --
    Episode.status Decoupling means Episode.status itself no longer
    carries ANALYZING/PLANNING/GENERATING/VERIFYING/REVISING, so this is
    the smallest replacement: the latest ARTICLE_GENERATION
    ProcessingJob's own status (PENDING/RUNNING/COMPLETED/FAILED) and
    error message, nothing more granular than that.
    """
    job = await service.get_latest_generation_job(episode_id)
    return ArticleGenerationStatusResponse.from_model(job)


@router.get("/{episode_id}/article", response_model=ArticleResponse)
async def get_article(
    episode_id: uuid.UUID, service: ArticleServiceDep, draft: bool = False
) -> ArticleResponse:
    """`draft=true` retrieves the episode's in-progress/awaiting-review
    DRAFT article instead of the live one -- how a reviewer looks at a
    draft (Live/Draft Article Workflow) without this endpoint's default,
    public-facing behavior changing at all: with no query param (or
    draft=false), this still returns exactly the live article it always
    has.
    """
    episode, article, chunks = await service.get_article_with_chunks(episode_id, is_draft=draft)
    chunks_by_id = {c.id: c for c in chunks}
    return ArticleResponse.from_models(article, episode, chunks_by_id)


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


@router.post("/{episode_id}/promote-draft", response_model=EpisodeResponse)
async def promote_draft(episode_id: uuid.UUID, service: ArticleServiceDep) -> EpisodeResponse:
    """The Live/Draft Article Workflow's own explicit, deliberate final
    step: makes the episode's current draft article (as retrieved via GET
    /episodes/{id}/article?draft=true) the new live one, and publishes
    the episode. See ArticleService.promote_draft for the full rejection
    rules (no draft, draft generation still running or failed, missing/
    failed validation) and the transactional swap itself.
    """
    episode = await service.promote_draft(episode_id)
    return EpisodeResponse.model_validate(episode)
