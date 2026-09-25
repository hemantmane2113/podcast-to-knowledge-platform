import uuid
from datetime import datetime

from pydantic import BaseModel

from app.models.article import Article
from app.models.chunk import Chunk
from app.models.episode import Episode, ProcessingStatus
from app.models.processing_job import JobStatus, ProcessingJob
from app.models.validation_result import ValidationResult

_SOURCE_PREVIEW_CHARS = 240


def _podcast_attribution_sentence(episode: Episode) -> str:
    """Deterministic, code-generated -- never delegated to the LLM.
    Appended as the article's final sentence at response-assembly time
    (ArticleResponse/PublicArticleResponse.from_models below), never
    persisted to ArticleSection.content: this keeps it completely outside
    the generation/revision/resumability pipeline (app/ai/graph.py) --
    zero risk of it being duplicated across revision rounds, zero risk of
    it leaking into a later section's "preceding content" context
    (app/ai/nodes/section_generation.py::excerpt_preceding_section), no
    interaction with deterministic validation's word-count/paragraph
    checks, and no migration/backfill needed for articles generated
    before this existed -- every article, old or new, gets it on every
    read. Built purely from Episode's own existing metadata (never a
    specific show/host/guest name hardcoded) so this works identically
    for every episode: youtube_url is always present (Episode.youtube_url
    is NOT NULL); channel_name/title are used only when actually
    populated (ingestion metadata isn't always complete), never
    fabricated when absent, the same "never fabricate" rule
    PublicArticleResponse's own source-metadata fields already follow."""
    if episode.channel_name:
        source = f"the podcast conversation from {episode.channel_name}"
    elif episode.title:
        source = f'the podcast episode "{episode.title}"'
    else:
        source = "the source podcast conversation"
    return f"This article is based on {source}. Watch the full podcast here: {episode.youtube_url}"


class SourceChunkResponse(BaseModel):
    """A section's evidence, resolved to real chunk data (not just a bare
    UUID) so a reviewer can jump straight to the timestamp -- Phase I's
    "supporting source chunks, transcript timestamps" requirement."""

    id: uuid.UUID
    start_ms: int
    end_ms: int
    text_preview: str


class ArticleSectionResponse(BaseModel):
    sequence_number: int
    heading: str
    content: str
    supporting_chunks: list[SourceChunkResponse]


class ValidationCheckResponse(BaseModel):
    name: str
    passed: bool
    # "pass" | "warning" | "failure" (app/services/article_validation.py) --
    # defaults keep this compatible with any older persisted checks JSON
    # that predates severity/sections/metrics.
    severity: str = "pass"
    details: str
    sections: list[int] = []
    metrics: dict = {}


class ValidationResultResponse(BaseModel):
    passed: bool
    checks: list[ValidationCheckResponse]
    created_at: datetime

    @classmethod
    def from_model(cls, result: ValidationResult) -> "ValidationResultResponse":
        return cls(
            passed=result.passed,
            checks=[ValidationCheckResponse(**c) for c in result.checks],
            created_at=result.created_at,
        )


class ArticleResponse(BaseModel):
    id: uuid.UUID
    episode_id: uuid.UUID
    title: str
    revision_count: int
    episode_status: ProcessingStatus
    sections: list[ArticleSectionResponse]
    latest_validation: ValidationResultResponse | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_models(
        cls,
        article: Article,
        episode: Episode,
        chunks_by_id: dict[uuid.UUID, Chunk],
    ) -> "ArticleResponse":
        # article.sections is already ordered (relationship's own
        # order_by="ArticleSection.sequence_number", app/models/article.py)
        # -- list()'d once so the deterministic podcast attribution below
        # can be appended to exactly the LAST one by index.
        ordered_sections = list(article.sections)
        attribution = _podcast_attribution_sentence(episode)
        sections = [
            ArticleSectionResponse(
                sequence_number=s.sequence_number,
                heading=s.heading,
                content=(
                    f"{s.content}\n\n{attribution}" if i == len(ordered_sections) - 1 else s.content
                ),
                supporting_chunks=[
                    SourceChunkResponse(
                        id=chunk.id,
                        start_ms=chunk.start_ms,
                        end_ms=chunk.end_ms,
                        text_preview=chunk.text[:_SOURCE_PREVIEW_CHARS],
                    )
                    for cid in s.supporting_chunk_ids
                    if (chunk := chunks_by_id.get(cid)) is not None
                ],
            )
            for i, s in enumerate(ordered_sections)
        ]
        latest_validation = (
            ValidationResultResponse.from_model(article.validation_results[-1])
            if article.validation_results
            else None
        )
        return cls(
            id=article.id,
            episode_id=article.episode_id,
            title=article.title,
            revision_count=article.revision_count,
            episode_status=episode.status,
            sections=sections,
            latest_validation=latest_validation,
            created_at=article.created_at,
            updated_at=article.updated_at,
        )


class GenerateArticleResponse(BaseModel):
    episode_id: uuid.UUID
    job_id: uuid.UUID


class ArticleGenerationStatusResponse(BaseModel):
    """Episode.status Decoupling: article-generation progress lives on
    ProcessingJob.status now, never on Episode.status -- this is the
    reviewer UI's replacement for polling Episode.status
    (app/dev/review/[episodeId]/page.tsx used to poll for
    ANALYZING/PLANNING/GENERATING/VERIFYING/REVISING, none of which
    Episode.status is ever set to anymore). `status` is None when no
    ARTICLE_GENERATION job has ever been created for this episode."""

    status: JobStatus | None
    error_message: str | None

    @classmethod
    def from_model(cls, job: ProcessingJob | None) -> "ArticleGenerationStatusResponse":
        if job is None:
            return cls(status=None, error_message=None)
        return cls(status=job.status, error_message=job.error_message)


# --- Public (published-articles-only) responses ------------------------------------------
#
# Deliberately separate from ArticleResponse above, which is the Phase I
# reviewer-facing shape -- it exposes validation internals (checks,
# severity, metrics) and the Article's own internal id, neither of which
# a public reader has any use for or should see. Never constructed for an
# article whose episode isn't PUBLISHED -- see
# ArticleService.get_published_article / list_published_articles.


class PublicSourceResponse(BaseModel):
    start_ms: int
    end_ms: int
    text_preview: str


class PublicArticleSectionResponse(BaseModel):
    sequence_number: int
    heading: str
    content: str
    supporting_sources: list[PublicSourceResponse]


class PublicArticleResponse(BaseModel):
    episode_id: uuid.UUID
    title: str
    published_at: datetime
    # Source metadata (Episode's own fields, never generated/reconstructed
    # by the LLM -- see app/ai/prompts.py::EpisodeContext's docstring) so
    # the frontend can render a standardized "based on the conversation"
    # block naming the original video/podcast and linking to it. `title`
    # above stays the GENERATED article's own title; `episode_title` is the
    # source video/podcast's own title, which commonly differs from it.
    # All four are nullable exactly as Episode's own columns are (ingestion
    # metadata isn't always fully populated) -- never fabricated when absent.
    episode_title: str | None
    channel_name: str | None
    youtube_url: str
    thumbnail_url: str | None
    sections: list[PublicArticleSectionResponse]

    @classmethod
    def from_models(
        cls, episode: Episode, article: Article, chunks_by_id: dict[uuid.UUID, Chunk]
    ) -> "PublicArticleResponse":
        # Sorted explicitly rather than trusting article.sections' own
        # order: the relationship's order_by="ArticleSection.sequence_number"
        # (app/models/article.py) only applies when SQLAlchemy actually
        # issues the load -- if the same session already has the
        # collection populated from an earlier write in a different order
        # (e.g. a caller that just built the article via
        # ArticleRepository.replace(), whose section list reflects
        # insertion order, not sequence_number), a later query can return
        # that already-loaded, wrongly-ordered in-memory list instead of
        # re-querying. Harmless in real production traffic (each request
        # gets its own fresh session/identity map) but not a guarantee
        # worth leaning on for a public response -- sort explicitly here.
        sections = sorted(article.sections, key=lambda s: s.sequence_number)
        attribution = _podcast_attribution_sentence(episode)
        return cls(
            episode_id=article.episode_id,
            title=article.title,
            published_at=episode.article_published_at,
            episode_title=episode.title,
            channel_name=episode.channel_name,
            youtube_url=episode.youtube_url,
            thumbnail_url=episode.thumbnail_url,
            sections=[
                PublicArticleSectionResponse(
                    sequence_number=s.sequence_number,
                    heading=s.heading,
                    content=(
                        f"{s.content}\n\n{attribution}" if i == len(sections) - 1 else s.content
                    ),
                    supporting_sources=[
                        PublicSourceResponse(
                            start_ms=chunk.start_ms,
                            end_ms=chunk.end_ms,
                            text_preview=chunk.text[:_SOURCE_PREVIEW_CHARS],
                        )
                        for cid in s.supporting_chunk_ids
                        if (chunk := chunks_by_id.get(cid)) is not None
                    ],
                )
                for i, s in enumerate(sections)
            ],
        )


class PublicArticleSummaryResponse(BaseModel):
    episode_id: uuid.UUID
    title: str
    published_at: datetime


class PublicArticleListResponse(BaseModel):
    articles: list[PublicArticleSummaryResponse]
