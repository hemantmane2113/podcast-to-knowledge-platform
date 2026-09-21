import uuid
from datetime import datetime

from pydantic import BaseModel

from app.models.article import Article
from app.models.chunk import Chunk
from app.models.episode import ProcessingStatus
from app.models.validation_result import ValidationResult

_SOURCE_PREVIEW_CHARS = 240


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
    details: str


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
        episode_status: ProcessingStatus,
        chunks_by_id: dict[uuid.UUID, Chunk],
    ) -> "ArticleResponse":
        sections = [
            ArticleSectionResponse(
                sequence_number=s.sequence_number,
                heading=s.heading,
                content=s.content,
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
            for s in article.sections
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
            episode_status=episode_status,
            sections=sections,
            latest_validation=latest_validation,
            created_at=article.created_at,
            updated_at=article.updated_at,
        )


class GenerateArticleResponse(BaseModel):
    episode_id: uuid.UUID
    job_id: uuid.UUID
