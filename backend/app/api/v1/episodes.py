import uuid

from fastapi import APIRouter, Response, status

from app.api.deps import EpisodeServiceDep
from app.schemas.episode import (
    EpisodeCreateRequest,
    EpisodeResponse,
    TranscriptResponse,
    TranscriptSegmentResponse,
)

router = APIRouter(prefix="/episodes", tags=["episodes"])


@router.post("", response_model=EpisodeResponse)
async def create_episode(
    payload: EpisodeCreateRequest, response: Response, service: EpisodeServiceDep
) -> EpisodeResponse:
    """Validates the URL, then idempotently creates the episode and
    enqueues ingestion (PRODUCT_SPEC.md §56, §57 -- returns immediately,
    never waits for the transcript fetch)."""
    episode, created = await service.create_episode(payload.youtube_url)
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return EpisodeResponse.model_validate(episode)


@router.get("/{episode_id}", response_model=EpisodeResponse)
async def get_episode(episode_id: uuid.UUID, service: EpisodeServiceDep) -> EpisodeResponse:
    episode = await service.get_episode(episode_id)
    return EpisodeResponse.model_validate(episode)


@router.get("/{episode_id}/transcript", response_model=TranscriptResponse)
async def get_episode_transcript(episode_id: uuid.UUID, service: EpisodeServiceDep) -> TranscriptResponse:
    transcript = await service.get_transcript(episode_id)
    return TranscriptResponse(
        episode_id=episode_id,
        language=transcript.language,
        segments=[TranscriptSegmentResponse.model_validate(s) for s in transcript.segments],
    )
