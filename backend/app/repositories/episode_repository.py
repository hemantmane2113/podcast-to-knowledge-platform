import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.episode import Episode, ProcessingStatus
from app.schemas.transcript import EpisodeMetadata


class EpisodeRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_id(self, episode_id: uuid.UUID) -> Episode | None:
        return await self._session.get(Episode, episode_id)

    async def get_by_id_for_update(self, episode_id: uuid.UUID) -> Episode | None:
        """SELECT ... FOR UPDATE -- locks the episode row for the rest of
        the caller's transaction, serializing concurrent multi-step
        read-then-write sequences against it (e.g. two promote_draft
        calls racing for the same episode) rather than letting them
        interleave. Only ArticleService.promote_draft needs this today."""
        result = await self._session.execute(
            select(Episode).where(Episode.id == episode_id).with_for_update()
        )
        return result.scalar_one_or_none()

    async def get_by_youtube_video_id(self, youtube_video_id: str) -> Episode | None:
        result = await self._session.execute(
            select(Episode).where(Episode.youtube_video_id == youtube_video_id)
        )
        return result.scalar_one_or_none()

    def create(self, *, youtube_video_id: str, youtube_url: str) -> Episode:
        # id assigned explicitly (rather than relying on the column default,
        # which only fires at flush) so callers can use episode.id
        # immediately, e.g. to create the ProcessingJob row in the same
        # transaction without an extra flush round-trip.
        episode = Episode(
            id=uuid.uuid4(),
            youtube_video_id=youtube_video_id,
            youtube_url=youtube_url,
            status=ProcessingStatus.INGESTING,
        )
        self._session.add(episode)
        return episode

    def apply_metadata(self, episode: Episode, metadata: EpisodeMetadata) -> None:
        """Only overwrites fields the provider actually returned — never
        clears a previously known value with a null one, and never
        fabricates a value the provider didn't supply."""
        if metadata.title is not None:
            episode.title = metadata.title
        if metadata.description is not None:
            episode.description = metadata.description
        if metadata.channel_name is not None:
            episode.channel_name = metadata.channel_name
        if metadata.channel_id is not None:
            episode.channel_id = metadata.channel_id
        if metadata.thumbnail_url is not None:
            episode.thumbnail_url = metadata.thumbnail_url
        if metadata.duration_seconds is not None:
            episode.duration_seconds = metadata.duration_seconds
        if metadata.language is not None:
            episode.language = metadata.language
        if metadata.published_at is not None:
            parsed = _parse_iso_datetime(metadata.published_at)
            if parsed is not None:
                episode.published_at = parsed

    def set_status(
        self, episode: Episode, status: ProcessingStatus, *, last_error: str | None = None
    ) -> None:
        episode.status = status
        episode.last_error = last_error

    def mark_published(self, episode: Episode) -> None:
        """Sets status=PUBLISHED and stamps article_published_at with the
        current time -- a raw setter, same shape as set_status above.
        Idempotency (never call this a second time for an
        already-PUBLISHED episode, so the timestamp never gets bumped
        forward on a repeat publish call) is the caller's job -- see
        ArticleService.publish_article."""
        episode.status = ProcessingStatus.PUBLISHED
        episode.article_published_at = datetime.now(UTC)
        episode.last_error = None


def _parse_iso_datetime(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
