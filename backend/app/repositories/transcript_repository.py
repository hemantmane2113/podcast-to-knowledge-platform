import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.transcript import Transcript
from app.models.transcript_segment import TranscriptSegment
from app.schemas.transcript import NormalizedTranscript


class TranscriptRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_episode_id(self, episode_id: uuid.UUID) -> Transcript | None:
        result = await self._session.execute(
            select(Transcript)
            .where(Transcript.episode_id == episode_id)
            .options(selectinload(Transcript.segments))
        )
        return result.scalar_one_or_none()

    def create_with_segments(
        self, episode_id: uuid.UUID, normalized: NormalizedTranscript
    ) -> Transcript:
        """Builds the Transcript and all TranscriptSegment rows in memory
        and adds them to the session as a single unit — the caller commits
        once, so either the whole transcript persists or none of it does
        (PRODUCT_SPEC.md §75, no partial persistence)."""
        transcript = Transcript(id=uuid.uuid4(), episode_id=episode_id, language=normalized.language)
        transcript.segments = [
            TranscriptSegment(
                id=uuid.uuid4(),
                sequence_number=index,
                text=segment.text,
                start_ms=segment.start_ms,
                duration_ms=segment.duration_ms,
                speaker=segment.speaker,
            )
            for index, segment in enumerate(normalized.segments)
        ]
        self._session.add(transcript)
        return transcript
