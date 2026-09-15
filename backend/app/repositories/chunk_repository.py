import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import Chunk
from app.services.chunking_service import ChunkCandidate


class ChunkRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_transcript_id(self, transcript_id: uuid.UUID) -> list[Chunk]:
        result = await self._session.execute(
            select(Chunk)
            .where(Chunk.transcript_id == transcript_id)
            .order_by(Chunk.sequence_number)
        )
        return list(result.scalars().all())

    async def replace_all(
        self, *, transcript_id: uuid.UUID, episode_id: uuid.UUID, candidates: list[ChunkCandidate]
    ) -> list[Chunk]:
        """Idempotency strategy (PRODUCT_SPEC.md §51/§75, Phase 3A §3A.8):
        chunking is a pure function of (cleaned segments, config), so
        there's nothing worth preserving across a rerun — delete this
        transcript's existing chunks and insert the freshly computed set,
        in the same transaction the caller commits. Running chunking twice
        on unchanged input therefore always ends with the same logical
        chunk set, and there's never a window where old and new chunks
        coexist or a unique-constraint collision from a naive re-insert.
        """
        await self._session.execute(delete(Chunk).where(Chunk.transcript_id == transcript_id))

        rows = [
            Chunk(
                id=uuid.uuid4(),
                transcript_id=transcript_id,
                episode_id=episode_id,
                sequence_number=candidate.sequence_number,
                text=candidate.text,
                start_ms=candidate.start_ms,
                end_ms=candidate.end_ms,
                source_segment_ids=candidate.source_segment_ids,
                token_count=candidate.token_count,
            )
            for candidate in candidates
        ]
        self._session.add_all(rows)
        return rows
