import uuid
from dataclasses import dataclass, field

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.topic import Topic


@dataclass
class TopicCandidate:
    """The repository's insertable-row shape, decoupled from whatever
    structured LLM schema produced it (app/ai/schemas.py) -- mirrors
    ChunkCandidate's role for chunking_service.py/ChunkRepository. The AI
    node resolves the model's chunk sequence_number references into real
    Chunk.id values before constructing one of these.
    """

    sequence_number: int
    title: str
    summary: str
    chunk_ids: list[uuid.UUID]
    key_claims: list[dict] = field(default_factory=list)
    subtopics: list[str] = field(default_factory=list)


class TopicRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_transcript_id(self, transcript_id: uuid.UUID) -> list[Topic]:
        result = await self._session.execute(
            select(Topic).where(Topic.transcript_id == transcript_id).order_by(Topic.sequence_number)
        )
        return list(result.scalars().all())

    async def replace_all(
        self, *, transcript_id: uuid.UUID, episode_id: uuid.UUID, candidates: list[TopicCandidate]
    ) -> list[Topic]:
        """Same idempotency strategy as ChunkRepository.replace_all: topic
        analysis is a (near-)pure function of a transcript's chunks, so a
        rerun deletes and reinserts rather than diffing."""
        await self._session.execute(delete(Topic).where(Topic.transcript_id == transcript_id))

        rows = [
            Topic(
                id=uuid.uuid4(),
                transcript_id=transcript_id,
                episode_id=episode_id,
                sequence_number=c.sequence_number,
                title=c.title,
                summary=c.summary,
                chunk_ids=c.chunk_ids,
                key_claims=c.key_claims,
                subtopics=c.subtopics,
            )
            for c in candidates
        ]
        self._session.add_all(rows)
        return rows
