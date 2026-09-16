import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import EpisodeNotFoundError, TranscriptNotFoundError
from app.models.episode import ProcessingStatus
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.episode_repository import EpisodeRepository
from app.repositories.transcript_repository import TranscriptRepository
from app.services.chunking_service import ChunkingConfig, chunk_transcript_segments
from app.services.cleaning_service import clean_transcript_segments


class TranscriptProcessingService:
    """Cleaning + chunking (Phase 3A) — the async step after ingestion.
    Only ever called from the worker (app/worker/tasks.py), same pattern
    as IngestionService.
    """

    def __init__(self, session: AsyncSession, config: ChunkingConfig):
        self._session = session
        self._config = config
        self._episodes = EpisodeRepository(session)
        self._transcripts = TranscriptRepository(session)
        self._chunks = ChunkRepository(session)

    async def run(self, episode_id: uuid.UUID) -> int:
        """Returns the number of chunks generated (0 is a valid outcome,
        not an error — e.g. a transcript that's entirely non-speech
        markers). Raises EpisodeNotFoundError / TranscriptNotFoundError if
        there's nothing to process yet.
        """
        episode = await self._episodes.get_by_id(episode_id)
        if episode is None:
            raise EpisodeNotFoundError(f"Episode {episode_id} not found")

        transcript = await self._transcripts.get_by_episode_id(episode_id)
        if transcript is None:
            raise TranscriptNotFoundError(f"Episode {episode_id} has no transcript to process")

        # Cleaning returns transient in-memory CleanedSegments -- raw
        # transcript.segments (and their .text) are never mutated. The
        # chunker consumes that cleaned output directly; nothing about
        # cleaning is persisted on its own.
        cleaned_segments = clean_transcript_segments(transcript.segments)

        candidates = chunk_transcript_segments(cleaned_segments, self._config)
        await self._chunks.replace_all(
            transcript_id=transcript.id, episode_id=episode_id, candidates=candidates
        )

        # CHUNKING is the resting state once this completes -- ANALYZING
        # (the next real pipeline stage) doesn't exist yet (Phase 4).
        self._episodes.set_status(episode, ProcessingStatus.CHUNKING)

        await self._session.commit()
        return len(candidates)
