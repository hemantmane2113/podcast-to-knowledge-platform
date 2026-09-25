import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.episode import Episode, ProcessingStatus
from app.models.transcript import Transcript
from app.repositories.chunk_repository import ChunkRepository
from app.services.chunking_service import ChunkCandidate


async def _seed_episode_and_transcript(session: AsyncSession) -> tuple[Episode, Transcript]:
    episode = Episode(
        id=uuid.uuid4(),
        youtube_video_id="abc12345678",
        youtube_url="https://www.youtube.com/watch?v=abc12345678",
        status=ProcessingStatus.TRANSCRIPT_FETCHED,
    )
    session.add(episode)
    transcript = Transcript(id=uuid.uuid4(), episode_id=episode.id, language="en")
    session.add(transcript)
    await session.commit()
    return episode, transcript


def _candidate(seq: int, text: str = "some chunk text") -> ChunkCandidate:
    return ChunkCandidate(
        sequence_number=seq,
        text=text,
        start_ms=seq * 1000,
        end_ms=seq * 1000 + 900,
        source_segment_ids=[uuid.uuid4()],
        token_count=len(text) // 4 or 1,
    )


async def test_replace_all_persists_chunks(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    repo = ChunkRepository(db_session)

    candidates = [_candidate(0, "first"), _candidate(1, "second")]
    await repo.replace_all(
        transcript_id=transcript.id, episode_id=episode.id, candidates=candidates
    )
    await db_session.commit()

    stored = await repo.get_by_transcript_id(transcript.id)
    assert len(stored) == 2
    assert [c.sequence_number for c in stored] == [0, 1]
    assert [c.text for c in stored] == ["first", "second"]
    assert all(c.episode_id == episode.id for c in stored)


async def test_replace_all_is_idempotent_on_identical_input(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    repo = ChunkRepository(db_session)
    candidates = [_candidate(0, "alpha"), _candidate(1, "beta"), _candidate(2, "gamma")]

    await repo.replace_all(
        transcript_id=transcript.id, episode_id=episode.id, candidates=candidates
    )
    await db_session.commit()
    first_run_ids = {c.id for c in await repo.get_by_transcript_id(transcript.id)}

    await repo.replace_all(
        transcript_id=transcript.id, episode_id=episode.id, candidates=candidates
    )
    await db_session.commit()
    second_run = await repo.get_by_transcript_id(transcript.id)

    # Same logical set (count + content), not literally the same rows --
    # replace_all deletes and reinserts, so IDs are expected to differ.
    assert len(second_run) == 3
    assert [c.text for c in second_run] == ["alpha", "beta", "gamma"]
    assert first_run_ids.isdisjoint({c.id for c in second_run})


async def test_replace_all_with_different_candidates_drops_stale_chunks(
    db_session: AsyncSession,
) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    repo = ChunkRepository(db_session)

    await repo.replace_all(
        transcript_id=transcript.id,
        episode_id=episode.id,
        candidates=[_candidate(0, "one"), _candidate(1, "two"), _candidate(2, "three")],
    )
    await db_session.commit()
    assert len(await repo.get_by_transcript_id(transcript.id)) == 3

    # A rerun after the source transcript/config changed, producing fewer chunks.
    await repo.replace_all(
        transcript_id=transcript.id,
        episode_id=episode.id,
        candidates=[_candidate(0, "merged one and two")],
    )
    await db_session.commit()

    stored = await repo.get_by_transcript_id(transcript.id)
    assert len(stored) == 1
    assert stored[0].text == "merged one and two"


async def test_deleting_episode_cascades_to_chunks(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    repo = ChunkRepository(db_session)
    await repo.replace_all(
        transcript_id=transcript.id, episode_id=episode.id, candidates=[_candidate(0)]
    )
    await db_session.commit()

    await db_session.delete(episode)
    await db_session.commit()

    assert await repo.get_by_transcript_id(transcript.id) == []
