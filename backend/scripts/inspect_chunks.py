"""Dev/debug tool: print the chunks generated for an episode (Phase 3A §3A.13).

Not an API endpoint -- deliberately a script, since chunk inspection is a
development/QA activity, not a product feature yet.

Usage (from backend/, with DATABASE_URL pointed at the DB to inspect):

    python -m scripts.inspect_chunks <episode_id> [--full-text] [--segments]

Example output:

    Chunk 001
    00:00:12 -> 00:05:48
    Tokens: 742
    Source segments: [1..37]

    The guest argues that scaling is not merely...
"""

import argparse
import asyncio
import sys
import uuid

from app.core.db import get_sessionmaker
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.episode_repository import EpisodeRepository
from app.repositories.transcript_repository import TranscriptRepository

_PREVIEW_CHARS = 240


def _format_timestamp(ms: int) -> str:
    total_seconds = ms // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _format_source_segments(sequence_numbers: list[int]) -> str:
    if not sequence_numbers:
        return "[]"
    ordered = sorted(sequence_numbers)
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"[{ordered[0]}..{ordered[-1]}]"
    return "[" + ", ".join(str(n) for n in ordered) + "]"


async def inspect(episode_id: uuid.UUID, *, full_text: bool, show_segments: bool) -> None:
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        episodes = EpisodeRepository(session)
        transcripts = TranscriptRepository(session)
        chunks_repo = ChunkRepository(session)

        episode = await episodes.get_by_id(episode_id)
        if episode is None:
            print(f"No episode found with id={episode_id}", file=sys.stderr)
            raise SystemExit(1)

        transcript = await transcripts.get_by_episode_id(episode_id)
        if transcript is None:
            print(f"Episode {episode_id} has no transcript yet (status={episode.status.value})")
            raise SystemExit(1)

        chunks = await chunks_repo.get_by_transcript_id(transcript.id)
        if not chunks:
            print(f"Episode {episode_id} has no chunks yet.")
            return

        segment_sequence_by_id = {s.id: s.sequence_number for s in transcript.segments}

        print(f"Episode {episode_id} ({episode.title or 'untitled'})")
        print(f"Transcript {transcript.id}: {len(transcript.segments)} segments")
        print(f"{len(chunks)} chunks\n")

        for chunk in chunks:
            sequence_numbers = [
                segment_sequence_by_id[sid]
                for sid in chunk.source_segment_ids
                if sid in segment_sequence_by_id
            ]
            print(f"Chunk {chunk.sequence_number + 1:03d}")
            print(f"{_format_timestamp(chunk.start_ms)} -> {_format_timestamp(chunk.end_ms)}")
            print(f"Tokens: {chunk.token_count}")
            print(f"Source segments: {_format_source_segments(sequence_numbers)}")
            if show_segments:
                print(f"Source segment IDs: {chunk.source_segment_ids}")
            print()
            text = chunk.text if full_text else chunk.text[:_PREVIEW_CHARS]
            if not full_text and len(chunk.text) > _PREVIEW_CHARS:
                text += "..."
            print(text)
            print("\n" + "-" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_id", type=str, help="Episode UUID")
    parser.add_argument(
        "--full-text", action="store_true", help="Print each chunk's full text, not a preview"
    )
    parser.add_argument(
        "--segments", action="store_true", help="Also print each chunk's raw source segment IDs"
    )
    args = parser.parse_args()

    try:
        episode_id = uuid.UUID(args.episode_id)
    except ValueError:
        print(f"Not a valid UUID: {args.episode_id}", file=sys.stderr)
        raise SystemExit(2) from None

    asyncio.run(inspect(episode_id, full_text=args.full_text, show_segments=args.segments))


if __name__ == "__main__":
    main()
