"""Dev/debug tool: print the article generated for an episode (Phase C-H).

Not an API endpoint -- same reasoning as scripts/inspect_chunks.py, this
is a read tool over whatever the real worker (app/worker/tasks.py::generate_article)
already persisted, not a way to run the pipeline itself. To actually
generate an article, use POST /api/v1/episodes/{id}/generate-article (or
the minimal review UI at frontend/app/dev/review/[episodeId]) against a
running backend + worker with a real LLM provider configured.

Usage (from backend/, with DATABASE_URL pointed at the DB to inspect):

    python -m scripts.inspect_article <episode_id> [--full-text]
"""

import argparse
import asyncio
import sys
import uuid

from app.core.db import get_sessionmaker
from app.repositories.article_repository import ArticleRepository
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.episode_repository import EpisodeRepository
from app.repositories.transcript_repository import TranscriptRepository

_PREVIEW_CHARS = 400


def _format_timestamp(ms: int) -> str:
    total_seconds = ms // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


async def inspect(episode_id: uuid.UUID, *, full_text: bool) -> None:
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        episodes = EpisodeRepository(session)
        transcripts = TranscriptRepository(session)
        chunks_repo = ChunkRepository(session)
        articles = ArticleRepository(session)

        episode = await episodes.get_by_id(episode_id)
        if episode is None:
            print(f"No episode found with id={episode_id}", file=sys.stderr)
            raise SystemExit(1)

        article = await articles.get_by_episode_id(episode_id)
        if article is None:
            print(f"Episode {episode_id} has no generated article yet (status={episode.status.value})")
            raise SystemExit(1)

        transcript = await transcripts.get_by_episode_id(episode_id)
        chunks = await chunks_repo.get_by_transcript_id(transcript.id) if transcript else []
        chunks_by_id = {c.id: c for c in chunks}

        print(f"Episode {episode_id} ({episode.title or 'untitled'}) -- status={episode.status.value}")
        print(f"Article: {article.title!r}  (revision_count={article.revision_count})")

        if article.validation_results:
            latest = article.validation_results[-1]
            print(f"\nLatest validation: {'PASSED' if latest.passed else 'FAILED'} ({latest.created_at})")
            for check in latest.checks:
                mark = "PASS" if check.get("passed") else "FAIL"
                print(f"  [{mark}] {check.get('name')}: {check.get('details')}")
        else:
            print("\nNo validation result recorded.")

        print(f"\n{len(article.sections)} section(s)\n")
        for section in article.sections:
            print(f"## {section.sequence_number + 1}. {section.heading}")
            source_ranges = []
            for cid in section.supporting_chunk_ids:
                chunk = chunks_by_id.get(cid)
                if chunk is not None:
                    source_ranges.append(f"{_format_timestamp(chunk.start_ms)}-{_format_timestamp(chunk.end_ms)}")
            print(f"Sources: {', '.join(source_ranges) if source_ranges else '(none)'}")
            print()
            text = section.content if full_text else section.content[:_PREVIEW_CHARS]
            if not full_text and len(section.content) > _PREVIEW_CHARS:
                text += "..."
            print(text)
            print("\n" + "-" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode_id", type=str, help="Episode UUID")
    parser.add_argument(
        "--full-text", action="store_true", help="Print each section's full text, not a preview"
    )
    args = parser.parse_args()

    try:
        episode_id = uuid.UUID(args.episode_id)
    except ValueError:
        print(f"Not a valid UUID: {args.episode_id}", file=sys.stderr)
        raise SystemExit(2) from None

    asyncio.run(inspect(episode_id, full_text=args.full_text))


if __name__ == "__main__":
    main()
