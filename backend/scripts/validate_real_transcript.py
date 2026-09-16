"""Real-Supadata Phase 3A validation script.

This sandbox's network policy blocks api.supadata.ai (confirmed directly
-- see ARCHITECTURE.md's top-of-document caveat), so this script has never
been run against a real transcript from here. It's meant to be run from an
environment that CAN reach the real API, against a real long-form
podcast/video, to gather the empirical evidence needed to validate (or
challenge) the Phase 3A chunking defaults.

It needs only SUPADATA_API_KEY and network access -- no Postgres, no
Redis, no arq worker. It calls the real SupadataTranscriptProvider directly
and runs the exact same in-memory cleaning/chunking pipeline that
TranscriptProcessingService uses (app/services/{cleaning,chunking}_service.py),
without persisting anything to a database.

It prints statistics and short (truncated by default) text previews to
stdout only -- it never writes the transcript or chunk text to a file, so
running it doesn't create a stored copy of the podcast's content.

Usage:
    cd backend
    pip install -e ".[dev]"
    export SUPADATA_API_KEY=<your real key>
    python -m scripts.validate_real_transcript "<youtube_url>"

    # optional: override chunking config to compare against the defaults
    python -m scripts.validate_real_transcript "<youtube_url>" \\
        --target-tokens 800 --max-tokens 1200 --min-tokens 300

    # optional: print full segment/chunk text instead of truncated previews
    python -m scripts.validate_real_transcript "<youtube_url>" --full-text

Pick a real, publicly accessible long-form podcast/video in the 1-2+ hour
range for this -- that's the regime the CHUNK_TARGET_TOKENS=800 decision
in the completion report needs real evidence for, not a short clip.
"""

import argparse
import asyncio
import os
import statistics
import sys
import uuid

from app.core.exceptions import TranscriptProviderError
from app.models.transcript_segment import TranscriptSegment
from app.providers.transcript.supadata import SupadataTranscriptProvider
from app.services.chunking_service import ChunkingConfig, chunk_transcript_segments
from app.services.cleaning_service import clean_transcript_segments

# Mirrors app/config/settings.py's defaults -- deliberately not importing
# Settings here, since that pulls in full app config validation (fails
# outside APP_ENV=development without unrelated secrets configured). This
# script only needs a Supadata API key.
_DEFAULT_MIN_TOKENS = 300
_DEFAULT_TARGET_TOKENS = 800
_DEFAULT_MAX_TOKENS = 1200
_DEFAULT_OVERLAP_TOKENS = 0
_DEFAULT_PAUSE_THRESHOLD_MS = 1500

_DEFAULT_PREVIEW_CHARS = 200


def _fmt_ts(ms: int) -> str:
    total_seconds = ms // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _preview(text: str, chars: int, full_text: bool) -> str:
    if full_text or len(text) <= chars:
        return text
    return text[:chars] + "..."


async def _fetch(video_url: str, api_key: str):
    provider = SupadataTranscriptProvider(api_key)
    try:
        metadata = None
        try:
            metadata = await provider.get_metadata(video_url)
        except TranscriptProviderError as exc:
            print(f"(metadata fetch failed, non-fatal: {exc})", file=sys.stderr)
        transcript = await provider.get_transcript(video_url)
        return metadata, transcript
    finally:
        await provider.aclose()


def _build_segments(transcript) -> list[TranscriptSegment]:
    # Same construction as TranscriptRepository.create_with_segments --
    # unpersisted (no session), just enough to hand to the real cleaner
    # and chunker exactly like a real ingested transcript would be.
    return [
        TranscriptSegment(
            id=uuid.uuid4(),
            sequence_number=index,
            text=segment.text,
            start_ms=segment.start_ms,
            duration_ms=segment.duration_ms,
            speaker=segment.speaker,
        )
        for index, segment in enumerate(transcript.segments)
    ]


def _report_metadata(metadata) -> None:
    print("=== Episode metadata ===")
    if metadata is None:
        print("(unavailable)")
        return
    print(f"Title: {metadata.title!r}")
    print(f"Channel: {metadata.channel_name!r}")
    print(f"Reported duration: {metadata.duration_seconds}s")
    print(f"Published at: {metadata.published_at!r}")
    print(f"Metadata language: {metadata.language!r}")


def _report_transcript(
    segments: list[TranscriptSegment], transcript, *, preview_chars: int, full_text: bool
) -> None:
    print("\n=== Transcript ===")
    print(f"Language (from transcript payload): {transcript.language!r}")
    print(f"Segment count: {len(segments)}")

    earliest = segments[0].start_ms
    latest = max(s.start_ms + s.duration_ms for s in segments)
    print(f"Earliest timestamp: {_fmt_ts(earliest)} ({earliest} ms)")
    print(f"Latest timestamp: {_fmt_ts(latest)} ({latest} ms)")
    print(f"Transcript duration (derived from segments): {_fmt_ts(latest - earliest)}")

    violations = [
        i for i in range(1, len(segments)) if segments[i].start_ms < segments[i - 1].start_ms
    ]
    if violations:
        print(
            f"Timestamp ordering: {len(violations)} out-of-order segment(s) "
            f"(first at index {violations[0]})"
        )
    else:
        print("Timestamp ordering: OK (non-decreasing start_ms throughout)")

    print("\nRepresentative segments:")
    for label, idx in (
        ("early", len(segments) // 10),
        ("middle", len(segments) // 2),
        ("late", min(len(segments) - 1, (len(segments) * 9) // 10)),
    ):
        s = segments[idx]
        print(
            f"  [{label:6}] segment {idx} @ {_fmt_ts(s.start_ms)}: "
            f"{_preview(s.text, preview_chars, full_text)!r}"
        )

    total_chars = sum(len(s.text) for s in segments)
    total_words = sum(len(s.text.split()) for s in segments)
    print(f"\nTotal raw text size: {total_chars} chars / {total_words} words")


def _report_chunks(
    segments: list[TranscriptSegment],
    cleaned,
    chunks,
    config: ChunkingConfig,
    *,
    preview_chars: int,
    full_text: bool,
) -> None:
    print(
        f"\n=== Chunks (min={config.min_tokens} target={config.target_tokens} "
        f"max={config.max_tokens} overlap={config.overlap_tokens} "
        f"pause_threshold={config.pause_threshold_ms}ms) ==="
    )
    print(f"Chunk count: {len(chunks)}")

    if not chunks:
        print("(no chunks produced)")
        return

    token_counts = [c.token_count for c in chunks]
    durations_ms = [c.end_ms - c.start_ms for c in chunks]
    print(
        f"Tokens: min={min(token_counts)} median={statistics.median(token_counts):.0f} "
        f"avg={statistics.mean(token_counts):.1f} max={max(token_counts)}"
    )
    print(
        f"Duration: min={_fmt_ts(min(durations_ms))} "
        f"median={_fmt_ts(round(statistics.median(durations_ms)))} "
        f"avg={_fmt_ts(round(statistics.mean(durations_ms)))} "
        f"max={_fmt_ts(max(durations_ms))}"
    )

    # --- source reconstruction / loss check ---
    original_words = " ".join(s.text for s in cleaned).split()
    reconstructed_words = " ".join(c.text for c in chunks).split()
    if original_words == reconstructed_words:
        print("Source reconstruction check: MATCH (no words lost or duplicated)")
    else:
        print(
            "Source reconstruction check: MISMATCH -- "
            f"{len(original_words)} cleaned words vs {len(reconstructed_words)} "
            "reconstructed words. Investigate before trusting these chunks."
        )

    # --- duplicate chunk text check ---
    seen: set[str] = set()
    duplicate_count = 0
    for c in chunks:
        if c.text in seen:
            duplicate_count += 1
        seen.add(c.text)
    print(
        "Duplicate chunk text: none"
        if duplicate_count == 0
        else f"Duplicate chunk text: {duplicate_count} chunk(s) share exact text with another"
    )

    # --- source_segment_ids completeness check ---
    all_segment_ids = {s.id for s in segments}
    covered_ids: set = set()
    for c in chunks:
        covered_ids.update(c.source_segment_ids)
    missing_ids = all_segment_ids - covered_ids
    empty_after_cleaning = {cs.segment_id for cs in cleaned if not cs.text}
    unexplained_missing = missing_ids - empty_after_cleaning
    print(
        f"Segments not covered by any chunk: {len(missing_ids)} "
        f"({len(missing_ids) - len(unexplained_missing)} explained by cleaning to empty text, "
        f"{len(unexplained_missing)} UNEXPLAINED)"
    )

    # --- representative chunk boundaries ---
    print("\n=== Representative chunk boundaries ===")
    indices = sorted(
        {0, len(chunks) // 4, len(chunks) // 2, (3 * len(chunks)) // 4, len(chunks) - 1}
    )
    for i in indices:
        c = chunks[i]
        print(
            f"Chunk {c.sequence_number + 1:03d}  {_fmt_ts(c.start_ms)} -> {_fmt_ts(c.end_ms)}  "
            f"tokens={c.token_count}  source_segments={len(c.source_segment_ids)}"
        )
        print(f"    {_preview(c.text, preview_chars, full_text)!r}")

    # --- chunks near the max-token ceiling ---
    near_max = [c for c in chunks if c.token_count >= config.max_tokens * 0.9]
    print(f"\nChunks within 90% of max_tokens ({config.max_tokens}): {len(near_max)}")
    for c in near_max[:10]:
        print(f"  Chunk {c.sequence_number + 1:03d}: {c.token_count} tokens")

    # --- boundary pause analysis: did each cut land on a real pause? ---
    print("\n=== Boundary pause analysis (does each cut align with a real pause?) ===")
    for i in range(len(chunks) - 1):
        gap_ms = chunks[i + 1].start_ms - chunks[i].end_ms
        signal = "PAUSE" if gap_ms >= config.pause_threshold_ms else "no-pause (sentence/forced cut)"
        print(
            f"  Boundary {chunks[i].sequence_number + 1:03d} -> "
            f"{chunks[i + 1].sequence_number + 1:03d}: gap={gap_ms}ms [{signal}]"
        )

    # --- trailing chunk check ---
    trailing = chunks[-1]
    print(
        f"\nTrailing chunk: {trailing.token_count} tokens "
        f"({'below min_tokens, expected for a trailing chunk' if trailing.token_count < config.min_tokens else 'at or above min_tokens'})"
    )


async def _run(args: argparse.Namespace) -> None:
    api_key = os.environ.get("SUPADATA_API_KEY")
    if not api_key:
        print("SUPADATA_API_KEY is not set in the environment.", file=sys.stderr)
        raise SystemExit(1)

    metadata, transcript = await _fetch(args.video_url, api_key)
    segments = _build_segments(transcript)
    if not segments:
        print("Provider returned zero segments -- nothing to validate.", file=sys.stderr)
        raise SystemExit(1)

    _report_metadata(metadata)
    _report_transcript(segments, transcript, preview_chars=args.preview_chars, full_text=args.full_text)

    cleaned = clean_transcript_segments(segments)
    config = ChunkingConfig(
        min_tokens=args.min_tokens,
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        pause_threshold_ms=args.pause_threshold_ms,
    )
    chunks = chunk_transcript_segments(cleaned, config)
    _report_chunks(
        segments, cleaned, chunks, config, preview_chars=args.preview_chars, full_text=args.full_text
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video_url", help="A real, publicly accessible YouTube URL")
    parser.add_argument("--min-tokens", type=int, default=_DEFAULT_MIN_TOKENS)
    parser.add_argument("--target-tokens", type=int, default=_DEFAULT_TARGET_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=_DEFAULT_MAX_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=_DEFAULT_OVERLAP_TOKENS)
    parser.add_argument("--pause-threshold-ms", type=int, default=_DEFAULT_PAUSE_THRESHOLD_MS)
    parser.add_argument("--preview-chars", type=int, default=_DEFAULT_PREVIEW_CHARS)
    parser.add_argument(
        "--full-text",
        action="store_true",
        help="Print full segment/chunk text instead of truncated previews (stdout only, never saved to a file)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(_run(args))
    except TranscriptProviderError as exc:
        print(f"\nSupadata request failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
