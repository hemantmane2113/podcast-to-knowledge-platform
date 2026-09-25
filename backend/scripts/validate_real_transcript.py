"""Real-Supadata Phase 3A validation script.

This sandbox's network policy blocks api.supadata.ai (confirmed directly
-- see ARCHITECTURE.md's top-of-document caveat). It's meant to be run
from an environment that CAN reach the real API, against a real long-form
podcast/video, to gather the empirical evidence needed to validate (or
challenge) the Phase 3A chunking defaults.

It needs only SUPADATA_API_KEY and network access -- no Postgres, no
Redis, no arq worker, and it never persists anything (no DB writes, no
transcript storage). It calls the real SupadataTranscriptProvider directly
and runs the exact same in-memory cleaning/chunking pipeline that
TranscriptProcessingService uses (app/services/{cleaning,chunking}_service.py)
completely unmodified -- this script is read-only diagnostics over that
pipeline's output, never a second implementation of it.

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
needs real evidence for, not a short clip.

--- Boundary diagnostics (why this file has a "pure calculation" section) ---

Every chunk-to-chunk boundary is classified using only the finalized
ChunkCandidate output plus the raw TranscriptSegment timestamps -- this
script does NOT re-run or instrument chunk_transcript_segments()'s internal
decision loop, so the classification below is a faithful *reconstruction*
of the algorithm's own stated preference order (pause > sentence > forced),
not a byte-exact replay of which branch fired. Two things fall out of the
real segment data rather than the chunk's own start_ms/end_ms fields:

1. A single raw segment whose text alone exceeds max_tokens is split into
   multiple chunk-sized pieces (see chunking_service.py::_split_long_text),
   and a forced split can land squarely on a chunk boundary -- the SAME
   raw segment then legitimately appears as the last contributing segment
   of chunk N and the first contributing segment of chunk N+1. Computing a
   "gap" between a segment and itself always comes out negative (its own
   -duration_ms), which is not a timestamp problem, just a degenerate case
   of the gap formula -- this script detects and labels it distinctly
   ("forced_segment_split") instead of reporting a nonsensical negative gap.
2. Genuine overlapping raw timestamps between two DIFFERENT consecutive
   segments (a known real-world artifact of rolling-window auto-captions --
   see cleaning_service.py's cross-segment overlap dedup) would also
   produce a negative gap, and unlike case 1 is worth flagging as a real
   data-quality observation. find_raw_segment_overlaps() checks this
   directly, transcript-wide, independent of chunk boundaries.
"""

import argparse
import asyncio
import os
import statistics
import sys
import uuid
from dataclasses import dataclass

from app.core.exceptions import TranscriptProviderError
from app.models.transcript_segment import TranscriptSegment
from app.providers.transcript.supadata import SupadataTranscriptProvider
from app.services.chunking_service import ChunkCandidate, ChunkingConfig, _ends_sentence, chunk_transcript_segments
from app.services.cleaning_service import CleanedSegment, clean_transcript_segments

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

# (label, predicate) in check order -- first match wins. Matches the exact
# buckets asked for in the Phase 3A validation-pass follow-up.
_TOKEN_BUCKETS: list[tuple[str, "callable"]] = [
    ("<500", lambda t: t < 500),
    ("500-699", lambda t: 500 <= t <= 699),
    ("700-799", lambda t: 700 <= t <= 799),
    ("800-899", lambda t: 800 <= t <= 899),
    ("900-999", lambda t: 900 <= t <= 999),
    ("1000-1079", lambda t: 1000 <= t <= 1079),
    ("1080-1199", lambda t: 1080 <= t <= 1199),
    ("1200+", lambda t: t >= 1200),
]


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


# ============================================================================
# Pure calculation helpers -- no I/O, unit-tested in
# tests/unit/test_validate_real_transcript.py.
# ============================================================================


@dataclass
class SegmentOverlap:
    """Two consecutive (by sequence_number) raw segments whose real
    timestamps overlap: the earlier one's end_ms (start_ms + duration_ms)
    is after the later one's start_ms."""

    prev_sequence: int
    next_sequence: int
    overlap_ms: int


def find_raw_segment_overlaps(segments: list[TranscriptSegment]) -> list[SegmentOverlap]:
    """Transcript-wide check, independent of chunking: do any two
    consecutive raw segments' timestamps genuinely overlap? Assumes
    `segments` is already in sequence_number order (as built by
    _build_segments / TranscriptRepository.create_with_segments)."""
    overlaps: list[SegmentOverlap] = []
    for i in range(1, len(segments)):
        prev, curr = segments[i - 1], segments[i]
        prev_end_ms = prev.start_ms + prev.duration_ms
        if curr.start_ms < prev_end_ms:
            overlaps.append(
                SegmentOverlap(
                    prev_sequence=prev.sequence_number,
                    next_sequence=curr.sequence_number,
                    overlap_ms=prev_end_ms - curr.start_ms,
                )
            )
    return overlaps


def segment_by_id_index(segments: list[TranscriptSegment]) -> dict[uuid.UUID, TranscriptSegment]:
    return {s.id: s for s in segments}


def chunk_contributing_segments(
    chunk: ChunkCandidate, segment_by_id: dict[uuid.UUID, TranscriptSegment]
) -> tuple[TranscriptSegment, TranscriptSegment]:
    """The first and last raw segment (by sequence_number) that contributed
    text to this chunk. Not simply source_segment_ids[0]/[-1]: that list is
    in first-occurrence order during chunk assembly, which already matches
    sequence order in practice, but sorting here makes that guarantee
    explicit rather than incidental."""
    contributing = sorted(
        (segment_by_id[sid] for sid in chunk.source_segment_ids if sid in segment_by_id),
        key=lambda s: s.sequence_number,
    )
    return contributing[0], contributing[-1]


@dataclass
class BoundaryInfo:
    """Diagnostics for one chunk-to-chunk boundary (between chunk i and
    chunk i+1). See the module docstring for how `mechanism` is derived
    and why `gap_ms` is None for a forced mid-segment split."""

    prev_chunk_sequence: int
    next_chunk_sequence: int
    prev_token_count: int
    next_token_count: int
    prev_last_segment_id: uuid.UUID
    prev_last_segment_sequence: int
    next_first_segment_id: uuid.UUID
    next_first_segment_sequence: int
    same_segment_split: bool
    gap_ms: int | None
    mechanism: str  # "pause" | "sentence" | "forced_max" | "forced_segment_split"
    prev_reached_target: bool


def compute_boundaries(
    chunks: list[ChunkCandidate],
    segments: list[TranscriptSegment],
    cleaned: list[CleanedSegment],
    config: ChunkingConfig,
) -> list[BoundaryInfo]:
    segment_by_id = segment_by_id_index(segments)
    cleaned_text_by_id = {cs.segment_id: cs.text for cs in cleaned}

    boundaries: list[BoundaryInfo] = []
    for i in range(len(chunks) - 1):
        prev_chunk, next_chunk = chunks[i], chunks[i + 1]
        _, prev_last = chunk_contributing_segments(prev_chunk, segment_by_id)
        next_first, _ = chunk_contributing_segments(next_chunk, segment_by_id)

        same_segment_split = prev_last.id == next_first.id
        if same_segment_split:
            # A single oversized segment straddles this boundary (see the
            # module docstring, case 1) -- "gap" against itself is
            # meaningless, not a timestamp defect.
            gap_ms = None
            mechanism = "forced_segment_split"
        else:
            gap_ms = next_first.start_ms - (prev_last.start_ms + prev_last.duration_ms)
            if gap_ms >= config.pause_threshold_ms:
                mechanism = "pause"
            elif _ends_sentence(cleaned_text_by_id.get(prev_last.id, "")):
                mechanism = "sentence"
            else:
                mechanism = "forced_max"

        boundaries.append(
            BoundaryInfo(
                prev_chunk_sequence=prev_chunk.sequence_number,
                next_chunk_sequence=next_chunk.sequence_number,
                prev_token_count=prev_chunk.token_count,
                next_token_count=next_chunk.token_count,
                prev_last_segment_id=prev_last.id,
                prev_last_segment_sequence=prev_last.sequence_number,
                next_first_segment_id=next_first.id,
                next_first_segment_sequence=next_first.sequence_number,
                same_segment_split=same_segment_split,
                gap_ms=gap_ms,
                mechanism=mechanism,
                # The finalized chunk's token_count IS the accumulated size
                # at the chosen cut point (chunking_service.py checks
                # current_tokens >= target_tokens before treating a unit as
                # a candidate boundary), so this reads that decision back
                # from the output rather than approximating it.
                prev_reached_target=prev_chunk.token_count >= config.target_tokens,
            )
        )
    return boundaries


def boundary_mechanism_summary(boundaries: list[BoundaryInfo]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for b in boundaries:
        counts[b.mechanism] = counts.get(b.mechanism, 0) + 1
    return counts


def token_distribution(chunks: list[ChunkCandidate]) -> dict[str, int]:
    counts = {label: 0 for label, _ in _TOKEN_BUCKETS}
    for c in chunks:
        for label, predicate in _TOKEN_BUCKETS:
            if predicate(c.token_count):
                counts[label] += 1
                break
    return counts


def size_bands(chunks: list[ChunkCandidate], config: ChunkingConfig) -> dict[str, int]:
    near_max_floor = config.max_tokens * 0.9
    bands = {"below_target": 0, "target_to_90pct_max": 0, "at_or_above_90pct_max": 0}
    for c in chunks:
        if c.token_count < config.target_tokens:
            bands["below_target"] += 1
        elif c.token_count >= near_max_floor:
            bands["at_or_above_90pct_max"] += 1
        else:
            bands["target_to_90pct_max"] += 1
    return bands


# ============================================================================
# Provider I/O
# ============================================================================


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


# ============================================================================
# Reporting (I/O -- everything above this line is pure and testable)
# ============================================================================


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

    overlaps = find_raw_segment_overlaps(segments)
    if overlaps:
        total_overlap_ms = sum(o.overlap_ms for o in overlaps)
        print(
            f"Raw segment timestamp overlaps: {len(overlaps)} consecutive pair(s) overlap "
            f"(total {total_overlap_ms}ms across all of them)"
        )
        for o in overlaps[:10]:
            print(f"    segment {o.prev_sequence} overlaps segment {o.next_sequence} by {o.overlap_ms}ms")
        if len(overlaps) > 10:
            print(f"    ... and {len(overlaps) - 10} more")
    else:
        print("Raw segment timestamp overlaps: none")

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


def _report_chunk_summary(chunks: list[ChunkCandidate], config: ChunkingConfig) -> None:
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

    print("\nToken distribution:")
    dist = token_distribution(chunks)
    for label, count in dist.items():
        pct = (count / len(chunks)) * 100
        print(f"  {label:10} {count:4d}  ({pct:.1f}%)")

    print("\nSize bands:")
    bands = size_bands(chunks, config)
    print(f"  below target                {bands['below_target']:4d}  ({bands['below_target'] / len(chunks) * 100:.1f}%)")
    print(f"  target .. <90% of max       {bands['target_to_90pct_max']:4d}  ({bands['target_to_90pct_max'] / len(chunks) * 100:.1f}%)")
    print(f"  >= 90% of max                {bands['at_or_above_90pct_max']:4d}  ({bands['at_or_above_90pct_max'] / len(chunks) * 100:.1f}%)")

    trailing = chunks[-1]
    print(
        f"\nTrailing chunk: {trailing.token_count} tokens "
        f"({'below min_tokens, expected for a trailing chunk' if trailing.token_count < config.min_tokens else 'at or above min_tokens'})"
    )


def _report_quality_checks(
    segments: list[TranscriptSegment], cleaned: list[CleanedSegment], chunks: list[ChunkCandidate]
) -> None:
    print("\n=== Quality checks ===")

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


def _report_segment_ranges(chunks: list[ChunkCandidate], segments: list[TranscriptSegment]) -> None:
    print("\n=== Chunk source segment ranges (first/last contributing segment per chunk) ===")
    segment_by_id = segment_by_id_index(segments)
    for c in chunks:
        first, last = chunk_contributing_segments(c, segment_by_id)
        print(
            f"  Chunk {c.sequence_number + 1:03d}: "
            f"first=seq {first.sequence_number} ({first.id})  "
            f"last=seq {last.sequence_number} ({last.id})"
        )


def _report_boundaries(boundaries: list[BoundaryInfo]) -> None:
    print("\n=== Chunk boundaries ===")
    if not boundaries:
        print("(fewer than 2 chunks -- no boundaries to report)")
        return

    for b in boundaries:
        gap_str = "N/A (forced mid-segment split)" if b.gap_ms is None else f"{b.gap_ms}ms"
        print(
            f"  Boundary {b.prev_chunk_sequence + 1:03d} -> {b.next_chunk_sequence + 1:03d}: "
            f"prev_tokens={b.prev_token_count} next_tokens={b.next_token_count} "
            f"gap={gap_str} mechanism={b.mechanism} "
            f"prev_reached_target={b.prev_reached_target}"
        )
        print(
            f"      prev_last_segment=seq {b.prev_last_segment_sequence} ({b.prev_last_segment_id})  "
            f"next_first_segment=seq {b.next_first_segment_sequence} ({b.next_first_segment_id})"
        )

    print("\nBoundary mechanism summary:")
    summary = boundary_mechanism_summary(boundaries)
    total = len(boundaries)
    for mechanism, count in summary.items():
        print(f"  {mechanism:22} {count:4d}  ({count / total * 100:.1f}%)")


def _report_representative_chunks(
    chunks: list[ChunkCandidate], config: ChunkingConfig, *, preview_chars: int, full_text: bool
) -> None:
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

    near_max = [c for c in chunks if c.token_count >= config.max_tokens * 0.9]
    print(f"\nChunks within 90% of max_tokens ({config.max_tokens}): {len(near_max)}")
    for c in near_max[:10]:
        print(f"  Chunk {c.sequence_number + 1:03d}: {c.token_count} tokens")


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

    _report_chunk_summary(chunks, config)
    _report_quality_checks(segments, cleaned, chunks)
    _report_representative_chunks(chunks, config, preview_chars=args.preview_chars, full_text=args.full_text)
    _report_segment_ranges(chunks, segments)
    boundaries = compute_boundaries(chunks, segments, cleaned, config)
    _report_boundaries(boundaries)


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
