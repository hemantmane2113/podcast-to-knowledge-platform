"""Tests for the pure calculation helpers in
scripts/validate_real_transcript.py -- the boundary-diagnostics fix from
the Phase 3A validation follow-up (negative gap_ms, forced-segment-split
detection, raw timestamp overlap detection, mechanism classification,
token distribution / size bands).

Only the pure functions are tested here (no network, no provider call) --
see the module docstring in validate_real_transcript.py for why the
boundary classification is a best-effort reconstruction, not a byte-exact
replay of the chunker's internal decisions.
"""

import uuid

from app.models.transcript_segment import TranscriptSegment
from app.services.chunking_service import ChunkingConfig, chunk_transcript_segments, estimate_tokens
from app.services.cleaning_service import clean_transcript_segments
from scripts.validate_real_transcript import (
    boundary_mechanism_summary,
    chunk_contributing_segments,
    compute_boundaries,
    find_raw_segment_overlaps,
    segment_by_id_index,
    size_bands,
    token_distribution,
)


def _raw_segment(
    text: str, start_ms: int, duration_ms: int, *, sequence_number: int = 0
) -> TranscriptSegment:
    return TranscriptSegment(
        id=uuid.uuid4(),
        transcript_id=uuid.uuid4(),
        sequence_number=sequence_number,
        text=text,
        start_ms=start_ms,
        duration_ms=duration_ms,
    )


# --- find_raw_segment_overlaps --------------------------------------------------


def test_no_overlap_when_segments_are_back_to_back() -> None:
    segments = [
        _raw_segment("a", 0, 1000, sequence_number=0),
        _raw_segment("b", 1000, 1000, sequence_number=1),
    ]
    assert find_raw_segment_overlaps(segments) == []


def test_no_overlap_when_there_is_a_gap() -> None:
    segments = [
        _raw_segment("a", 0, 1000, sequence_number=0),
        _raw_segment("b", 2000, 1000, sequence_number=1),
    ]
    assert find_raw_segment_overlaps(segments) == []


def test_detects_genuine_overlap_between_consecutive_segments() -> None:
    segments = [
        _raw_segment("a", 0, 5000, sequence_number=0),
        _raw_segment("b", 3000, 4000, sequence_number=1),
    ]
    overlaps = find_raw_segment_overlaps(segments)
    assert len(overlaps) == 1
    assert overlaps[0].prev_sequence == 0
    assert overlaps[0].next_sequence == 1
    assert overlaps[0].overlap_ms == 2000


def test_finds_every_overlapping_pair_not_just_the_first() -> None:
    segments = [
        _raw_segment("a", 0, 5000, sequence_number=0),
        _raw_segment("b", 3000, 5000, sequence_number=1),  # overlaps a by 2000ms
        _raw_segment("c", 7000, 1000, sequence_number=2),  # overlaps b by 1000ms
    ]
    overlaps = find_raw_segment_overlaps(segments)
    assert [(o.prev_sequence, o.next_sequence, o.overlap_ms) for o in overlaps] == [
        (0, 1, 2000),
        (1, 2, 1000),
    ]


# --- chunk_contributing_segments -------------------------------------------------


def test_chunk_contributing_segments_returns_first_and_last_by_sequence() -> None:
    segments = [
        _raw_segment("Alpha.", 0, 900, sequence_number=0),
        _raw_segment("Beta.", 900, 900, sequence_number=1),
        _raw_segment("Gamma.", 1800, 900, sequence_number=2),
    ]
    cleaned = clean_transcript_segments(segments)
    config = ChunkingConfig(min_tokens=1, target_tokens=1000, max_tokens=2000)
    chunks = chunk_transcript_segments(cleaned, config)
    assert len(chunks) == 1

    segment_by_id = segment_by_id_index(segments)
    first, last = chunk_contributing_segments(chunks[0], segment_by_id)
    assert first.sequence_number == 0
    assert last.sequence_number == 2


# --- compute_boundaries: gap calculation, no negative gaps from splits ----------


def test_boundary_gap_uses_real_segment_end_not_chunk_end_ms() -> None:
    # A real pause between two adjacent, non-split segments -- the basic
    # case the gap formula must get right.
    segments = [
        _raw_segment("First sentence ends here.", 0, 1000, sequence_number=0),
        _raw_segment("Second one starts after a pause.", 3000, 1000, sequence_number=1),
    ]
    cleaned = clean_transcript_segments(segments)
    target = estimate_tokens(cleaned[0].text)
    # max_tokens well above either segment's own size -- the pause
    # boundary is a strong signal that cuts immediately once target is
    # reached, regardless of remaining headroom, so this doesn't need to
    # be tight; it just needs to avoid forcing segment 1's own text to
    # split (that's a separate scenario, tested below).
    config = ChunkingConfig(min_tokens=1, target_tokens=target, max_tokens=target * 5, pause_threshold_ms=1500)
    chunks = chunk_transcript_segments(cleaned, config)
    assert len(chunks) == 2

    boundaries = compute_boundaries(chunks, segments, cleaned, config)
    assert len(boundaries) == 1
    b = boundaries[0]
    assert b.same_segment_split is False
    assert b.gap_ms == 2000  # 3000 - (0 + 1000)
    assert b.gap_ms >= 0


def test_forced_segment_split_boundary_has_no_negative_gap() -> None:
    # A single oversized segment forced to split across a chunk boundary
    # (chunking_service.py::_split_long_text) -- this is exactly the
    # scenario that previously produced negative gaps like -2800ms: the
    # same raw segment is both the "last" contributor of chunk N and the
    # "first" contributor of chunk N+1, so a naive gap calculation
    # subtracts a segment's own end from its own start.
    long_text = " ".join(f"word{i}" for i in range(400))
    segments = [
        _raw_segment("Intro sentence.", 0, 1000, sequence_number=0),
        _raw_segment(long_text, 1000, 60000, sequence_number=1),
        _raw_segment("Outro sentence.", 61000, 1000, sequence_number=2),
    ]
    cleaned = clean_transcript_segments(segments)
    config = ChunkingConfig(min_tokens=10, target_tokens=50, max_tokens=100)
    chunks = chunk_transcript_segments(cleaned, config)
    assert len(chunks) > 2  # confirms the long segment actually got split

    boundaries = compute_boundaries(chunks, segments, cleaned, config)
    split_boundaries = [b for b in boundaries if b.same_segment_split]
    assert split_boundaries, "expected at least one forced-segment-split boundary"
    for b in split_boundaries:
        assert b.gap_ms is None
        assert b.mechanism == "forced_segment_split"
        assert b.prev_last_segment_id == b.next_first_segment_id

    # No boundary anywhere reports a negative gap.
    for b in boundaries:
        assert b.gap_ms is None or b.gap_ms >= 0


def test_negative_gap_reflects_genuine_raw_timestamp_overlap() -> None:
    # When two DIFFERENT (non-split) consecutive segments genuinely
    # overlap in the raw data, the gap really is negative -- that's a
    # real data-quality signal, not a bug in the calculation, and must be
    # distinguishable from the forced-segment-split case above.
    # segment 1's text is deliberately short -- long enough that it isn't
    # itself split by max_tokens (that's the forced-segment-split
    # scenario, tested separately), short enough that max_tokens=target
    # (segment 0's own size) still forces the cut to land between the two
    # segments rather than merging both into one "end of transcript" chunk.
    segments = [
        _raw_segment("First segment here.", 0, 5000, sequence_number=0),
        _raw_segment("Bye.", 3000, 4000, sequence_number=1),
    ]
    cleaned = clean_transcript_segments(segments)
    target = estimate_tokens(cleaned[0].text)
    config = ChunkingConfig(min_tokens=1, target_tokens=target, max_tokens=target)
    chunks = chunk_transcript_segments(cleaned, config)
    assert len(chunks) == 2

    boundaries = compute_boundaries(chunks, segments, cleaned, config)
    b = boundaries[0]
    assert b.same_segment_split is False
    assert b.gap_ms == -2000
    assert find_raw_segment_overlaps(segments)  # confirmed by the overlap check too


def test_boundary_mechanism_pause_when_gap_at_or_above_threshold() -> None:
    segments = [
        _raw_segment("this segment has no terminal punctuation", 0, 900, sequence_number=0),
        _raw_segment("a new thought begins here", 900 + 3000, 900, sequence_number=1),
    ]
    cleaned = clean_transcript_segments(segments)
    target = estimate_tokens(cleaned[0].text)
    config = ChunkingConfig(
        min_tokens=1, target_tokens=target, max_tokens=target * 5, pause_threshold_ms=1500
    )
    chunks = chunk_transcript_segments(cleaned, config)
    assert len(chunks) == 2

    boundaries = compute_boundaries(chunks, segments, cleaned, config)
    assert boundaries[0].mechanism == "pause"
    assert boundaries[0].gap_ms == 3000


def test_boundary_mechanism_sentence_when_no_pause_but_ends_a_sentence() -> None:
    filler = "word " * 20
    segments = [
        _raw_segment(filler.strip() + ".", 0, 900, sequence_number=0),
        _raw_segment("more words follow after this one", 900, 900, sequence_number=1),
    ]
    cleaned = clean_transcript_segments(segments)
    target = estimate_tokens(cleaned[0].text)
    config = ChunkingConfig(min_tokens=1, target_tokens=target, max_tokens=target)
    chunks = chunk_transcript_segments(cleaned, config)
    assert len(chunks) == 2

    boundaries = compute_boundaries(chunks, segments, cleaned, config)
    assert boundaries[0].mechanism == "sentence"
    assert boundaries[0].gap_ms == 0


def test_prev_reached_target_reflects_finalized_chunk_token_count() -> None:
    segments = [
        _raw_segment("First sentence ends here.", 0, 1000, sequence_number=0),
        _raw_segment("Second one starts after a pause.", 3000, 1000, sequence_number=1),
    ]
    cleaned = clean_transcript_segments(segments)
    target = estimate_tokens(cleaned[0].text)
    config = ChunkingConfig(min_tokens=1, target_tokens=target, max_tokens=target, pause_threshold_ms=1500)
    chunks = chunk_transcript_segments(cleaned, config)
    boundaries = compute_boundaries(chunks, segments, cleaned, config)
    assert boundaries[0].prev_reached_target is True
    assert boundaries[0].prev_token_count == chunks[0].token_count
    assert boundaries[0].next_token_count == chunks[1].token_count


def test_no_boundaries_for_a_single_chunk() -> None:
    segments = [_raw_segment("hello there.", 0, 900, sequence_number=0)]
    cleaned = clean_transcript_segments(segments)
    config = ChunkingConfig(min_tokens=100, target_tokens=200, max_tokens=400)
    chunks = chunk_transcript_segments(cleaned, config)
    assert len(chunks) == 1
    assert compute_boundaries(chunks, segments, cleaned, config) == []


# --- boundary_mechanism_summary ---------------------------------------------------


def test_boundary_mechanism_summary_counts_each_mechanism() -> None:
    segments = [
        _raw_segment("First sentence ends here.", 0, 1000, sequence_number=0),
        _raw_segment("Second one starts after a pause.", 3000, 1000, sequence_number=1),
        _raw_segment("Third sentence follows immediately.", 4000, 1000, sequence_number=2),
    ]
    cleaned = clean_transcript_segments(segments)
    target = estimate_tokens(cleaned[0].text)
    config = ChunkingConfig(min_tokens=1, target_tokens=target, max_tokens=target, pause_threshold_ms=1500)
    chunks = chunk_transcript_segments(cleaned, config)
    boundaries = compute_boundaries(chunks, segments, cleaned, config)

    summary = boundary_mechanism_summary(boundaries)
    assert sum(summary.values()) == len(boundaries)
    assert "pause" in summary


# --- token_distribution -----------------------------------------------------------


def test_token_distribution_buckets_are_exhaustive_and_non_overlapping() -> None:
    from app.services.chunking_service import ChunkCandidate

    def _candidate(tokens: int) -> ChunkCandidate:
        return ChunkCandidate(
            sequence_number=0, text="x", start_ms=0, end_ms=0, source_segment_ids=[], token_count=tokens
        )

    chunks = [_candidate(t) for t in (100, 500, 699, 700, 799, 800, 899, 900, 999, 1000, 1079, 1080, 1199, 1200, 5000)]
    dist = token_distribution(chunks)

    assert sum(dist.values()) == len(chunks)
    assert dist["<500"] == 1  # 100
    assert dist["500-699"] == 2  # 500, 699
    assert dist["700-799"] == 2  # 700, 799
    assert dist["800-899"] == 2  # 800, 899
    assert dist["900-999"] == 2  # 900, 999
    assert dist["1000-1079"] == 2  # 1000, 1079
    assert dist["1080-1199"] == 2  # 1080, 1199
    assert dist["1200+"] == 2  # 1200, 5000


# --- size_bands --------------------------------------------------------------------


def test_size_bands_classifies_below_target_mid_and_near_max() -> None:
    from app.services.chunking_service import ChunkCandidate

    def _candidate(tokens: int) -> ChunkCandidate:
        return ChunkCandidate(
            sequence_number=0, text="x", start_ms=0, end_ms=0, source_segment_ids=[], token_count=tokens
        )

    config = ChunkingConfig(min_tokens=300, target_tokens=800, max_tokens=1200)
    # 90% of max_tokens=1200 is 1080.
    chunks = [_candidate(t) for t in (500, 800, 1000, 1079, 1080, 1200)]
    bands = size_bands(chunks, config)

    assert bands["below_target"] == 1  # 500
    assert bands["target_to_90pct_max"] == 3  # 800, 1000, 1079
    assert bands["at_or_above_90pct_max"] == 2  # 1080, 1200
    assert sum(bands.values()) == len(chunks)
