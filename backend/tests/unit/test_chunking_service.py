import uuid

import pytest

from app.services.chunking_service import (
    ChunkingConfig,
    chunk_transcript_segments,
    estimate_tokens,
)
from app.services.cleaning_service import CleanedSegment


def _seg(
    text: str,
    start_ms: int,
    duration_ms: int,
    *,
    cleaned_text: str | None = None,
    sequence_number: int = 0,
) -> CleanedSegment:
    return CleanedSegment(
        segment_id=uuid.uuid4(),
        sequence_number=sequence_number,
        text=cleaned_text if cleaned_text is not None else text,
        start_ms=start_ms,
        duration_ms=duration_ms,
    )


def _sequential(*texts_and_gaps: tuple[str, int]) -> list[CleanedSegment]:
    """Builds segments back-to-back in time, each followed by the given
    gap (ms) before the next one starts. texts_and_gaps is (text, gap_after_ms)."""
    segments = []
    cursor = 0
    for i, (text, gap_after) in enumerate(texts_and_gaps):
        duration = 900
        segments.append(_seg(text, cursor, duration, sequence_number=i))
        cursor += duration + gap_after
    return segments


# --- token estimator -----------------------------------------------------------


def test_estimate_tokens_empty() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_roughly_four_chars_per_token() -> None:
    assert estimate_tokens("a" * 40) == 10


def test_estimate_tokens_never_zero_for_nonempty() -> None:
    assert estimate_tokens("a") >= 1


# --- config validation -----------------------------------------------------------


def test_config_rejects_min_greater_than_target() -> None:
    with pytest.raises(ValueError, match="min_tokens"):
        ChunkingConfig(min_tokens=100, target_tokens=50, max_tokens=200)


def test_config_rejects_target_greater_than_max() -> None:
    with pytest.raises(ValueError, match="min_tokens"):
        ChunkingConfig(min_tokens=10, target_tokens=500, max_tokens=200)


def test_config_rejects_negative_overlap() -> None:
    with pytest.raises(ValueError, match="overlap_tokens"):
        ChunkingConfig(min_tokens=10, target_tokens=100, max_tokens=200, overlap_tokens=-1)


def test_config_rejects_overlap_at_least_target() -> None:
    with pytest.raises(ValueError, match="overlap_tokens"):
        ChunkingConfig(min_tokens=10, target_tokens=100, max_tokens=200, overlap_tokens=100)


# --- empty/invalid input ---------------------------------------------------------


def test_empty_segment_list_returns_no_chunks() -> None:
    config = ChunkingConfig(min_tokens=10, target_tokens=50, max_tokens=100)
    assert chunk_transcript_segments([], config) == []


def test_all_segments_clean_to_empty_returns_no_chunks() -> None:
    segments = [_seg("[Music]", 0, 900, cleaned_text="", sequence_number=0)]
    config = ChunkingConfig(min_tokens=10, target_tokens=50, max_tokens=100)
    assert chunk_transcript_segments(segments, config) == []


# --- basic accumulation, target/min/max ------------------------------------------


def test_single_short_segment_is_one_chunk_even_under_min() -> None:
    # A trailing/only chunk is allowed to be below min_tokens.
    segments = [_seg("hello there.", 0, 900, sequence_number=0)]
    config = ChunkingConfig(min_tokens=100, target_tokens=200, max_tokens=400)
    chunks = chunk_transcript_segments(segments, config)
    assert len(chunks) == 1
    assert chunks[0].text == "hello there."


def test_does_not_cut_before_target_even_at_sentence_boundary() -> None:
    # Each segment ends a sentence, but total tokens stay well under
    # target -- the chunker must keep accumulating, not cut early.
    segments = _sequential(("One.", 200), ("Two.", 200), ("Three.", 200))
    config = ChunkingConfig(min_tokens=1, target_tokens=1000, max_tokens=2000)
    chunks = chunk_transcript_segments(segments, config)
    assert len(chunks) == 1
    assert chunks[0].text == "One. Two. Three."


def test_cuts_at_sentence_boundary_when_no_room_to_look_for_a_pause() -> None:
    # A sentence boundary is only used once there's no room left (before
    # max_tokens) to keep looking for a stronger pause-based boundary --
    # see test_prefers_pause_boundary_over_nearer_sentence_boundary for
    # the case where a pause is available further ahead.
    filler = "word " * 20  # ~25 tokens, no terminal punctuation
    segments = [
        _seg(filler.strip() + ".", 0, 900, sequence_number=0),  # ends a sentence
        _seg("more words follow after this one", 900, 900, sequence_number=1),
    ]
    target = estimate_tokens(segments[0].text)  # reached exactly by segment 0
    # max == target: no room at all to scan past segment 0 looking for a
    # pause boundary, so the remembered sentence boundary is used.
    config = ChunkingConfig(min_tokens=1, target_tokens=target, max_tokens=target)
    chunks = chunk_transcript_segments(segments, config)
    assert len(chunks) == 2
    assert chunks[0].text == segments[0].text
    assert chunks[1].text == segments[1].text


def test_prefers_pause_boundary_over_nearer_sentence_boundary() -> None:
    # Found via manual quality inspection (tests/fixtures/podcast_transcript.py):
    # taking the first sentence-ending boundary past target ignored real
    # topic-transition pauses only a segment or two further ahead. This
    # pins the fix: a pause boundary within the max_tokens window wins
    # even though a sentence boundary appears first.
    segments = [
        _seg("First sentence ends here.", 0, 900, sequence_number=0),  # sentence, no pause
        _seg("Second sentence right after with barely a gap.", 1000, 900, sequence_number=1),
    ]
    # A real pause before segment 2 -- a likely topic transition.
    segments.append(_seg("A new topic begins after a long pause.", 1900 + 3000, 900, sequence_number=2))
    target = estimate_tokens(segments[0].text)
    config = ChunkingConfig(
        min_tokens=1,
        target_tokens=target,
        max_tokens=target * 10,
        pause_threshold_ms=1500,
    )

    chunks = chunk_transcript_segments(segments, config)

    assert len(chunks) == 2
    assert chunks[0].text == segments[0].text + " " + segments[1].text
    assert chunks[1].text == segments[2].text


def test_cuts_at_pause_gap_once_target_reached_without_sentence_end() -> None:
    segments = [
        _seg("this segment has no terminal punctuation", 0, 900, sequence_number=0),
    ]
    # A long pause after segment 0.
    segments.append(_seg("a new thought begins here", 900 + 3000, 900, sequence_number=1))
    target = estimate_tokens(segments[0].text)
    config = ChunkingConfig(
        min_tokens=1, target_tokens=target, max_tokens=target * 5, pause_threshold_ms=1500
    )
    chunks = chunk_transcript_segments(segments, config)
    assert len(chunks) == 2
    assert chunks[0].text == segments[0].text


def test_small_pause_below_threshold_is_not_treated_as_boundary() -> None:
    segments = [
        _seg("this segment has no terminal punctuation", 0, 900, sequence_number=0),
        _seg("and continues right after with barely a pause", 1100, 900, sequence_number=1),
    ]
    target = estimate_tokens(segments[0].text)
    config = ChunkingConfig(
        min_tokens=1, target_tokens=target, max_tokens=target * 10, pause_threshold_ms=1500
    )
    chunks = chunk_transcript_segments(segments, config)
    # 1100 - 900 = 200ms gap, well under the 1500ms threshold -> no cut here.
    assert len(chunks) == 1
    assert chunks[0].text == segments[0].text + " " + segments[1].text


def test_forces_cut_at_max_without_any_good_boundary() -> None:
    # No sentence punctuation anywhere, no meaningful pauses -- the
    # chunker must still stop at max_tokens rather than growing forever.
    segments = _sequential(
        ("alpha beta gamma delta", 100),
        ("epsilon zeta eta theta", 100),
        ("iota kappa lambda mu", 100),
        ("nu xi omicron pi", 100),
    )
    max_tokens = estimate_tokens(segments[0].text) + estimate_tokens(segments[1].text)
    config = ChunkingConfig(min_tokens=1, target_tokens=1, max_tokens=max_tokens)
    chunks = chunk_transcript_segments(segments, config)
    assert all(c.token_count <= max_tokens for c in chunks)
    assert len(chunks) > 1


# --- long uninterrupted section: safe splitting -----------------------------------


def test_long_uninterrupted_segment_is_split_on_sentence_boundaries() -> None:
    text = "First sentence here. Second sentence here. Third sentence here. Fourth one too."
    segments = [_seg(text, 0, 60_000, sequence_number=0)]
    max_tokens = estimate_tokens("First sentence here. Second sentence here.")
    config = ChunkingConfig(min_tokens=1, target_tokens=1, max_tokens=max_tokens)

    chunks = chunk_transcript_segments(segments, config)

    assert len(chunks) > 1
    assert all(c.token_count <= max_tokens for c in chunks)
    # No words lost or duplicated across the split.
    reconstructed = " ".join(c.text for c in chunks)
    assert reconstructed == text
    # Both resulting chunks trace back to the one original segment.
    for chunk in chunks:
        assert chunk.source_segment_ids == [segments[0].segment_id]


def test_single_long_sentence_with_no_punctuation_splits_on_words() -> None:
    text = " ".join(f"word{i}" for i in range(200))  # one giant "sentence"
    segments = [_seg(text, 0, 60_000, sequence_number=0)]
    config = ChunkingConfig(min_tokens=1, target_tokens=1, max_tokens=50)

    chunks = chunk_transcript_segments(segments, config)

    assert len(chunks) > 1
    assert all(c.token_count <= 50 for c in chunks)
    # Never splits mid-word.
    reconstructed_words = " ".join(c.text for c in chunks).split(" ")
    assert reconstructed_words == text.split(" ")


# --- source-segment traceability + timestamps -------------------------------------


def test_source_segment_ids_match_contributing_segments_in_order() -> None:
    segments = _sequential(("Alpha.", 200), ("Beta.", 200), ("Gamma.", 200))
    config = ChunkingConfig(min_tokens=1, target_tokens=1000, max_tokens=2000)
    chunks = chunk_transcript_segments(segments, config)
    assert len(chunks) == 1
    assert chunks[0].source_segment_ids == [s.segment_id for s in segments]


def test_start_ms_and_end_ms_use_segment_boundaries() -> None:
    segments = [
        _seg("Alpha.", 1_000, 900, sequence_number=0),
        _seg("Beta.", 2_500, 1_200, sequence_number=1),
    ]
    config = ChunkingConfig(min_tokens=1, target_tokens=1000, max_tokens=2000)
    chunks = chunk_transcript_segments(segments, config)
    assert len(chunks) == 1
    assert chunks[0].start_ms == 1_000
    assert chunks[0].end_ms == 2_500 + 1_200


def test_sequence_numbers_start_at_zero_and_increment() -> None:
    segments = _sequential(*[(f"Sentence {i}.", 3000) for i in range(6)])
    target = estimate_tokens("Sentence 0.")
    config = ChunkingConfig(min_tokens=1, target_tokens=target, max_tokens=target * 2)
    chunks = chunk_transcript_segments(segments, config)
    assert [c.sequence_number for c in chunks] == list(range(len(chunks)))
    assert len(chunks) > 1


# --- overlap ------------------------------------------------------------------------


def test_zero_overlap_produces_no_shared_segments_between_chunks() -> None:
    segments = _sequential(*[(f"Sentence number {i}.", 3000) for i in range(6)])
    target = estimate_tokens("Sentence number 0.")
    config = ChunkingConfig(
        min_tokens=1, target_tokens=target, max_tokens=target * 2, overlap_tokens=0
    )
    chunks = chunk_transcript_segments(segments, config)
    assert len(chunks) > 1
    seen: set = set()
    for chunk in chunks:
        for seg_id in chunk.source_segment_ids:
            assert seg_id not in seen, "segment repeated across chunks with overlap disabled"
            seen.add(seg_id)


def test_overlap_repeats_trailing_segment_in_next_chunk() -> None:
    segments = _sequential(*[(f"Sentence number {i}.", 3000) for i in range(6)])
    one_segment_tokens = estimate_tokens("Sentence number 0.")
    config = ChunkingConfig(
        min_tokens=1,
        target_tokens=one_segment_tokens * 2,
        max_tokens=one_segment_tokens * 3,
        overlap_tokens=one_segment_tokens,
    )
    chunks = chunk_transcript_segments(segments, config)
    assert len(chunks) > 1
    # Consecutive chunks should share at least the boundary segment.
    for earlier, later in zip(chunks, chunks[1:]):
        shared = set(earlier.source_segment_ids) & set(later.source_segment_ids)
        assert shared, "expected overlap to repeat a trailing segment in the next chunk"


# --- determinism -------------------------------------------------------------------


def test_chunking_is_deterministic_across_repeated_runs() -> None:
    segments = _sequential(*[(f"Sentence {i}.", 1800) for i in range(10)])
    target = estimate_tokens("Sentence 0.") * 2
    config = ChunkingConfig(min_tokens=1, target_tokens=target, max_tokens=target * 2)

    first_run = chunk_transcript_segments(segments, config)
    second_run = chunk_transcript_segments(segments, config)

    assert [c.__dict__ for c in first_run] == [c.__dict__ for c in second_run]
