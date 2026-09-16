"""End-to-end cleaner + chunker checks against the realistic long-form
fixture (Phase 3A §3A.12/§3A.14) — catches structural regressions that
small hand-built unit tests might miss (e.g. the pause-vs-sentence
boundary preference bug found by manually inspecting this fixture's
output, see chunking_service.py's chunk_transcript_segments).
"""

from app.services.chunking_service import ChunkingConfig, chunk_transcript_segments
from app.services.cleaning_service import clean_transcript_segments
from tests.fixtures.podcast_transcript import SEGMENT_SCRIPT, build_transcript_segments


def _default_config() -> ChunkingConfig:
    return ChunkingConfig(min_tokens=300, target_tokens=800, max_tokens=1200)


def test_fixture_cleans_without_losing_or_duplicating_words() -> None:
    segments = build_transcript_segments()
    cleaned = clean_transcript_segments(segments)

    original_word_count = sum(len(text.split()) for text, _ in SEGMENT_SCRIPT)
    cleaned_word_count = sum(len(s.text.split()) for s in cleaned)
    # Equal, not just close: this fixture has no [Music]-style markers or
    # deliberate rolling-caption overlaps, so cleaning should be a no-op
    # on word count (only whitespace/punctuation formatting changes).
    assert cleaned_word_count == original_word_count


def test_fixture_produces_no_oversized_chunks() -> None:
    segments = build_transcript_segments()
    cleaned = clean_transcript_segments(segments)
    config = _default_config()

    chunks = chunk_transcript_segments(cleaned, config)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.token_count <= config.max_tokens


def test_fixture_chunks_are_sequential_with_no_gaps() -> None:
    segments = build_transcript_segments()
    cleaned = clean_transcript_segments(segments)
    chunks = chunk_transcript_segments(cleaned, _default_config())

    assert [c.sequence_number for c in chunks] == list(range(len(chunks)))


def test_fixture_chunk_text_reconstructs_every_word_exactly_once() -> None:
    segments = build_transcript_segments()
    cleaned = clean_transcript_segments(segments)
    chunks = chunk_transcript_segments(cleaned, _default_config())

    reconstructed_words = " ".join(c.text for c in chunks).split()
    original_words = " ".join(s.text for s in cleaned).split()
    assert reconstructed_words == original_words


def test_fixture_long_uninterrupted_segment_is_traceable_and_intact() -> None:
    segments = build_transcript_segments()
    cleaned = clean_transcript_segments(segments)
    # Tight max_tokens forces the long uninterrupted segment to actually
    # be split across multiple chunks (see chunking_service.py's
    # _split_long_text) rather than fitting in one.
    config = ChunkingConfig(min_tokens=20, target_tokens=60, max_tokens=90)

    chunks = chunk_transcript_segments(cleaned, config)

    long_segment = next(s for s in cleaned if s.text.startswith("and honestly"))
    touching_chunks = [c for c in chunks if long_segment.segment_id in c.source_segment_ids]
    assert len(touching_chunks) > 1  # confirms the split actually happened

    reconstructed = " ".join(c.text for c in touching_chunks)
    assert long_segment.text in reconstructed


def test_fixture_every_chunk_has_valid_source_segments() -> None:
    segments = build_transcript_segments()
    cleaned = clean_transcript_segments(segments)
    chunks = chunk_transcript_segments(cleaned, _default_config())

    all_segment_ids = {s.segment_id for s in cleaned}
    for chunk in chunks:
        assert chunk.source_segment_ids  # never empty
        assert set(chunk.source_segment_ids) <= all_segment_ids
        assert chunk.start_ms <= chunk.end_ms


def test_fixture_chunking_is_deterministic() -> None:
    segments_a = build_transcript_segments()
    cleaned_a = clean_transcript_segments(segments_a)
    segments_b = build_transcript_segments()
    cleaned_b = clean_transcript_segments(segments_b)
    config = _default_config()

    chunks_a = chunk_transcript_segments(cleaned_a, config)
    chunks_b = chunk_transcript_segments(cleaned_b, config)

    assert len(chunks_a) == len(chunks_b)
    assert [c.text for c in chunks_a] == [c.text for c in chunks_b]
    assert [c.token_count for c in chunks_a] == [c.token_count for c in chunks_b]
