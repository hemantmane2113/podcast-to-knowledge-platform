import uuid

from app.services.cleaning_service import (
    clean_segment_text,
    clean_transcript_segments,
)


def _segment(text: str, sequence_number: int = 0):
    from app.models.transcript_segment import TranscriptSegment

    return TranscriptSegment(
        id=uuid.uuid4(),
        transcript_id=uuid.uuid4(),
        sequence_number=sequence_number,
        text=text,
        start_ms=sequence_number * 1000,
        duration_ms=900,
    )


# --- whitespace normalization -------------------------------------------------


def test_collapses_multiple_spaces() -> None:
    assert clean_segment_text("hello    world") == "hello world"


def test_collapses_tabs_and_newlines() -> None:
    assert clean_segment_text("hello\tworld\nfoo") == "hello world foo"


def test_strips_leading_and_trailing_whitespace() -> None:
    assert clean_segment_text("   hello world   ") == "hello world"


def test_empty_and_whitespace_only_text() -> None:
    assert clean_segment_text("") == ""
    assert clean_segment_text("   ") == ""


# --- safe punctuation/formatting normalization --------------------------------


def test_removes_space_before_punctuation() -> None:
    assert clean_segment_text("hello , world .") == "hello, world."


def test_strips_known_non_speech_markers() -> None:
    assert clean_segment_text("that was great [Music] right?") == "that was great right?"
    assert clean_segment_text("[Applause] thank you") == "thank you"
    assert clean_segment_text("[INAUDIBLE] I think so") == "I think so"


def test_unknown_bracketed_content_is_preserved() -> None:
    # Only the small documented allowlist is stripped -- arbitrary bracketed
    # text could be real spoken/quoted content, so it must survive.
    assert clean_segment_text("he said [laughing nervously] hello") == (
        "he said [laughing nervously] hello"
    )


# --- preservation of meaning: things cleaning must NOT do ----------------------


def test_does_not_remove_within_segment_word_repetition() -> None:
    # Could be a real disfluency/emphasis ("no no", "very very"), not
    # necessarily an artifact -- must not be touched.
    assert clean_segment_text("no no I don't think so") == "no no I don't think so"
    assert clean_segment_text("it was very very good") == "it was very very good"


def test_does_not_alter_numbers_or_abbreviations() -> None:
    assert clean_segment_text("that costs $3.14 per unit") == "that costs $3.14 per unit"
    assert clean_segment_text("see e.g. the appendix") == "see e.g. the appendix"


def test_does_not_reorder_or_drop_words() -> None:
    original = "the guest argues that scaling is not merely about size"
    assert clean_segment_text(original) == original


# --- cross-segment rolling-caption overlap trimming -----------------------------


def test_trims_exact_rolling_caption_overlap_across_segments() -> None:
    segments = [
        _segment("scaling laws have changed how we think about", 0),
        _segment("how we think about model training", 1),
    ]
    clean_transcript_segments(segments)
    assert segments[0].cleaned_text == "scaling laws have changed how we think about"
    assert segments[1].cleaned_text == "model training"


def test_short_overlap_below_threshold_is_not_trimmed() -> None:
    # Only 2 words overlap ("of course") -- below the 3-word minimum, so
    # this could plausibly be a real, separate utterance and is preserved.
    segments = [
        _segment("that's true of course", 0),
        _segment("of course we should check", 1),
    ]
    clean_transcript_segments(segments)
    assert segments[0].cleaned_text == "that's true of course"
    assert segments[1].cleaned_text == "of course we should check"


def test_no_overlap_leaves_segments_unchanged() -> None:
    segments = [
        _segment("first topic entirely", 0),
        _segment("second unrelated topic", 1),
    ]
    clean_transcript_segments(segments)
    assert segments[0].cleaned_text == "first topic entirely"
    assert segments[1].cleaned_text == "second unrelated topic"


def test_clean_transcript_segments_preserves_order_and_count() -> None:
    segments = [_segment(f"segment number {i}", i) for i in range(5)]
    clean_transcript_segments(segments)
    assert len(segments) == 5
    for i, segment in enumerate(segments):
        assert segment.cleaned_text == f"segment number {i}"


def test_raw_text_is_never_mutated() -> None:
    segments = [_segment("  hello   [Music] world  ", 0)]
    clean_transcript_segments(segments)
    assert segments[0].text == "  hello   [Music] world  "
    assert segments[0].cleaned_text == "hello world"
