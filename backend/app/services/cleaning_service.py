"""Deterministic transcript cleaning (PRODUCT_SPEC.md §18, Phase 3A §3A.3).

Pure, rule-based text transforms only — no LLM, nothing probabilistic.
Every transform here is chosen to be unambiguously safe: it never removes
or reorders words that could be substantive speech, only formatting noise
and one well-defined transcription artifact (see `_dedupe_cross_segment_overlap`).

`TranscriptSegment.text` is raw provider output and is never touched or
persisted-over here. Cleaning output is a transient in-memory
`CleanedSegment` per segment, consumed directly by the chunker
(app/services/chunking_service.py) — it is never written back onto the
ORM model or persisted on its own, so there's no derived-state staleness
risk if the cleaning rules change (see ARCHITECTURE.md's raw-vs-clean
discussion).
"""

import re
from dataclasses import dataclass
from uuid import UUID

from app.models.transcript_segment import TranscriptSegment

# Bracketed non-speech markers some caption sources emit. Deliberately a
# small, explicit allowlist rather than "strip anything in brackets" --
# the latter risks deleting real spoken content that happens to be quoted
# or bracketed for other reasons.
_NON_SPEECH_MARKERS = re.compile(
    r"\[\s*(music|applause|laughter|inaudible|silence|crosstalk)\s*\]",
    re.IGNORECASE,
)

_WHITESPACE_RUN = re.compile(r"\s+")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.!?;:])")

# How many trailing/leading words to compare when looking for a rolling
# caption overlap between two consecutive segments.
_MAX_OVERLAP_WORDS_CHECKED = 10
_MIN_OVERLAP_WORDS_TO_TRIM = 3


def clean_segment_text(text: str) -> str:
    """Whitespace/punctuation/formatting normalization for one segment's
    text, in isolation (no knowledge of neighboring segments — see
    `_dedupe_cross_segment_overlap` for the one transform that needs that).
    """
    if not text:
        return ""

    cleaned = text.replace("\n", " ").replace("\t", " ")
    cleaned = _NON_SPEECH_MARKERS.sub(" ", cleaned)
    cleaned = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", cleaned)
    cleaned = _WHITESPACE_RUN.sub(" ", cleaned)
    return cleaned.strip()


def _dedupe_cross_segment_overlap(previous_cleaned: str, current_cleaned: str) -> str:
    """Trims a leading phrase from `current_cleaned` if it's an exact
    (case-insensitive) repeat of the end of `previous_cleaned`.

    This targets one specific, well-understood artifact: auto-generated
    captions are often produced from a rolling audio window, so the tail
    of one caption and the head of the next can be the literal same words
    re-transcribed. It is NOT used to remove disfluency-style repetition
    within a single segment ("the the", "very very") -- that could be a
    real speech pattern, not an artifact, so it's left untouched.

    Requires an exact match of at least `_MIN_OVERLAP_WORDS_TO_TRIM`
    consecutive words to act, specifically to avoid trimming a short,
    possibly-meaningful coincidental repeat (e.g. "you know you know").
    """
    if not previous_cleaned or not current_cleaned:
        return current_cleaned

    prev_words = previous_cleaned.split(" ")
    curr_words = current_cleaned.split(" ")

    max_check = min(_MAX_OVERLAP_WORDS_CHECKED, len(prev_words), len(curr_words))
    best_overlap = 0
    for n in range(max_check, _MIN_OVERLAP_WORDS_TO_TRIM - 1, -1):
        prev_tail = [w.lower() for w in prev_words[-n:]]
        curr_head = [w.lower() for w in curr_words[:n]]
        if prev_tail == curr_head:
            best_overlap = n
            break

    if best_overlap == 0:
        return current_cleaned

    return " ".join(curr_words[best_overlap:]).strip()


@dataclass(frozen=True)
class CleanedSegment:
    """One segment's deterministically-cleaned text, transient and never
    persisted on its own -- the chunker (chunking_service.py) consumes
    these directly and only its output (Chunk, via source_segment_ids)
    is written to the database. Carries `segment_id` so that traceability
    survives the cleaning step.
    """

    segment_id: UUID
    sequence_number: int
    text: str
    start_ms: int
    duration_ms: int


def clean_transcript_segments(segments: list[TranscriptSegment]) -> list[CleanedSegment]:
    """Returns one CleanedSegment per input segment, in transcript order.
    Pure function of each segment's `.text` (plus the immediately
    preceding segment's already-cleaned text, for overlap trimming) --
    running this twice on the same input produces the same output, which
    is what makes reprocessing safe (PRODUCT_SPEC.md §51/§75 idempotency).

    Never mutates the input ORM objects.
    """
    cleaned_segments: list[CleanedSegment] = []
    previous_cleaned = ""
    for segment in segments:
        cleaned = clean_segment_text(segment.text)
        cleaned = _dedupe_cross_segment_overlap(previous_cleaned, cleaned)
        cleaned_segments.append(
            CleanedSegment(
                segment_id=segment.id,
                sequence_number=segment.sequence_number,
                text=cleaned,
                start_ms=segment.start_ms,
                duration_ms=segment.duration_ms,
            )
        )
        # An empty result (e.g. a segment that was only "[Music]") still
        # updates previous_cleaned to "" so a following segment isn't
        # compared against stale, no-longer-adjacent text.
        previous_cleaned = cleaned
    return cleaned_segments
