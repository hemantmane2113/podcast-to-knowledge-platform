"""Semantic/hybrid transcript chunking (PRODUCT_SPEC.md §20-21, Phase 3A §3A.4-3A.9).

No embeddings, no LLM, no external calls — a deterministic function of
already-cleaned transcript segments plus size configuration. See
ARCHITECTURE.md for the algorithm writeup and why embeddings were
deliberately not used here (Phase 3C, once an EmbeddingProvider exists).

Algorithm, in one paragraph: walk segments in order, accumulating an
approximate token count. Once the target size is reached, look at the
current segment boundary for two "semantic" signals -- does the text end a
sentence, and was there a real pause before the next segment -- and cut
there if either is present; otherwise keep accumulating until either a
good boundary appears or the hard max is hit. A single segment whose own
text alone exceeds the max is split on sentence, then word, boundaries
(never mid-word) so it never produces an oversized chunk by itself.
Optional configurable overlap repeats trailing segments at the start of
the next chunk; source_segment_ids and start_ms/end_ms are always derived
from real segment boundaries, never interpolated.
"""

import re
import uuid
from dataclasses import dataclass, field

from app.config.settings import Settings
from app.models.transcript_segment import TranscriptSegment

_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]?$")


def estimate_tokens(text: str) -> int:
    """Approximate token count (~4 characters/token for English), NOT a
    real tokenizer. No concrete LLM/tokenizer has been chosen yet (that's
    Phase 4), so this exists purely to drive consistent, self-referential
    chunk sizing — do not treat it as an authoritative token count for
    billing or context-window math against any real model.
    """
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def _ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(text.rstrip()))


@dataclass(frozen=True)
class ChunkingConfig:
    target_tokens: int = 800
    min_tokens: int = 300
    max_tokens: int = 1200
    overlap_tokens: int = 0
    pause_threshold_ms: int = 1500

    def __post_init__(self) -> None:
        if not (0 < self.min_tokens <= self.target_tokens <= self.max_tokens):
            raise ValueError(
                "ChunkingConfig requires 0 < min_tokens <= target_tokens <= max_tokens "
                f"(got min={self.min_tokens}, target={self.target_tokens}, max={self.max_tokens})"
            )
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be >= 0")
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be smaller than target_tokens")

    @classmethod
    def from_settings(cls, settings: Settings) -> "ChunkingConfig":
        return cls(
            target_tokens=settings.chunk_target_tokens,
            min_tokens=settings.chunk_min_tokens,
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            pause_threshold_ms=settings.chunk_pause_threshold_ms,
        )


@dataclass
class ChunkCandidate:
    """A chunker output, not yet persisted — app/repositories/chunk_repository.py
    turns these into Chunk rows."""

    sequence_number: int
    text: str
    start_ms: int
    end_ms: int
    source_segment_ids: list[uuid.UUID]
    token_count: int


@dataclass
class _Unit:
    """One piece of chunkable text. Normally one Unit == one segment; a
    segment whose text alone exceeds max_tokens produces several Units
    (see _split_long_text), all sharing that segment's id/timing.
    """

    text: str
    token_count: int
    segment_id: uuid.UUID
    segment_start_ms: int
    segment_end_ms: int
    is_last_piece_of_segment: bool
    # True for every piece of a forced multi-piece split except the last.
    # Each such piece was already sized up to max_tokens on its own, so the
    # chunk must close immediately after it -- otherwise two individually
    # safe pieces could be accumulated together past max_tokens, which is
    # exactly what happened before this field existed (see git history).
    forces_chunk_boundary: bool = field(default=False)
    pause_after_ms: int | None = field(default=None)
    ends_sentence: bool = field(default=False)


def _split_by_words(text: str, max_tokens: int) -> list[str]:
    # Measures the actual joined candidate string each time, rather than
    # summing per-word estimate_tokens() calls: the latter silently
    # undercounts (misses space characters, and each word's own max(1, ..)
    # floor compounds), which let a joined piece exceed max_tokens even
    # though its parts' estimates summed to less than max_tokens.
    words = text.split(" ")
    pieces: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = current + [word]
        if current and estimate_tokens(" ".join(candidate)) > max_tokens:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current = candidate
    if current:
        pieces.append(" ".join(current))
    return pieces or [text]


def _split_long_text(text: str, max_tokens: int) -> list[str]:
    """Splits text whose own token estimate exceeds max_tokens into safe
    pieces, preferring sentence boundaries and falling back to word
    boundaries. Never splits a word (§3A.4: "split it safely")."""
    if estimate_tokens(text) <= max_tokens:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1:
        return _split_by_words(text, max_tokens)

    pieces = []
    current = []
    for sentence in sentences:
        candidate = current + [sentence]
        if current and estimate_tokens(" ".join(candidate)) > max_tokens:
            pieces.append(" ".join(current))
            current = [sentence]
        else:
            current = candidate
    if current:
        pieces.append(" ".join(current))

    # A single sentence can itself still exceed max_tokens.
    final_pieces: list[str] = []
    for piece in pieces:
        if estimate_tokens(piece) > max_tokens:
            final_pieces.extend(_split_by_words(piece, max_tokens))
        else:
            final_pieces.append(piece)
    return final_pieces


def _build_units(segments: list[TranscriptSegment], max_tokens: int) -> list[_Unit]:
    units: list[_Unit] = []
    for index, segment in enumerate(segments):
        text = segment.cleaned_text if segment.cleaned_text is not None else segment.text
        if not text:
            continue

        segment_end_ms = segment.start_ms + segment.duration_ms
        pause_after_ms = None
        if index + 1 < len(segments):
            pause_after_ms = segments[index + 1].start_ms - segment_end_ms

        pieces = _split_long_text(text, max_tokens)
        for piece_index, piece in enumerate(pieces):
            is_last = piece_index == len(pieces) - 1
            units.append(
                _Unit(
                    text=piece,
                    token_count=estimate_tokens(piece),
                    segment_id=segment.id,
                    segment_start_ms=segment.start_ms,
                    segment_end_ms=segment_end_ms,
                    is_last_piece_of_segment=is_last,
                    forces_chunk_boundary=len(pieces) > 1 and not is_last,
                    pause_after_ms=pause_after_ms if is_last else None,
                    ends_sentence=_ends_sentence(piece) if is_last else False,
                )
            )
    return units


def _is_pause_boundary(unit: _Unit, config: ChunkingConfig) -> bool:
    return (
        unit.is_last_piece_of_segment
        and unit.pause_after_ms is not None
        and unit.pause_after_ms >= config.pause_threshold_ms
    )


def _is_sentence_boundary(unit: _Unit) -> bool:
    return unit.is_last_piece_of_segment and unit.ends_sentence


def _finalize_chunk(units: list[_Unit], sequence_number: int) -> ChunkCandidate:
    text = " ".join(u.text for u in units if u.text)
    source_segment_ids: list[uuid.UUID] = []
    seen: set[uuid.UUID] = set()
    for unit in units:
        if unit.segment_id not in seen:
            source_segment_ids.append(unit.segment_id)
            seen.add(unit.segment_id)

    return ChunkCandidate(
        sequence_number=sequence_number,
        text=text,
        start_ms=units[0].segment_start_ms,
        end_ms=units[-1].segment_end_ms,
        source_segment_ids=source_segment_ids,
        # Measured on the actual joined text, not summed from per-unit
        # estimates -- summing undercounts (misses the join-space
        # characters, and each unit's own floor compounds); see
        # _split_by_words for the same fix applied to word-splitting.
        token_count=estimate_tokens(text),
    )


def chunk_transcript_segments(
    segments: list[TranscriptSegment], config: ChunkingConfig
) -> list[ChunkCandidate]:
    """Segments must already be in transcript order (sequence_number
    ascending) and already cleaned (cleaned_text populated) — this
    function does not sort or clean.
    """
    units = _build_units(segments, config.max_tokens)
    if not units:
        return []

    chunks: list[ChunkCandidate] = []
    sequence_number = 0
    index = 0
    total = len(units)

    while index < total:
        current_units: list[_Unit] = []
        # The best cut point found so far, as a unit count into
        # current_units (i.e. "stop after this many units"). A pause
        # boundary is preferred and, once found past target, taken
        # immediately -- it's the closest thing to a real topic/thought
        # boundary this heuristic has. A sentence boundary is weaker (most
        # segments end with one, so treating it as equally good would mean
        # almost always cutting at the very first one past target, ignoring
        # a real pause boundary that might be only a few segments further
        # -- this was a real quality bug, found by manually inspecting
        # chunks from tests/fixtures/podcast_transcript.py: topic
        # transitions were being ignored in favor of the nearest comma-
        # adjacent period). So a sentence boundary is only *remembered* as
        # a fallback and scanning continues, hoping for a pause boundary,
        # until max_tokens forces a decision.
        best_cut_at: int | None = None
        best_cut_is_pause = False

        while index < total:
            unit = units[index]

            # Peek before adding: every individual unit is already
            # guaranteed <= max_tokens on its own (via _split_long_text),
            # so if adding this one would push the running total over
            # max_tokens, close the chunk now and leave it for the next
            # one, rather than adding it and only then discovering the
            # chunk overshot (checking *after* adding can only ever catch
            # an overshoot, never prevent it).
            candidate_units = current_units + [unit]
            candidate_tokens = estimate_tokens(
                " ".join(u.text for u in candidate_units if u.text)
            )
            if current_units and candidate_tokens > config.max_tokens:
                break

            current_units = candidate_units
            current_tokens = candidate_tokens
            index += 1

            if index >= total:
                best_cut_at = len(current_units)  # end of transcript -- take everything
                break
            if unit.forces_chunk_boundary:
                best_cut_at = len(current_units)  # already sized up to max_tokens alone
                break

            if current_tokens >= config.target_tokens:
                if _is_pause_boundary(unit, config):
                    best_cut_at = len(current_units)
                    best_cut_is_pause = True
                    break  # the strong signal -- stop scanning, use it
                if _is_sentence_boundary(unit) and not best_cut_is_pause:
                    best_cut_at = len(current_units)  # remembered as a fallback only

            if current_tokens >= config.max_tokens:
                break  # hard ceiling -- stop scanning regardless of boundary quality

        if best_cut_at is None:
            # Nothing matched (no sentence end, no pause) before max_tokens
            # forced a stop -- take everything accumulated, same as a
            # forced mid-text split.
            best_cut_at = len(current_units)
        elif best_cut_at < len(current_units):
            # Roll back to the remembered fallback (a sentence boundary
            # found earlier than where scanning actually stopped) -- the
            # units after it go back into the pool for the next chunk.
            index -= len(current_units) - best_cut_at
            current_units = current_units[:best_cut_at]

        chunks.append(_finalize_chunk(current_units, sequence_number))
        sequence_number += 1

        if config.overlap_tokens > 0 and index < total:
            overlap_tokens = 0
            step_back = 0
            for unit in reversed(current_units):
                candidate = overlap_tokens + unit.token_count
                if candidate > config.overlap_tokens:
                    break
                overlap_tokens = candidate
                step_back += 1
            index -= step_back

    return chunks
