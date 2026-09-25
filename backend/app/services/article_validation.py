"""Deterministic article validation (Phase H). Pure Python, no LLM and no
DB session -- takes already-loaded rows and plain data, returns a report.
This is the reliability backbone of Phase H: an optional LLM-based
evaluator (app/ai/nodes/validation.py, gated by
Settings.enable_llm_validation) is an additional, clearly-separate signal
appended to `checks`, never a replacement for any check here.

Each check is independently callable and independently testable
(tests/unit/test_article_validation.py) -- run_validation() just runs all
of them and folds the result into one report.

Every check reports a `severity`, not just a bool -- this module identifies
*measurable or strongly suspicious* failure modes, not prose quality:

- FAILURE: an objective integrity/structural violation (a broken
  provenance chain, an invalid reference, a missing/empty/duplicate
  section, exceeding the hard length ceiling). These are what
  ValidationReport.passed and app/ai/graph.py's deterministic revision
  trigger key off, unchanged from before this severity distinction
  existed -- a WARNING never blocks the pipeline or forces revision on
  its own.
- WARNING: a probable editorial concern (repetition, similarity,
  templated openings, a soft length-target miss) worth a human or the
  optional editorial review's attention, but not proof the article is
  broken -- these checks are deliberately conservative (see each check's
  own docstring) because legitimate writing repeats names, concepts, and
  qualifications.
- PASS: nothing worth flagging.

`CheckResult.passed` stays a plain bool (True unless severity is FAILURE)
so every existing consumer that only ever read `.passed` -- revision.py's
regex-based section targeting, the persisted ValidationResult.passed
column, API responses -- keeps working unchanged; `severity` is purely
additive information for anyone who wants finer detail.
"""

import re
import uuid
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app.models.article_plan import ArticlePlan
from app.models.article_section import ArticleSection
from app.models.chunk import Chunk
from app.models.topic import Topic

_MIN_SECTION_CONTENT_CHARS = 50
_MIN_DUPLICATE_PARAGRAPH_CHARS = 40
_MIN_ARTICLE_WORDS = 100

# Generation-artifact detection: the model's structured-output response
# occasionally leaks meta-commentary about its own generation/validation
# process into what should be article prose -- e.g. "paragraphs
# continuation error"/"paragraphs continuation invalid", observed in a
# real generated article. Root cause: GeneratedSection's own Pydantic
# validator (app/ai/schemas.py) previously only checked that paragraphs
# were non-empty, never that they read as plausible prose -- a
# syntactically valid response with garbage content sailed straight
# through structured-output validation on the FIRST attempt, never
# triggering ChatCompletionsProvider.generate_structured's existing
# corrective-retry loop (app/providers/llm/_chat_completions.py) at all.
# GeneratedSection's validator now imports generation_artifact_match from
# here and rejects the same patterns immediately, at the point of
# generation -- that DOES trigger the existing retry, so this is the
# primary fix. This module's own use of the same patterns (via
# is_section_content_valid/check_no_empty_sections below) is the final
# deterministic safety net for anything that slips through anyway (a
# pattern this list doesn't anticipate, or content persisted before this
# fix existed) -- never the primary defense, per the explicit requirement
# not to merely strip artifact text as a first resort.
_GENERATION_ARTIFACT_PATTERNS = (
    re.compile(r"\bparagraphs?\s+continuation\b", re.IGNORECASE),
    # A second, real generated article (post-fix) surfaced a different word
    # form of the same failure mode: "paragraphs continued?" -- the model
    # narrating whether it should keep writing, rather than "paragraphs
    # continuation [error|invalid]" narrating a validation failure. Same
    # meta-commentary-about-the-response category, just a different verb
    # form the original pattern didn't cover. Deliberately requires
    # "continued" immediately after "paragraph(s)" (not just "continued"
    # anywhere) so it can never match unrelated prose that happens to use
    # that word (e.g. "the tradition continued for decades").
    re.compile(r"\bparagraphs?\s+continued\b", re.IGNORECASE),
    # A third real generated article surfaced yet another word form of the
    # same category: the model directly narrating a validation verdict
    # about its own output ("paragraphs are invalid", "paragraph is not
    # allowed") instead of writing prose. Requires "paragraph(s)" directly
    # followed by "is"/"are" and then "invalid" or "not allowed" -- so it
    # can never fire on legitimate prose that separately uses "paragraphs
    # are ..." with an unrelated predicate ("are not always easy to
    # follow", "are shorter than others") or "invalid"/"not allowed" about
    # something else entirely ("paragraph is valid", "rules are not
    # allowed to override the evidence").
    re.compile(r"\bparagraphs?\s+(?:is|are)\s+(?:invalid|not\s+allowed)\b", re.IGNORECASE),
    re.compile(r"\bcontinuation\s+(?:error|invalid|failed)\b", re.IGNORECASE),
    re.compile(r"\b(?:json|schema)\s+validation\s+(?:error|failed)\b", re.IGNORECASE),
    re.compile(r"\bas an ai (?:language model|assistant)\b", re.IGNORECASE),
    re.compile(r"\bi cannot (?:fulfill|comply with|generate|complete) (?:this|that|the)\b", re.IGNORECASE),
    re.compile(r'"(?:heading|paragraphs)"\s*:', re.IGNORECASE),  # raw JSON leaking into prose
    re.compile(r"```"),  # a markdown code fence should never appear in article prose
)


def generation_artifact_match(text: str) -> str | None:
    """Returns the first matched artifact pattern's exact text, or None if
    `text` reads as plausible article prose -- never a full "is this good
    prose" judgment, just a narrow, conservative match against known
    failure signatures, deliberately unlikely to ever match legitimate
    editorial writing about a podcast. Shared by
    app/ai/schemas.py::GeneratedSection (rejects at the structured-output
    boundary, per paragraph) and is_section_content_valid below (the
    final deterministic safety net, on the whole section's content)."""
    for pattern in _GENERATION_ARTIFACT_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


# Mojibake repair: a real generated article contained "Hubermanâs",
# "âsomething hereâ" etc., where the source (either the transcript text a
# section quotes, or the model's own output) already contains classic
# UTF-8-decoded-as-Windows-1252 mojibake -- e.g. a right single quote (U+2019,
# UTF-8 bytes E2 80 99) misread one byte at a time as cp1252 becomes "â€™".
# Investigated and ruled out as a bug in THIS codebase's own I/O before
# adding this: there is no manual `.encode(`/`.decode(` or charset override
# anywhere in backend/app -- httpx (LLM providers + Supadata transcript
# fetches), asyncpg (DB), Starlette's JSONResponse, and the frontend's plain
# `fetch().json()` are all UTF-8-correct by default and none of them is
# overridden. That rules out every serialization/transport boundary this
# codebase controls, which leaves the raw text itself (as returned by the
# LLM, possibly echoing already-corrupted transcript text) as the only
# remaining source -- the same category of problem as
# generation_artifact_match above, fixed at the same generation boundary
# (app/ai/schemas.py::GeneratedSection) rather than downstream.
#
# Two single-byte codecs are tried, cp1252 and latin-1 (ISO-8859-1), since
# either is a plausible real-world mis-decode and they're mutually
# exclusive by construction: cp1252 repurposes bytes 0x80-0x9F for visible
# punctuation (a right single quote comes back as "â€™" -- three visible
# characters), while latin-1 maps those same bytes to the C1 control range
# (the SAME right single quote comes back as "â" followed by two invisible
# control characters -- "â" alone is what a human would actually see or
# retype, matching the real report this was investigated from). Encoding
# `text` under whichever codec did NOT originally produce it always raises
# (proven experimentally: each codec's encode step rejects codepoints only
# the other one's decode step can produce), so trying both in sequence
# never risks the wrong repair being silently applied.
#
# The round trip (`text -> {cp1252,latin-1} bytes -> utf-8`) is
# self-validating, not a blind character-replacement table: it only ever
# changes `text` when re-decoding those bytes as UTF-8 actually succeeds,
# which happens only when the text truly was UTF-8 mis-decoded as one of
# these two codecs in the first place -- ordinary prose, including
# genuinely accented names/words ("François", "café"), fails both
# round trips (verified) and is returned unchanged.
def repair_mojibake(text: str) -> str:
    """Reverses UTF-8-decoded-as-cp1252 or UTF-8-decoded-as-latin-1
    mojibake if `text` shows the pattern, otherwise returns `text`
    unchanged."""
    for source_codec in ("cp1252", "latin-1"):
        try:
            candidate = text.encode(source_codec).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        return candidate
    return text

# Relative-shortness signal for check_no_empty_sections (requirement #5):
# a section under both the ratio AND the absolute floor relative to the
# article's own mean section length is "implausibly short" -- requiring
# BOTH keeps a normal short intro/conclusion (common, legitimate) from
# tripping this on its own, and only evaluated when there are enough
# sections to make "the rest of the article" a meaningful baseline.
_RELATIVE_SHORT_SECTION_MIN_SECTIONS = 3
_RELATIVE_SHORT_SECTION_RATIO = 0.2
_RELATIVE_SHORT_SECTION_ABS_WORDS = 100

# check_no_duplicate_paragraphs: near-duplicate detection threshold
# (difflib ratio, stdlib, no new dependency).
_NEAR_DUPLICATE_PARAGRAPH_RATIO = 0.85

# check_excessive_phrase_repetition: a 6-word window is long enough that a
# name, technical term, or short recurring phrase can never trigger it on
# its own -- only a genuinely repeated multi-word run can. "Across
# sections" is enforced by counting DISTINCT sections a phrase appears in
# (deduped per section first), not raw occurrence count, so a phrase used
# several times within one section's own argument doesn't count.
_PHRASE_NGRAM_WORDS = 6
_PHRASE_MIN_SECTIONS = 3
_PHRASE_MAX_EXAMPLES = 3

# check_repeated_section_openings: only fires when at least this many
# sections share a near-identical opening -- two sections legitimately
# sharing an attribution/template opening is normal, three or more
# repeating it verbatim is the "clearly excessive" case the requirement
# asks for.
_OPENING_WINDOW_WORDS = 8
_OPENING_SIMILARITY_RATIO = 0.8
_OPENING_MIN_GROUP_SIZE = 3

# check_similar_sections: Jaccard similarity of each section's set of
# 6-word shingles (reuses the same n-gram unit as the phrase-repetition
# check) -- a lightweight, deterministic, already-stdlib-only technique,
# never embeddings/vector search. 0.3 (30% shared 6-word runs) is
# conservative on purpose: two sections discussing the same broader
# concept from different angles share vocabulary but essentially never
# share this many exact 6-word runs unless there's real, substantial
# overlap. Sections with too few shingles to compare meaningfully are
# skipped rather than trivially "matched".
_SIMILARITY_SHINGLE_WORDS = 6
_SIMILARITY_JACCARD_THRESHOLD = 0.3
_MIN_SHINGLES_FOR_SIMILARITY = 5

# check_repeated_paragraph_openings: same near-match grouping technique as
# check_repeated_section_openings above, but at PARAGRAPH granularity across
# the WHOLE article (paragraph boundaries are the persisted "\n\n" join --
# see app/ai/schemas.py::GeneratedSection and app/ai/nodes/section_generation.py).
# A shorter 4-word window (vs. 8 for section openings) since a paragraph
# opening's repetitive part is usually just its first word or two ("That...",
# "This..."), and a group-size threshold SCALED to the article's own
# paragraph count (never a fixed absolute count) -- a handful of articles
# share an opening word once or twice by coincidence; what's actually
# excessive is a meaningful fraction of all paragraphs doing it. Skipped
# entirely (PASS) below a minimum paragraph count, where "a fraction of
# all paragraphs" isn't a meaningful signal yet.
_PARAGRAPH_OPENING_WINDOW_WORDS = 4
_PARAGRAPH_OPENING_SIMILARITY_RATIO = 0.8
_PARAGRAPH_OPENING_MIN_PARAGRAPHS = 8
_PARAGRAPH_OPENING_MIN_GROUP_SIZE = 4
_PARAGRAPH_OPENING_GROUP_RATIO = 0.25

# check_attribution_phrase_density: the minimum set the requirement names
# (says/explains/argues/notes/claims), each tense-expanded so a normal past-
# tense article isn't undercounted. Matched with \b word-boundary regexes
# (never a naive substring count) so e.g. "notes" doesn't also match inside
# an unrelated word. Density (occurrences per 100 words), not a fixed
# absolute count, so this scales across article lengths -- see the
# check's own docstring for why a per-100-word rate is the right unit.
_ATTRIBUTION_PHRASES = (
    "says",
    "said",
    "explains",
    "explained",
    "argues",
    "argued",
    "notes",
    "noted",
    "claims",
    "claimed",
)
_ATTRIBUTION_DENSITY_PER_100_WORDS_THRESHOLD = 1.5
_ATTRIBUTION_MIN_ARTICLE_WORDS = 200
_ATTRIBUTION_MAX_EXAMPLES = 3

PASS = "pass"
WARNING = "warning"
FAILURE = "failure"


@dataclass(frozen=True)
class CheckResult:
    name: str
    severity: str  # PASS | WARNING | FAILURE
    details: str
    # Affected section sequence_numbers, when the check can attribute the
    # issue to specific sections (empty for article-wide checks like
    # article_length or section_count).
    sections: tuple[int, ...] = ()
    # Optional counts/measurements a human reviewer or the editorial
    # review can use without re-deriving them (e.g. a similarity ratio, a
    # word count, how many sections a phrase repeats in).
    metrics: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True unless this is a hard FAILURE -- a WARNING still counts as
        passed for every existing gate (ValidationReport.passed, revision
        triggering), exactly like PASS. This keeps every consumer that
        predates `severity` working unchanged."""
        return self.severity != FAILURE


@dataclass(frozen=True)
class ValidationReport:
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity == WARNING]

    def to_json(self) -> list[dict]:
        return [
            {
                "name": c.name,
                "passed": c.passed,
                "severity": c.severity,
                "details": c.details,
                "sections": list(c.sections),
                "metrics": c.metrics,
            }
            for c in self.checks
        ]


# --- helpers shared by the repetition/similarity checks -------------------------------


def _word_ngrams(text: str, n: int) -> list[str]:
    words = text.split()
    if len(words) < n:
        return []
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def _shingles(content: str, n: int) -> set[str]:
    normalized = " ".join(content.split()).lower()
    return set(_word_ngrams(normalized, n))


# --- 1. Source coverage --------------------------------------------------------------


def check_source_coverage(sections: list[ArticleSection], all_chunk_ids: set[uuid.UUID]) -> CheckResult:
    if not all_chunk_ids:
        return CheckResult("source_coverage", PASS, "No chunks to cover.")
    referenced: set[uuid.UUID] = set()
    for s in sections:
        referenced.update(s.supporting_chunk_ids)
    referenced &= all_chunk_ids
    coverage = len(referenced) / len(all_chunk_ids)
    # Not every chunk must be cited (intros/small-talk are often dropped) --
    # only a total absence of grounding is treated as a hard failure; the
    # actual percentage is always reported for human judgment.
    severity = PASS if coverage > 0 else FAILURE
    return CheckResult(
        "source_coverage",
        severity,
        f"{len(referenced)}/{len(all_chunk_ids)} chunks ({coverage:.0%}) referenced by at least one section.",
        metrics={"referenced": len(referenced), "total": len(all_chunk_ids), "coverage_ratio": round(coverage, 4)},
    )


# --- 2. Source traceability (chunk references are real) -----------------------------


def check_source_traceability(sections: list[ArticleSection], valid_chunk_ids: set[uuid.UUID]) -> CheckResult:
    invalid: list[str] = []
    bad_sections: set[int] = set()
    for s in sections:
        unknown = set(s.supporting_chunk_ids) - valid_chunk_ids
        if unknown:
            invalid.append(f"section {s.sequence_number} ({s.heading!r}): {len(unknown)} unknown chunk id(s)")
            bad_sections.add(s.sequence_number)
    return CheckResult(
        "source_traceability",
        PASS if not invalid else FAILURE,
        "All supporting_chunk_ids resolve to real chunks." if not invalid else "; ".join(invalid),
        sections=tuple(sorted(bad_sections)),
    )


# --- 3. No empty / implausibly short sections ------------------------------------------


def is_section_content_valid(heading: str, content: str) -> bool:
    """The same minimal validity bar check_no_empty_sections enforces
    below, factored out so app/ai/nodes/section_generation.py can reuse it
    to decide whether an already-persisted ArticleSection represents
    genuinely completed work (safe to reuse on a resumed job) or must be
    regenerated -- the strongest existing deterministic signal, rather
    than inventing a separate stage-completion marker. A section whose
    content matches a known generation-artifact pattern (see
    generation_artifact_match above) is never "genuinely completed work"
    either, regardless of length -- resumability must regenerate it, not
    silently reuse it (app/repositories/article_repository.py::upsert_section
    then replaces it in place, the same as any other invalid-existing-row
    case)."""
    stripped_content = content.strip()
    if not heading.strip() or len(stripped_content) < _MIN_SECTION_CONTENT_CHARS:
        return False
    return generation_artifact_match(stripped_content) is None


def check_no_empty_sections(sections: list[ArticleSection]) -> CheckResult:
    empty: list[int] = []
    reasons: dict[int, str] = {}
    for s in sections:
        if is_section_content_valid(s.heading, s.content):
            continue
        empty.append(s.sequence_number)
        artifact = generation_artifact_match(s.content.strip())
        # "section N" phrasing per flagged section (not a bare list) so
        # app/ai/nodes/revision.py's regex-based targeting can regenerate
        # exactly the failing section(s) instead of falling back to
        # regenerating the whole article.
        reasons[s.sequence_number] = (
            f"section {s.sequence_number} contains a generation artifact ({artifact!r}), not real article prose"
            if artifact is not None
            else f"section {s.sequence_number} is empty or too short"
        )
    if empty:
        details = "; ".join(reasons[n] for n in empty)
        return CheckResult("no_empty_sections", FAILURE, details, sections=tuple(empty))

    # Relative-shortness: a section far shorter than the article's own
    # typical section is a plausible sign of a truncated/broken
    # generation, but only a WARNING -- a legitimately short intro or
    # conclusion is normal and must not fail the article. Both the ratio
    # AND the absolute floor must be tripped, and only evaluated with
    # enough sections to make "the rest of the article" meaningful.
    if len(sections) < _RELATIVE_SHORT_SECTION_MIN_SECTIONS:
        return CheckResult("no_empty_sections", PASS, "No empty/too-short sections.")

    word_counts = {s.sequence_number: len(s.content.split()) for s in sections}
    mean_words = sum(word_counts.values()) / len(word_counts)
    threshold = mean_words * _RELATIVE_SHORT_SECTION_RATIO
    short = [
        seq
        for seq, wc in word_counts.items()
        if wc < threshold and wc < _RELATIVE_SHORT_SECTION_ABS_WORDS
    ]
    if not short:
        return CheckResult("no_empty_sections", PASS, "No empty/too-short sections.")
    details = "; ".join(f"section {n} is implausibly short relative to the rest of the article" for n in short)
    return CheckResult(
        "no_empty_sections",
        WARNING,
        details,
        sections=tuple(sorted(short)),
        metrics={"mean_section_words": round(mean_words), "threshold_words": round(threshold)},
    )


# --- 4. No duplicate sections (stricter than the general similarity check) --------------


def check_no_duplicate_sections(sections: list[ArticleSection]) -> CheckResult:
    seen_headings: dict[str, int] = {}
    seen_content: dict[str, int] = {}
    duplicates: list[str] = []
    bad_sections: set[int] = set()
    for s in sections:
        heading_key = s.heading.strip().lower()
        content_key = s.content.strip().lower()
        if heading_key and heading_key in seen_headings:
            other = seen_headings[heading_key]
            duplicates.append(f"section {s.sequence_number} duplicates heading of section {other}")
            bad_sections.update({s.sequence_number, other})
        if content_key and content_key in seen_content:
            other = seen_content[content_key]
            duplicates.append(f"section {s.sequence_number} duplicates content of section {other}")
            bad_sections.update({s.sequence_number, other})
        seen_headings.setdefault(heading_key, s.sequence_number)
        seen_content.setdefault(content_key, s.sequence_number)
    return CheckResult(
        "no_duplicate_sections",
        PASS if not duplicates else FAILURE,
        "No duplicate sections." if not duplicates else "; ".join(duplicates),
        sections=tuple(sorted(bad_sections)),
    )


# --- 5. Excessive repeated/near-duplicate paragraphs (within/across sections) -----------


def check_no_duplicate_paragraphs(sections: list[ArticleSection]) -> CheckResult:
    """A WARNING, not a FAILURE: a verbatim or near-verbatim paragraph
    repeated elsewhere is a probable editorial issue (the exact scenario
    requirement #3 describes -- a later section re-explaining something
    from scratch), but not definitive proof the article is broken -- a
    genuinely duplicated whole section is the stricter, FAILURE-level
    check_no_duplicate_sections above. Near-duplicates (not just exact
    matches) are caught via difflib's SequenceMatcher ratio, stdlib, no
    new dependency. Short lines (transitions, single-sentence asides)
    below _MIN_DUPLICATE_PARAGRAPH_CHARS are ignored so a short phrase
    repeated legitimately never counts."""
    seen: list[tuple[int, str]] = []
    duplicates: list[str] = []
    bad_sections: set[int] = set()
    for s in sections:
        for paragraph in s.content.split("\n\n"):
            normalized = " ".join(paragraph.split()).lower()
            if len(normalized) < _MIN_DUPLICATE_PARAGRAPH_CHARS:
                continue
            match_seq = None
            for seen_seq, seen_paragraph in seen:
                if normalized == seen_paragraph or (
                    SequenceMatcher(None, normalized, seen_paragraph).ratio() >= _NEAR_DUPLICATE_PARAGRAPH_RATIO
                ):
                    match_seq = seen_seq
                    break
            if match_seq is not None:
                duplicates.append(f"section {s.sequence_number} repeats a paragraph from section {match_seq}")
                bad_sections.update({s.sequence_number, match_seq})
            else:
                seen.append((s.sequence_number, normalized))
    return CheckResult(
        "no_duplicate_paragraphs",
        PASS if not duplicates else WARNING,
        "No duplicate/near-duplicate paragraphs." if not duplicates else "; ".join(duplicates),
        sections=tuple(sorted(bad_sections)),
        metrics={"duplicate_paragraph_count": len(duplicates)},
    )


# --- 6. No missing planned sections -----------------------------------------------------


def check_no_missing_planned_sections(plan: ArticlePlan, sections: list[ArticleSection]) -> CheckResult:
    planned_sequence_numbers = {s["sequence_number"] for s in plan.sections}
    generated_sequence_numbers = {s.sequence_number for s in sections}
    missing = sorted(planned_sequence_numbers - generated_sequence_numbers)
    # "section N" phrasing (see check_no_empty_sections) so a missing
    # section gets targeted for (re)generation, not just reported.
    details = (
        "All planned sections were generated."
        if not missing
        else "; ".join(f"section {n} was planned but never generated" for n in missing)
    )
    return CheckResult("no_missing_planned_sections", PASS if not missing else FAILURE, details, sections=tuple(missing))


# --- 7. Article length: hard safety ceiling (FAILURE) vs. soft editorial target (WARNING) --


def check_article_length(
    sections: list[ArticleSection],
    transcript_word_count: int,
    max_length_ratio: float,
    target_word_count_min: int | None = None,
    target_word_count_max: int | None = None,
) -> CheckResult:
    """Two-tier by design (unchanged from the existing architecture):
    `max_length_ratio` (Settings.article_max_length_ratio) is the HARD
    safety ceiling -- exceeding it, or an implausibly tiny article, is a
    FAILURE. `target_word_count_min/max` (Settings.article_target_word_count_*)
    is a SOFT editorial goal -- missing it while still under the hard
    ceiling is only a WARNING, and is skipped entirely (no tier at all)
    when the caller doesn't supply it, so this check's existing
    ratio-only behavior is unchanged for any caller that doesn't pass the
    new, optional soft-target arguments."""
    article_word_count = sum(len(s.content.split()) for s in sections)
    if article_word_count < _MIN_ARTICLE_WORDS:
        return CheckResult(
            "article_length",
            FAILURE,
            f"Article is only {article_word_count} words -- likely a broken/empty generation.",
            metrics={"word_count": article_word_count},
        )

    metrics: dict = {"word_count": article_word_count}
    if transcript_word_count > 0:
        ratio = article_word_count / transcript_word_count
        metrics["ratio_of_transcript"] = round(ratio, 4)
        if ratio > max_length_ratio:
            return CheckResult(
                "article_length",
                FAILURE,
                f"Article is {article_word_count} words ({ratio:.0%} of the {transcript_word_count}-word "
                f"transcript; hard safety ceiling is <= {max_length_ratio:.0%}).",
                metrics=metrics,
            )

    if (
        target_word_count_min is not None
        and target_word_count_max is not None
        and not (target_word_count_min <= article_word_count <= target_word_count_max)
    ):
        return CheckResult(
            "article_length",
            WARNING,
            f"Article is {article_word_count} words, outside the {target_word_count_min}-"
            f"{target_word_count_max} soft editorial target (still within the safety ceiling).",
            metrics=metrics,
        )

    return CheckResult("article_length", PASS, f"Article is {article_word_count} words.", metrics=metrics)


# --- 8. Unsupported / unreferenced generated content -------------------------------------


def check_unsupported_content(sections: list[ArticleSection]) -> CheckResult:
    # A structural proxy for "unsupported claims" -- true claim-level
    # fidelity checking needs an LLM/NLP judge (out of scope for a
    # deterministic check); this flags the unambiguous case of a section
    # with literally no evidence trail at all.
    unsupported = [
        s.sequence_number for s in sections if not s.supporting_chunk_ids and not s.supporting_topic_ids
    ]
    # "section N" phrasing (see check_no_empty_sections) so revision.py
    # can target exactly the unsupported section(s).
    details = (
        "Every section cites at least one source."
        if not unsupported
        else "; ".join(f"section {n} has no source at all" for n in unsupported)
    )
    return CheckResult(
        "unsupported_content", PASS if not unsupported else FAILURE, details, sections=tuple(unsupported)
    )


# --- 9. Invalid source references (topics + the plan's own references) -------------------


def check_invalid_source_references(
    sections: list[ArticleSection], plan: ArticlePlan, valid_topic_ids: set[uuid.UUID]
) -> CheckResult:
    problems: list[str] = []
    bad_sections: set[int] = set()
    for s in sections:
        unknown = {tid for tid in s.supporting_topic_ids if tid not in valid_topic_ids}
        if unknown:
            problems.append(f"section {s.sequence_number}: {len(unknown)} unknown topic id(s)")
            bad_sections.add(s.sequence_number)
    valid_topic_id_strs = {str(t) for t in valid_topic_ids}
    for planned in plan.sections:
        unknown_plan_topics = set(planned.get("supporting_topic_ids", [])) - valid_topic_id_strs
        if unknown_plan_topics:
            problems.append(
                f"plan section {planned.get('sequence_number')}: {len(unknown_plan_topics)} unknown topic id(s)"
            )
    return CheckResult(
        "invalid_source_references",
        PASS if not problems else FAILURE,
        "All topic references are valid." if not problems else "; ".join(problems),
        sections=tuple(sorted(bad_sections)),
    )


# --- 10. Section count sanity (pathological counts only, not the target range) ----------


def check_section_count(sections: list[ArticleSection], min_sections: int, max_sections: int) -> CheckResult:
    """`min_sections`/`max_sections` (Settings.section_count_min/max) are a
    pathology bound, not the planner's target range
    (Settings.section_count_target_min/max, used only as prompt guidance
    in app/ai/prompts.py::planning_prompt) -- deliberately much wider, so
    a genuinely well-structured article that happens to land outside the
    *typical* target (e.g. 5 or 11 sections) never fails this check.
    """
    count = len(sections)
    passed = min_sections <= count <= max_sections
    details = (
        f"Article has {count} section(s)."
        if passed
        else f"Article has {count} section(s), outside the expected {min_sections}-{max_sections} range."
    )
    return CheckResult("section_count", PASS if passed else FAILURE, details, metrics={"section_count": count})


# --- 11. Broken transcript timestamp references -------------------------------------------


def check_broken_timestamp_references(
    sections: list[ArticleSection], chunks_by_id: dict[uuid.UUID, Chunk]
) -> CheckResult:
    broken: list[str] = []
    bad_sections: set[int] = set()
    for s in sections:
        for chunk_id in s.supporting_chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue  # already reported by check_source_traceability
            if chunk.start_ms < 0 or chunk.end_ms < 0 or chunk.start_ms > chunk.end_ms:
                broken.append(
                    f"section {s.sequence_number} cites chunk {chunk_id} with invalid timing "
                    f"(start_ms={chunk.start_ms}, end_ms={chunk.end_ms})"
                )
                bad_sections.add(s.sequence_number)
    return CheckResult(
        "broken_timestamp_references",
        PASS if not broken else FAILURE,
        "All cited chunks have valid timestamps." if not broken else "; ".join(broken),
        sections=tuple(sorted(bad_sections)),
    )


# --- 12. Broken provenance chain (section -> its own topics -> their chunks) ------------


def check_broken_provenance_chain(
    sections: list[ArticleSection], topics_by_id: dict[uuid.UUID, Topic]
) -> CheckResult:
    """Verifies the part of the provenance chain the other checks don't:
    check_source_traceability confirms a section's supporting_chunk_ids
    exist SOMEWHERE in the episode; this confirms they exist among the
    chunks of the section's OWN cited topics -- i.e. that
    app/ai/nodes/planning.py's invariant (a planned section's
    supporting_chunk_ids is always exactly the union of its supporting
    topics' chunk_ids) still holds for this section. A chunk id that
    resolves globally but isn't traceable to any topic THIS section
    itself cites is the "silently lost or corrupted during revision"
    failure mode requirement #9 describes -- e.g. one section's
    provenance getting mixed up with another's. A topic id that doesn't
    resolve at all is left to check_invalid_source_references (skipped
    here, not double-reported); a section isn't checked at all if none of
    its topic references are known, since there's nothing to verify the
    chain against."""
    broken: list[str] = []
    bad_sections: set[int] = set()
    for s in sections:
        known_topic_chunk_ids: set[uuid.UUID] = set()
        has_known_topic = False
        for tid in s.supporting_topic_ids:
            topic = topics_by_id.get(tid)
            if topic is None:
                continue
            has_known_topic = True
            known_topic_chunk_ids.update(topic.chunk_ids)
        if not has_known_topic:
            continue
        orphaned = [cid for cid in s.supporting_chunk_ids if cid not in known_topic_chunk_ids]
        if orphaned:
            broken.append(
                f"section {s.sequence_number}: {len(orphaned)} supporting chunk id(s) not traceable to any "
                "of its own cited topics -- provenance chain broken"
            )
            bad_sections.add(s.sequence_number)
    return CheckResult(
        "broken_provenance_chain",
        PASS if not broken else FAILURE,
        "Every section's chunk references trace back to its own cited topics." if not broken else "; ".join(broken),
        sections=tuple(sorted(bad_sections)),
    )


# --- 13. Excessive exact multi-word phrase repetition across sections -------------------


def check_excessive_phrase_repetition(sections: list[ArticleSection]) -> CheckResult:
    """WARNING only -- a probable sign the same point is being repeated
    almost verbatim across the article (requirement #2) rather than
    genuinely revisited, but never proof on its own. See module-level
    constants for the conservative thresholds (6-word window, must span
    >= 3 distinct sections) that keep names/terms/short phrases and
    naturally recurring terminology from ever tripping this."""
    phrase_sections: dict[str, set[int]] = {}
    for s in sections:
        normalized = " ".join(s.content.split()).lower()
        for ngram in set(_word_ngrams(normalized, _PHRASE_NGRAM_WORDS)):
            phrase_sections.setdefault(ngram, set()).add(s.sequence_number)

    offending = {phrase: secs for phrase, secs in phrase_sections.items() if len(secs) >= _PHRASE_MIN_SECTIONS}
    if not offending:
        return CheckResult("excessive_phrase_repetition", PASS, "No excessively repeated phrases across sections.")

    all_sections = sorted({seq for secs in offending.values() for seq in secs})
    examples = sorted(offending.items(), key=lambda kv: -len(kv[1]))[:_PHRASE_MAX_EXAMPLES]
    details = "; ".join(f"phrase {phrase!r} appears in {len(secs)} sections {sorted(secs)}" for phrase, secs in examples)
    return CheckResult(
        "excessive_phrase_repetition",
        WARNING,
        details,
        sections=tuple(all_sections),
        metrics={"repeated_phrase_count": len(offending)},
    )


# --- 14. Repeated section-opening patterns -----------------------------------------------


def check_repeated_section_openings(sections: list[ArticleSection]) -> CheckResult:
    """WARNING only -- catches many sections repeatedly opening with the
    same attribution/template structure (requirement #3's example), while
    a natural, legitimately-repeated name or attribution shared by only
    two sections stays unflagged (see _OPENING_MIN_GROUP_SIZE)."""
    if len(sections) < _OPENING_MIN_GROUP_SIZE:
        return CheckResult("repeated_section_openings", PASS, "Too few sections to assess opening patterns.")

    openings = [
        (s.sequence_number, " ".join(s.content.split()).lower().split(" ")[:_OPENING_WINDOW_WORDS])
        for s in sections
        if s.content.strip()
    ]
    groups: list[list[tuple[int, str]]] = []
    for seq, words in openings:
        opening = " ".join(words)
        if not opening:
            continue
        placed = False
        for group in groups:
            representative = group[0][1]
            if opening == representative or SequenceMatcher(None, opening, representative).ratio() >= _OPENING_SIMILARITY_RATIO:
                group.append((seq, opening))
                placed = True
                break
        if not placed:
            groups.append([(seq, opening)])

    offending = [g for g in groups if len(g) >= _OPENING_MIN_GROUP_SIZE]
    if not offending:
        return CheckResult("repeated_section_openings", PASS, "No excessively repeated section-opening patterns.")

    largest = max(offending, key=len)
    secs = tuple(sorted(seq for seq, _ in largest))
    details = f"{len(largest)} sections open with a near-identical pattern: sections {list(secs)}"
    return CheckResult(
        "repeated_section_openings",
        WARNING,
        details,
        sections=secs,
        metrics={"largest_group_size": len(largest), "groups_flagged": len(offending)},
    )


# --- 15. Suspiciously similar sections (conservative, non-exact) ------------------------


def check_similar_sections(sections: list[ArticleSection]) -> CheckResult:
    """WARNING only, and deliberately conservative (requirement #4):
    Jaccard similarity of 6-word shingle sets, a lightweight deterministic
    technique already used by check_excessive_phrase_repetition above --
    no embeddings, no new model. Two sections legitimately covering the
    same broader concept from different angles share vocabulary but
    essentially never share this many exact 6-word runs, so the 30%
    threshold rarely fires on genuinely distinct content. An exact
    complete-content duplicate is also reported here (as 100% similar)
    but is already the stricter, FAILURE-level check_no_duplicate_sections
    above -- this check adds value for the *partial* overlap case that
    one doesn't catch."""
    shingles_by_seq = {s.sequence_number: _shingles(s.content, _SIMILARITY_SHINGLE_WORDS) for s in sections}
    seqs = sorted(shingles_by_seq)
    flagged: list[str] = []
    bad_sections: set[int] = set()
    for i in range(len(seqs)):
        a = shingles_by_seq[seqs[i]]
        if len(a) < _MIN_SHINGLES_FOR_SIMILARITY:
            continue
        for j in range(i + 1, len(seqs)):
            b = shingles_by_seq[seqs[j]]
            if len(b) < _MIN_SHINGLES_FOR_SIMILARITY:
                continue
            union = len(a | b)
            jaccard = len(a & b) / union if union else 0.0
            if jaccard >= _SIMILARITY_JACCARD_THRESHOLD:
                flagged.append(f"section {seqs[i]} and section {seqs[j]} are {jaccard:.0%} similar")
                bad_sections.update({seqs[i], seqs[j]})

    if not flagged:
        return CheckResult("similar_sections", PASS, "No suspiciously similar sections.")
    return CheckResult(
        "similar_sections",
        WARNING,
        "; ".join(flagged),
        sections=tuple(sorted(bad_sections)),
        metrics={"similar_pair_count": len(flagged)},
    )


# --- 16. Repeated paragraph-opening patterns (article-wide, paragraph granularity) ------


def check_repeated_paragraph_openings(sections: list[ArticleSection]) -> CheckResult:
    """WARNING only -- the paragraph-level counterpart to
    check_repeated_section_openings above: several paragraphs across the
    article (not just within one section) opening with the same word or
    construction ("That...", "That...", "That...", "That...") is a common,
    genuinely distracting editorial pattern that section-level checks can't
    see at all. Uses actual paragraph boundaries (the persisted "\\n\\n"
    join -- see app/ai/schemas.py::GeneratedSection's docstring), not
    sentence splitting or any other heuristic. Deliberately conservative:
    grouped by near-match (SequenceMatcher, same technique and ratio as the
    section-opening check) rather than exact match, so two differently-worded
    openings never count as the same group, and flagged only once the
    offending group is a large enough FRACTION of all paragraphs in the
    article (never a fixed absolute count) -- a couple of paragraphs
    naturally starting the same common way is normal, not a defect.
    """
    paragraphs: list[tuple[int, str]] = []
    for s in sections:
        for paragraph in s.content.split("\n\n"):
            if paragraph.strip():
                paragraphs.append((s.sequence_number, paragraph.strip()))

    if len(paragraphs) < _PARAGRAPH_OPENING_MIN_PARAGRAPHS:
        return CheckResult(
            "repeated_paragraph_openings", PASS, "Too few paragraphs to assess opening patterns."
        )

    openings: list[tuple[int, str]] = []
    for seq, text in paragraphs:
        words = " ".join(text.split()).lower().split(" ")[:_PARAGRAPH_OPENING_WINDOW_WORDS]
        opening = " ".join(words)
        if opening:
            openings.append((seq, opening))

    groups: list[list[tuple[int, str]]] = []
    for seq, opening in openings:
        placed = False
        for group in groups:
            representative = group[0][1]
            if (
                opening == representative
                or SequenceMatcher(None, opening, representative).ratio() >= _PARAGRAPH_OPENING_SIMILARITY_RATIO
            ):
                group.append((seq, opening))
                placed = True
                break
        if not placed:
            groups.append([(seq, opening)])

    threshold = max(
        _PARAGRAPH_OPENING_MIN_GROUP_SIZE, round(len(paragraphs) * _PARAGRAPH_OPENING_GROUP_RATIO)
    )
    offending = [g for g in groups if len(g) >= threshold]
    if not offending:
        return CheckResult(
            "repeated_paragraph_openings", PASS, "No excessively repeated paragraph-opening patterns."
        )

    largest = max(offending, key=len)
    secs = tuple(sorted({seq for seq, _ in largest}))
    example = largest[0][1]
    details = (
        f"{len(largest)} of {len(paragraphs)} paragraphs open with a near-identical pattern "
        f"(e.g. {example!r}), spanning sections {list(secs)}"
    )
    return CheckResult(
        "repeated_paragraph_openings",
        WARNING,
        details,
        sections=secs,
        metrics={
            "largest_group_size": len(largest),
            "total_paragraphs": len(paragraphs),
            "threshold": threshold,
        },
    )


# --- 17. Excessive attribution-phrase density (article-wide) ----------------------------


def check_attribution_phrase_density(sections: list[ArticleSection]) -> CheckResult:
    """WARNING only -- flags an article that leans on the same handful of
    attribution verbs (says/explains/argues/notes/claims, at minimum) so
    often it reads mechanically, WITHOUT ever suggesting attribution be
    removed (see app/ai/prompts.py::section_generation_prompt's ATTRIBUTION
    guidance, which explicitly says to vary the construction, never drop
    it -- this check exists to catch a failure to vary, not to discourage
    attribution itself). Measured as a DENSITY (occurrences per 100 words),
    not a fixed absolute count, so a long article naturally attributing more
    often in raw-count terms doesn't trip this while a short one repeating
    the same verb constantly does. Matched with word-boundary regexes so a
    phrase is never counted as a substring of an unrelated word. Skipped
    entirely below a minimum article length, where a density estimate isn't
    meaningful yet."""
    full_text = " ".join(s.content for s in sections)
    total_words = len(full_text.split())
    if total_words < _ATTRIBUTION_MIN_ARTICLE_WORDS:
        return CheckResult(
            "attribution_phrase_density", PASS, "Article too short to assess attribution density."
        )

    normalized = full_text.lower()
    counts: dict[str, int] = {}
    for phrase in _ATTRIBUTION_PHRASES:
        matches = re.findall(rf"\b{re.escape(phrase)}\b", normalized)
        if matches:
            counts[phrase] = len(matches)

    total_occurrences = sum(counts.values())
    density = (total_occurrences / total_words) * 100
    if density <= _ATTRIBUTION_DENSITY_PER_100_WORDS_THRESHOLD:
        return CheckResult(
            "attribution_phrase_density",
            PASS,
            f"Attribution phrases appear {total_occurrences} times in {total_words} words "
            f"({density:.2f} per 100 words).",
            metrics={
                "total_attribution_occurrences": total_occurrences,
                "total_words": total_words,
                "density_per_100_words": round(density, 2),
            },
        )

    top_examples = sorted(counts.items(), key=lambda kv: -kv[1])[:_ATTRIBUTION_MAX_EXAMPLES]
    examples_text = ", ".join(f"{phrase!r} x{count}" for phrase, count in top_examples)
    details = (
        f"Attribution phrases appear {total_occurrences} times in {total_words} words "
        f"({density:.2f} per 100 words, above the {_ATTRIBUTION_DENSITY_PER_100_WORDS_THRESHOLD} "
        f"per-100-word guide) -- most frequent: {examples_text}."
    )
    return CheckResult(
        "attribution_phrase_density",
        WARNING,
        details,
        metrics={
            "total_attribution_occurrences": total_occurrences,
            "total_words": total_words,
            "density_per_100_words": round(density, 2),
        },
    )


def run_validation(
    *,
    sections: list[ArticleSection],
    plan: ArticlePlan,
    chunks: list[Chunk],
    topics: list[Topic],
    transcript_word_count: int,
    max_length_ratio: float,
    min_sections: int,
    max_sections: int,
    target_word_count_min: int | None = None,
    target_word_count_max: int | None = None,
) -> ValidationReport:
    valid_chunk_ids = {c.id for c in chunks}
    valid_topic_ids = {t.id for t in topics}
    chunks_by_id = {c.id: c for c in chunks}
    topics_by_id = {t.id: t for t in topics}

    checks = [
        check_source_coverage(sections, valid_chunk_ids),
        check_source_traceability(sections, valid_chunk_ids),
        check_no_empty_sections(sections),
        check_no_duplicate_sections(sections),
        check_no_duplicate_paragraphs(sections),
        check_no_missing_planned_sections(plan, sections),
        check_article_length(sections, transcript_word_count, max_length_ratio, target_word_count_min, target_word_count_max),
        check_unsupported_content(sections),
        check_invalid_source_references(sections, plan, valid_topic_ids),
        check_section_count(sections, min_sections, max_sections),
        check_broken_timestamp_references(sections, chunks_by_id),
        check_broken_provenance_chain(sections, topics_by_id),
        check_excessive_phrase_repetition(sections),
        check_repeated_section_openings(sections),
        check_similar_sections(sections),
        check_repeated_paragraph_openings(sections),
        check_attribution_phrase_density(sections),
    ]
    return ValidationReport(checks=checks)
