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
    than inventing a separate stage-completion marker."""
    return bool(heading.strip()) and len(content.strip()) >= _MIN_SECTION_CONTENT_CHARS


def check_no_empty_sections(sections: list[ArticleSection]) -> CheckResult:
    empty = [s.sequence_number for s in sections if not is_section_content_valid(s.heading, s.content)]
    if empty:
        # "section N" phrasing per flagged section (not a bare list) so
        # app/ai/nodes/revision.py's regex-based targeting can regenerate
        # exactly the failing section(s) instead of falling back to
        # regenerating the whole article.
        details = "; ".join(f"section {n} is empty or too short" for n in empty)
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
    ]
    return ValidationReport(checks=checks)
