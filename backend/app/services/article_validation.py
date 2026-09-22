"""Deterministic article validation (Phase H). Pure Python, no LLM and no
DB session -- takes already-loaded rows and plain data, returns a report.
This is the reliability backbone of Phase H: an optional LLM-based
evaluator (app/ai/nodes/validation.py, gated by
Settings.enable_llm_validation) is an additional, clearly-separate signal
appended to `checks`, never a replacement for any check here.

Each check is independently callable and independently testable
(tests/unit/test_article_validation.py) -- run_validation() just runs all
of them and folds the result into one report.
"""

import uuid
from dataclasses import dataclass

from app.models.article_plan import ArticlePlan
from app.models.article_section import ArticleSection
from app.models.chunk import Chunk
from app.models.topic import Topic

_MIN_SECTION_CONTENT_CHARS = 50
_MIN_DUPLICATE_PARAGRAPH_CHARS = 40
_MIN_ARTICLE_WORDS = 100


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    details: str


@dataclass(frozen=True)
class ValidationReport:
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_json(self) -> list[dict]:
        return [{"name": c.name, "passed": c.passed, "details": c.details} for c in self.checks]


# --- 1. Source coverage --------------------------------------------------------------


def check_source_coverage(sections: list[ArticleSection], all_chunk_ids: set[uuid.UUID]) -> CheckResult:
    if not all_chunk_ids:
        return CheckResult("source_coverage", True, "No chunks to cover.")
    referenced: set[uuid.UUID] = set()
    for s in sections:
        referenced.update(s.supporting_chunk_ids)
    referenced &= all_chunk_ids
    coverage = len(referenced) / len(all_chunk_ids)
    # Not every chunk must be cited (intros/small-talk are often dropped) --
    # only a total absence of grounding is treated as a hard failure; the
    # actual percentage is always reported for human judgment.
    passed = coverage > 0
    return CheckResult(
        "source_coverage",
        passed,
        f"{len(referenced)}/{len(all_chunk_ids)} chunks ({coverage:.0%}) referenced by at least one section.",
    )


# --- 2. Source traceability (chunk references are real) -----------------------------


def check_source_traceability(
    sections: list[ArticleSection], valid_chunk_ids: set[uuid.UUID]
) -> CheckResult:
    invalid: list[str] = []
    for s in sections:
        unknown = set(s.supporting_chunk_ids) - valid_chunk_ids
        if unknown:
            invalid.append(f"section {s.sequence_number} ({s.heading!r}): {len(unknown)} unknown chunk id(s)")
    return CheckResult(
        "source_traceability",
        not invalid,
        "All supporting_chunk_ids resolve to real chunks." if not invalid else "; ".join(invalid),
    )


# --- 3. No empty sections -------------------------------------------------------------


def is_section_content_valid(heading: str, content: str) -> bool:
    """The same minimal validity bar check_no_empty_sections enforces
    below, factored out so app/ai/nodes/section_generation.py can reuse it
    to decide whether an already-persisted ArticleSection represents
    genuinely completed work (safe to reuse on a resumed job) or must be
    regenerated -- the strongest existing deterministic signal, rather
    than inventing a separate stage-completion marker."""
    return bool(heading.strip()) and len(content.strip()) >= _MIN_SECTION_CONTENT_CHARS


def check_no_empty_sections(sections: list[ArticleSection]) -> CheckResult:
    empty = [
        s.sequence_number for s in sections if not is_section_content_valid(s.heading, s.content)
    ]
    # "section N" phrasing per flagged section (not a bare list) so
    # app/ai/nodes/revision.py's regex-based targeting can regenerate
    # exactly the failing section(s) instead of falling back to
    # regenerating the whole article.
    details = (
        "No empty/too-short sections."
        if not empty
        else "; ".join(f"section {n} is empty or too short" for n in empty)
    )
    return CheckResult("no_empty_sections", not empty, details)


# --- 4. No duplicate sections ----------------------------------------------------------


def check_no_duplicate_sections(sections: list[ArticleSection]) -> CheckResult:
    seen_headings: dict[str, int] = {}
    seen_content: dict[str, int] = {}
    duplicates: list[str] = []
    for s in sections:
        heading_key = s.heading.strip().lower()
        content_key = s.content.strip().lower()
        if heading_key and heading_key in seen_headings:
            duplicates.append(f"section {s.sequence_number} duplicates heading of section {seen_headings[heading_key]}")
        if content_key and content_key in seen_content:
            duplicates.append(f"section {s.sequence_number} duplicates content of section {seen_content[content_key]}")
        seen_headings.setdefault(heading_key, s.sequence_number)
        seen_content.setdefault(content_key, s.sequence_number)
    return CheckResult(
        "no_duplicate_sections",
        not duplicates,
        "No duplicate sections." if not duplicates else "; ".join(duplicates),
    )


# --- 5. No duplicate paragraphs (within/across sections) -------------------------------


def check_no_duplicate_paragraphs(sections: list[ArticleSection]) -> CheckResult:
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for s in sections:
        for paragraph in s.content.split("\n\n"):
            normalized = " ".join(paragraph.split()).lower()
            if len(normalized) < _MIN_DUPLICATE_PARAGRAPH_CHARS:
                continue  # short transitional lines aren't meaningful duplicates
            if normalized in seen:
                duplicates.append(
                    f"section {s.sequence_number} repeats a paragraph from section {seen[normalized]}"
                )
            else:
                seen[normalized] = s.sequence_number
    return CheckResult(
        "no_duplicate_paragraphs",
        not duplicates,
        "No duplicate paragraphs." if not duplicates else "; ".join(duplicates),
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
    return CheckResult("no_missing_planned_sections", not missing, details)


# --- 7. Article length (substantially shorter than the source) -------------------------


def check_article_length(
    sections: list[ArticleSection], transcript_word_count: int, max_length_ratio: float
) -> CheckResult:
    article_word_count = sum(len(s.content.split()) for s in sections)
    if article_word_count < _MIN_ARTICLE_WORDS:
        return CheckResult(
            "article_length",
            False,
            f"Article is only {article_word_count} words -- likely a broken/empty generation.",
        )
    if transcript_word_count == 0:
        return CheckResult("article_length", True, f"Article is {article_word_count} words.")
    ratio = article_word_count / transcript_word_count
    passed = ratio <= max_length_ratio
    return CheckResult(
        "article_length",
        passed,
        f"Article is {article_word_count} words ({ratio:.0%} of the {transcript_word_count}-word "
        f"transcript; target is <= {max_length_ratio:.0%}).",
    )


# --- 8. Unsupported / unreferenced generated content -------------------------------------


def check_unsupported_content(sections: list[ArticleSection]) -> CheckResult:
    # A structural proxy for "unsupported claims" -- true claim-level
    # fidelity checking needs an LLM/NLP judge (out of scope for a
    # deterministic check); this flags the unambiguous case of a section
    # with literally no evidence trail at all.
    unsupported = [
        s.sequence_number
        for s in sections
        if not s.supporting_chunk_ids and not s.supporting_topic_ids
    ]
    # "section N" phrasing (see check_no_empty_sections) so revision.py
    # can target exactly the unsupported section(s).
    details = (
        "Every section cites at least one source."
        if not unsupported
        else "; ".join(f"section {n} has no source at all" for n in unsupported)
    )
    return CheckResult("unsupported_content", not unsupported, details)


# --- 9. Invalid source references (topics + the plan's own references) -------------------


def check_invalid_source_references(
    sections: list[ArticleSection], plan: ArticlePlan, valid_topic_ids: set[uuid.UUID]
) -> CheckResult:
    problems: list[str] = []
    for s in sections:
        unknown = {tid for tid in s.supporting_topic_ids if tid not in valid_topic_ids}
        if unknown:
            problems.append(f"section {s.sequence_number}: {len(unknown)} unknown topic id(s)")
    valid_topic_id_strs = {str(t) for t in valid_topic_ids}
    for planned in plan.sections:
        unknown_plan_topics = set(planned.get("supporting_topic_ids", [])) - valid_topic_id_strs
        if unknown_plan_topics:
            problems.append(
                f"plan section {planned.get('sequence_number')}: {len(unknown_plan_topics)} unknown topic id(s)"
            )
    return CheckResult(
        "invalid_source_references",
        not problems,
        "All topic references are valid." if not problems else "; ".join(problems),
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
    return CheckResult("section_count", passed, details)


# --- 11. Broken transcript timestamp references -------------------------------------------


def check_broken_timestamp_references(
    sections: list[ArticleSection], chunks_by_id: dict[uuid.UUID, Chunk]
) -> CheckResult:
    broken: list[str] = []
    for s in sections:
        for chunk_id in s.supporting_chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue  # already reported by check_source_traceability
            if chunk.start_ms < 0 or chunk.end_ms < 0 or chunk.start_ms > chunk.end_ms:
                broken.append(f"section {s.sequence_number} cites chunk {chunk_id} with invalid timing "
                               f"(start_ms={chunk.start_ms}, end_ms={chunk.end_ms})")
    return CheckResult(
        "broken_timestamp_references",
        not broken,
        "All cited chunks have valid timestamps." if not broken else "; ".join(broken),
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
) -> ValidationReport:
    valid_chunk_ids = {c.id for c in chunks}
    valid_topic_ids = {t.id for t in topics}
    chunks_by_id = {c.id: c for c in chunks}

    checks = [
        check_source_coverage(sections, valid_chunk_ids),
        check_source_traceability(sections, valid_chunk_ids),
        check_no_empty_sections(sections),
        check_no_duplicate_sections(sections),
        check_no_duplicate_paragraphs(sections),
        check_no_missing_planned_sections(plan, sections),
        check_article_length(sections, transcript_word_count, max_length_ratio),
        check_unsupported_content(sections),
        check_invalid_source_references(sections, plan, valid_topic_ids),
        check_section_count(sections, min_sections, max_sections),
        check_broken_timestamp_references(sections, chunks_by_id),
    ]
    return ValidationReport(checks=checks)
