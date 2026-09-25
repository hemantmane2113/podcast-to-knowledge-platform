import uuid

import pytest

from app.models.article_plan import ArticlePlan
from app.models.article_section import ArticleSection
from app.models.chunk import Chunk
from app.models.topic import Topic
from app.services.article_validation import (
    FAILURE,
    PASS,
    WARNING,
    check_article_length,
    check_attribution_phrase_density,
    check_broken_provenance_chain,
    check_broken_timestamp_references,
    check_excessive_phrase_repetition,
    check_invalid_source_references,
    check_no_duplicate_paragraphs,
    check_no_duplicate_sections,
    check_no_empty_sections,
    check_no_missing_planned_sections,
    check_repeated_paragraph_openings,
    check_repeated_section_openings,
    check_section_count,
    check_similar_sections,
    check_source_coverage,
    check_source_traceability,
    check_unsupported_content,
    generation_artifact_match,
    is_section_content_valid,
    repair_mojibake,
    run_validation,
)


def _chunk(chunk_id: uuid.UUID | None = None, start_ms: int = 0, end_ms: int = 1000) -> Chunk:
    return Chunk(
        id=chunk_id or uuid.uuid4(),
        transcript_id=uuid.uuid4(),
        episode_id=uuid.uuid4(),
        sequence_number=0,
        text="chunk text",
        start_ms=start_ms,
        end_ms=end_ms,
        source_segment_ids=[],
        token_count=10,
    )


def _topic(topic_id: uuid.UUID | None = None, chunk_ids: list[uuid.UUID] | None = None) -> Topic:
    return Topic(
        id=topic_id or uuid.uuid4(),
        transcript_id=uuid.uuid4(),
        episode_id=uuid.uuid4(),
        sequence_number=0,
        title="t",
        summary="s",
        chunk_ids=chunk_ids or [],
        key_claims=[],
        subtopics=[],
    )


def _section(
    *,
    sequence_number: int = 0,
    heading: str = "Heading",
    content: str = "x" * 60,
    supporting_chunk_ids: list[uuid.UUID] | None = None,
    supporting_topic_ids: list[uuid.UUID] | None = None,
) -> ArticleSection:
    return ArticleSection(
        id=uuid.uuid4(),
        article_id=uuid.uuid4(),
        sequence_number=sequence_number,
        heading=heading,
        content=content,
        supporting_chunk_ids=supporting_chunk_ids or [],
        supporting_topic_ids=supporting_topic_ids or [],
    )


def _plan(sections: list[dict] | None = None) -> ArticlePlan:
    return ArticlePlan(
        id=uuid.uuid4(),
        episode_id=uuid.uuid4(),
        title="Article",
        introduction_summary="i",
        conclusion_summary="c",
        sections=sections or [],
    )


# --- source coverage ------------------------------------------------------------------


def test_source_coverage_passes_with_no_chunks_at_all() -> None:
    result = check_source_coverage([], set())
    assert result.passed


def test_source_coverage_fails_when_nothing_is_referenced() -> None:
    chunk_id = uuid.uuid4()
    result = check_source_coverage([_section(supporting_chunk_ids=[])], {chunk_id})
    assert not result.passed


def test_source_coverage_passes_with_partial_coverage_and_reports_percentage() -> None:
    c1, c2 = uuid.uuid4(), uuid.uuid4()
    result = check_source_coverage([_section(supporting_chunk_ids=[c1])], {c1, c2})
    assert result.passed
    assert "50%" in result.details


# --- source traceability ---------------------------------------------------------------


def test_source_traceability_passes_when_all_chunk_ids_are_valid() -> None:
    c1 = uuid.uuid4()
    result = check_source_traceability([_section(supporting_chunk_ids=[c1])], {c1})
    assert result.passed


def test_source_traceability_fails_for_unknown_chunk_id() -> None:
    known, unknown = uuid.uuid4(), uuid.uuid4()
    result = check_source_traceability([_section(supporting_chunk_ids=[unknown])], {known})
    assert not result.passed
    assert "unknown chunk id" in result.details


# --- empty sections ----------------------------------------------------------------------


def test_no_empty_sections_passes_for_real_content() -> None:
    result = check_no_empty_sections([_section(content="x" * 60)])
    assert result.passed


def test_no_empty_sections_fails_for_too_short_content() -> None:
    result = check_no_empty_sections([_section(content="short")])
    assert not result.passed


def test_no_empty_sections_fails_for_blank_heading() -> None:
    result = check_no_empty_sections([_section(heading="   ", content="x" * 60)])
    assert not result.passed


def test_no_empty_sections_does_not_flag_a_short_intro_in_a_two_section_article() -> None:
    # Only 2 sections -- too few for a meaningful "typical section length"
    # baseline, so the relative-shortness signal must not fire at all.
    short_intro = " ".join(["word"] * 40)
    long_body = " ".join(["word"] * 900)
    result = check_no_empty_sections(
        [_section(sequence_number=0, content=short_intro), _section(sequence_number=1, content=long_body)]
    )
    assert result.passed
    assert result.severity == PASS


def test_no_empty_sections_warns_for_a_section_implausibly_short_relative_to_the_article() -> None:
    tiny = " ".join(["word"] * 30)  # above the absolute 50-char floor, but tiny next to the others
    normal = " ".join(["word"] * 900)
    result = check_no_empty_sections(
        [
            _section(sequence_number=0, content=tiny),
            _section(sequence_number=1, content=normal),
            _section(sequence_number=2, content=normal),
        ]
    )
    assert result.passed  # WARNING, not FAILURE
    assert result.severity == WARNING
    assert result.sections == (0,)


def test_no_empty_sections_does_not_warn_for_a_legitimately_shorter_but_substantial_section() -> None:
    # A section noticeably shorter than the others but still well over the
    # absolute floor (>= 100 words) is normal variation, not a warning.
    shorter = " ".join(["word"] * 300)
    normal = " ".join(["word"] * 900)
    result = check_no_empty_sections(
        [
            _section(sequence_number=0, content=shorter),
            _section(sequence_number=1, content=normal),
            _section(sequence_number=2, content=normal),
        ]
    )
    assert result.severity == PASS


# --- generation-artifact detection (Live/Draft Article Workflow follow-up fix) -----------
# Regression tests for a real observed defect: the model's response leaking
# meta-commentary about its own generation/validation process into what
# should be article prose. Both literal reported artifacts are covered.


def test_generation_artifact_match_detects_the_first_reported_artifact() -> None:
    assert generation_artifact_match("paragraphs continuation error") is not None


def test_generation_artifact_match_detects_the_second_reported_artifact() -> None:
    assert generation_artifact_match("paragraphs continuation invalid") is not None


def test_generation_artifact_match_detects_the_third_reported_artifact_paragraphs_continued() -> None:
    # A second real generated article (after the first fix) surfaced a
    # different word form of the same meta-commentary failure mode:
    # "paragraphs continued?" (a verb, not the noun "continuation" the
    # original pattern covered).
    assert generation_artifact_match("paragraphs continued?") is not None
    assert generation_artifact_match("paragraphs continued") is not None


def test_generation_artifact_match_does_not_flag_legitimate_prose_mentioning_paragraphs() -> None:
    # The same real article's section 10 legitimately discusses paragraphs
    # as a topic ("paragraphs like 'chop wood, carry water' matter...") --
    # the detector must stay narrow to "paragraph(s) continued/continuation",
    # never a broad `"paragraphs" in text` rule that would reject this too.
    legitimate = 'Short paragraphs like "chop wood, carry water" matter because they mirror everyday practice.'
    assert generation_artifact_match(legitimate) is None
    # Same sentence with the actual curly quotes the model would use, not
    # straight ones -- proves the narrowness isn't an artifact of ASCII
    # punctuation in the test fixture above.
    legitimate_curly = "Short paragraphs like “chop wood, carry water” matter because they mirror everyday practice."
    assert generation_artifact_match(legitimate_curly) is None


def test_generation_artifact_match_does_not_flag_continued_unrelated_to_paragraphs() -> None:
    # "continued" on its own, nowhere near "paragraph(s)", is ordinary
    # prose (a tradition, a practice, a habit continuing) -- the pattern
    # requires "paragraph(s)" immediately before "continued", never a bare
    # "continued" anywhere in the text.
    assert generation_artifact_match("The tradition continued for decades after the founder's death.") is None
    assert generation_artifact_match("Training continued well into the evening, he said.") is None


def test_generation_artifact_match_returns_none_for_real_prose() -> None:
    real_prose = (
        "The conversation turns to how sleep affects memory consolidation, with concrete "
        "examples drawn from the research discussed earlier in the episode."
    )
    assert generation_artifact_match(real_prose) is None


def test_is_section_content_valid_rejects_a_generation_artifact_even_if_long_enough() -> None:
    # Long enough to clear the plain length floor on its own -- the
    # artifact pattern must still be what invalidates it, not length.
    content = "paragraphs continuation error. " * 5
    assert len(content) >= 50
    assert not is_section_content_valid("Intro", content)


def test_check_no_empty_sections_fails_for_a_generation_artifact_and_reports_it_specifically() -> None:
    result = check_no_empty_sections([_section(content="paragraphs continuation invalid")])
    assert not result.passed
    assert result.severity == FAILURE
    assert "generation artifact" in result.details
    # The matched pattern text is included so a reviewer can see exactly
    # what triggered this -- not necessarily the whole flagged sentence.
    assert "paragraphs continuation" in result.details


def test_check_no_empty_sections_only_flags_the_artifact_section_not_legitimate_ones() -> None:
    # Proves the strengthened check doesn't over-reject: a genuinely short
    # section is left alone (or only WARNs, per the existing relative-
    # shortness logic), while an artifact-laden section in the SAME
    # article is a hard FAILURE naming only itself.
    normal = " ".join(["word"] * 900)
    result = check_no_empty_sections(
        [
            _section(sequence_number=0, content="paragraphs continuation error"),
            _section(sequence_number=1, content=normal),
            _section(sequence_number=2, content=normal),
        ]
    )
    assert result.severity == FAILURE
    assert result.sections == (0,)


def test_generated_section_schema_rejects_a_generation_artifact_paragraph() -> None:
    """Root-cause fix, isolated from the retry-loop mechanics tested in
    tests/unit/test_llm_providers.py: GeneratedSection's own Pydantic
    validator (app/ai/schemas.py) rejects a paragraph matching a known
    artifact pattern at construction time -- this is what turns a
    syntactically-valid-but-garbage response into a ValidationError the
    LLM provider's existing corrective-retry loop can actually act on."""
    from pydantic import ValidationError

    from app.ai.schemas import GeneratedSection

    with pytest.raises(ValidationError, match="generation artifact"):
        GeneratedSection(heading="Intro", paragraphs=["paragraphs continuation error"])


def test_generated_section_schema_rejects_the_paragraphs_continued_variant() -> None:
    from pydantic import ValidationError

    from app.ai.schemas import GeneratedSection

    with pytest.raises(ValidationError, match="generation artifact"):
        GeneratedSection(heading="Section 5", paragraphs=["Some real content here.", "paragraphs continued?"])


def test_generated_section_schema_accepts_legitimate_prose_about_paragraphs() -> None:
    from app.ai.schemas import GeneratedSection

    section = GeneratedSection(
        heading="Section 10",
        paragraphs=['Short paragraphs like "chop wood, carry water" matter because they mirror practice.'],
    )
    assert "chop wood" in section.paragraphs[0]


def test_generated_section_schema_accepts_real_prose() -> None:
    from app.ai.schemas import GeneratedSection

    section = GeneratedSection(
        heading="Intro", paragraphs=["A real paragraph discussing the podcast's actual content."]
    )
    assert section.paragraphs == ["A real paragraph discussing the podcast's actual content."]


def test_run_validation_fails_the_whole_report_when_a_section_has_a_generation_artifact() -> None:
    normal = " ".join(["word"] * 900)
    sections = [
        _section(sequence_number=0, content="paragraphs continuation error", supporting_chunk_ids=[]),
        _section(sequence_number=1, content=normal, supporting_chunk_ids=[]),
    ]
    plan = _plan([{"sequence_number": 0}, {"sequence_number": 1}])
    report = run_validation(
        sections=sections,
        plan=plan,
        chunks=[],
        topics=[],
        transcript_word_count=10_000,
        max_length_ratio=0.4,
        min_sections=1,
        max_sections=20,
    )
    assert not report.passed
    empty_check = next(c for c in report.checks if c.name == "no_empty_sections")
    assert empty_check.severity == FAILURE
    assert "generation artifact" in empty_check.details


# --- mojibake repair (Live/Draft Article Workflow follow-up fix #2) ---------------------
# Regression tests for a real observed defect: a generated article contained
# "Hubermanâs", "âsomething hereâ" -- UTF-8 text mis-decoded as a
# single-byte codec somewhere upstream of this pipeline (no encode/decode
# call exists anywhere in this codebase's own I/O -- see repair_mojibake's
# docstring), most consistent with a latin-1 mis-decode (the corrupted
# bytes for the invisible C1 control characters a cp1252 mis-decode would
# instead render as visible €/™-style characters, matching what was
# actually reported: a bare "â" with nothing else visible around it).


def test_repair_mojibake_fixes_the_exact_reported_huberman_apostrophe() -> None:
    # The real reported string, "Hubermanâs", derived the same way the
    # actual corruption happens (UTF-8 bytes for the apostrophe mis-decoded
    # as latin-1) rather than hand-typed -- typing just the letter "â"
    # would silently drop the two C1 control characters (U+0080, U+0099)
    # that come along with a real latin-1 mis-decode and are invisible in
    # a terminal/most fonts, which is exactly why the bug report only
    # *showed* a bare "â": those two characters are genuinely part of the
    # corrupted string, just not visible when printed or copy-pasted.
    correct = "Huberman’s"
    corrupted = correct.encode("utf-8").decode("latin-1")
    # \xe2\x80\x99 in a str literal is three codepoints (U+00E2, U+0080,
    # U+0099), matching what `corrupted` actually holds -- U+0080/U+0099
    # are non-printable C1 controls, so `corrupted` PRINTS as "Hubermanâs"
    # (verified: str.isprintable() is False, and stripping the two C1
    # controls leaves exactly the visible "Hubermanâs" the bug report
    # showed) even though it is a 12-character string, not an 8-character
    # one with a single substituted letter.
    assert corrupted == "Huberman\xe2\x80\x99s"
    assert not corrupted.isprintable()
    printable_only = "".join(ch for ch in corrupted if ch.isprintable())
    assert printable_only == "Hubermanâs"
    assert repair_mojibake(corrupted) == correct
    assert repair_mojibake(corrupted) == "Huberman's".replace("'", "’")


def test_repair_mojibake_fixes_the_exact_reported_neuroscientist_quote() -> None:
    correct = "“nearly three decades as a neuroscientist”"
    corrupted = correct.encode("utf-8").decode("latin-1")
    printable_only = "".join(ch for ch in corrupted if ch.isprintable())
    assert printable_only == "ânearly three decades as a neuroscientistâ"
    assert repair_mojibake(corrupted) == correct


def test_repair_mojibake_fixes_a_cp1252_misdecoded_apostrophe() -> None:
    correct = "Huberman’s"
    corrupted = correct.encode("utf-8").decode("cp1252")
    assert repair_mojibake(corrupted) == correct


def test_repair_mojibake_leaves_normal_ascii_prose_unchanged() -> None:
    text = "This sentence has nothing unusual about it at all."
    assert repair_mojibake(text) == text


def test_repair_mojibake_leaves_genuinely_accented_text_unchanged() -> None:
    # Real accented names/words, never a mis-decoded sequence -- both
    # round-trip codecs must fail to re-decode as UTF-8 and fall back to
    # the original text unchanged (verified: neither codec can reconstruct
    # valid UTF-8 bytes from a single already-correct Unicode character).
    text = "The guest, François, explained the concept clearly over café."
    assert repair_mojibake(text) == text


def test_repair_mojibake_leaves_genuine_curly_quotes_and_em_dash_unchanged() -> None:
    # Correctly-encoded typographic punctuation (not mis-decoded) must
    # round-trip unchanged -- proves the repair is narrowly self-validating
    # rather than a blanket transform of anything in this character range.
    text = "“This matters—deeply,” she said, ‘and so does that.’"
    assert repair_mojibake(text) == text


def test_repair_mojibake_is_idempotent_on_already_corrupted_text() -> None:
    correct = "Huberman’s"
    corrupted = correct.encode("utf-8").decode("latin-1")
    once = repair_mojibake(corrupted)
    twice = repair_mojibake(once)
    assert once == correct
    assert twice == once


def test_repair_mojibake_is_idempotent_on_already_correct_text() -> None:
    text = "“nearly three decades as a neuroscientist,” Huberman’s work — all correct."
    once = repair_mojibake(text)
    twice = repair_mojibake(once)
    assert once == text
    assert twice == text


def test_generated_section_schema_repairs_mojibake_in_a_paragraph() -> None:
    from app.ai.schemas import GeneratedSection

    correct = "Huberman’s research spans nearly three decades."
    corrupted = correct.encode("utf-8").decode("latin-1")
    section = GeneratedSection(heading="Intro", paragraphs=[corrupted])
    assert section.paragraphs == [correct]


# --- duplicate sections ------------------------------------------------------------------


def test_no_duplicate_sections_passes_for_distinct_sections() -> None:
    result = check_no_duplicate_sections(
        [_section(sequence_number=0, heading="A", content="aaa"), _section(sequence_number=1, heading="B", content="bbb")]
    )
    assert result.passed


def test_no_duplicate_sections_fails_for_identical_heading() -> None:
    result = check_no_duplicate_sections(
        [
            _section(sequence_number=0, heading="Same", content="aaa"),
            _section(sequence_number=1, heading="Same", content="bbb"),
        ]
    )
    assert not result.passed


def test_no_duplicate_sections_fails_for_identical_content() -> None:
    result = check_no_duplicate_sections(
        [
            _section(sequence_number=0, heading="A", content="identical text"),
            _section(sequence_number=1, heading="B", content="identical text"),
        ]
    )
    assert not result.passed


# --- duplicate paragraphs ------------------------------------------------------------------


def test_no_duplicate_paragraphs_passes_for_distinct_paragraphs() -> None:
    result = check_no_duplicate_paragraphs(
        [_section(content="This is a first unique paragraph of reasonable length.")]
    )
    assert result.passed


def test_no_duplicate_paragraphs_warns_across_sections_but_does_not_fail() -> None:
    # A repeated paragraph is a probable editorial issue (requirement #1/#3)
    # -- worth a WARNING, but not proof the article is broken (that's the
    # stricter, FAILURE-level check_no_duplicate_sections for a whole
    # duplicated section).
    repeated = "This exact paragraph appears twice across two different sections in the article."
    result = check_no_duplicate_paragraphs(
        [
            _section(sequence_number=0, content=repeated),
            _section(sequence_number=1, content=repeated),
        ]
    )
    assert result.passed
    assert result.severity == WARNING
    assert result.sections == (0, 1)


def test_no_duplicate_paragraphs_detects_near_duplicates_not_just_exact_matches() -> None:
    a = "The researchers found that morning cortisol levels strongly predict afternoon energy crashes."
    b = "The researchers found that morning cortisol levels strongly predict afternoon energy declines."
    result = check_no_duplicate_paragraphs(
        [_section(sequence_number=0, content=a), _section(sequence_number=1, content=b)]
    )
    assert result.severity == WARNING


def test_no_duplicate_paragraphs_ignores_short_transitional_lines() -> None:
    result = check_no_duplicate_paragraphs(
        [_section(sequence_number=0, content="OK."), _section(sequence_number=1, content="OK.")]
    )
    assert result.passed
    assert result.severity == PASS


def test_no_duplicate_paragraphs_does_not_fail_a_legitimate_shared_qualification() -> None:
    # A short, necessary qualification repeated verbatim across sections
    # (just over the 40-char noise floor) must never be a hard failure.
    qualification = "The study authors note this finding is preliminary."
    result = check_no_duplicate_paragraphs(
        [
            _section(sequence_number=0, content=f"Some analysis here.\n\n{qualification}"),
            _section(sequence_number=1, content=f"Different analysis here.\n\n{qualification}"),
        ]
    )
    assert result.passed  # a WARNING, never a FAILURE, however many legitimate reasons it repeats


# --- missing planned sections --------------------------------------------------------------


def test_no_missing_planned_sections_passes_when_all_generated() -> None:
    plan = _plan(sections=[{"sequence_number": 0}, {"sequence_number": 1}])
    result = check_no_missing_planned_sections(
        plan, [_section(sequence_number=0), _section(sequence_number=1)]
    )
    assert result.passed


def test_no_missing_planned_sections_fails_when_one_is_missing() -> None:
    plan = _plan(sections=[{"sequence_number": 0}, {"sequence_number": 1}])
    result = check_no_missing_planned_sections(plan, [_section(sequence_number=0)])
    assert not result.passed
    assert "section 1" in result.details


# --- article length ----------------------------------------------------------------------


def test_article_length_passes_when_substantially_shorter() -> None:
    content = " ".join(["word"] * 200)
    result = check_article_length([_section(content=content)], transcript_word_count=10000, max_length_ratio=0.4)
    assert result.passed


def test_article_length_fails_when_too_close_to_transcript_length() -> None:
    content = " ".join(["word"] * 5000)
    result = check_article_length([_section(content=content)], transcript_word_count=10000, max_length_ratio=0.4)
    assert not result.passed


def test_article_length_fails_when_essentially_empty() -> None:
    result = check_article_length([_section(content="too short")], transcript_word_count=10000, max_length_ratio=0.4)
    assert not result.passed


def test_article_length_ignores_soft_target_when_not_provided() -> None:
    # Backward compatibility: a caller that doesn't pass target_word_count_*
    # (like the tests above) gets exactly the old ratio-only behavior --
    # no WARNING tier appears out of nowhere.
    content = " ".join(["word"] * 200)  # nowhere near a 5500-6500 target
    result = check_article_length([_section(content=content)], transcript_word_count=10000, max_length_ratio=0.4)
    assert result.severity == PASS


def test_article_length_warns_when_under_the_soft_target_but_within_the_hard_ceiling() -> None:
    content = " ".join(["word"] * 2000)  # well under 5500, still under the 40% hard ratio
    result = check_article_length(
        [_section(content=content)],
        transcript_word_count=100_000,
        max_length_ratio=0.4,
        target_word_count_min=5500,
        target_word_count_max=6500,
    )
    assert result.passed  # a WARNING never fails the article
    assert result.severity == WARNING


def test_article_length_warns_when_over_the_soft_target_but_within_the_hard_ceiling() -> None:
    content = " ".join(["word"] * 7000)  # over 6500, still under the 40% hard ratio of a 100k-word transcript
    result = check_article_length(
        [_section(content=content)],
        transcript_word_count=100_000,
        max_length_ratio=0.4,
        target_word_count_min=5500,
        target_word_count_max=6500,
    )
    assert result.passed
    assert result.severity == WARNING


def test_article_length_passes_within_the_soft_target_range() -> None:
    content = " ".join(["word"] * 6000)
    result = check_article_length(
        [_section(content=content)],
        transcript_word_count=100_000,
        max_length_ratio=0.4,
        target_word_count_min=5500,
        target_word_count_max=6500,
    )
    assert result.severity == PASS


def test_article_length_hard_ceiling_still_fails_even_within_the_soft_target() -> None:
    # The hard ceiling always wins over the soft target -- a soft-target
    # "pass" never masks exceeding article_max_length_ratio.
    content = " ".join(["word"] * 6000)
    result = check_article_length(
        [_section(content=content)],
        transcript_word_count=10_000,  # 6000/10000 = 60% > 40% hard ceiling
        max_length_ratio=0.4,
        target_word_count_min=5500,
        target_word_count_max=6500,
    )
    assert not result.passed
    assert result.severity == FAILURE


# --- unsupported content ------------------------------------------------------------------


def test_unsupported_content_passes_when_section_has_chunk_support() -> None:
    result = check_unsupported_content([_section(supporting_chunk_ids=[uuid.uuid4()])])
    assert result.passed


def test_unsupported_content_passes_when_section_has_only_topic_support() -> None:
    result = check_unsupported_content([_section(supporting_topic_ids=[uuid.uuid4()])])
    assert result.passed


def test_unsupported_content_fails_when_section_has_no_source_at_all() -> None:
    result = check_unsupported_content([_section(supporting_chunk_ids=[], supporting_topic_ids=[])])
    assert not result.passed


# --- invalid source references -------------------------------------------------------------


def test_invalid_source_references_passes_for_valid_topic_ids() -> None:
    topic_id = uuid.uuid4()
    result = check_invalid_source_references(
        [_section(supporting_topic_ids=[topic_id])], _plan(), {topic_id}
    )
    assert result.passed


def test_invalid_source_references_fails_for_unknown_topic_id_in_section() -> None:
    known, unknown = uuid.uuid4(), uuid.uuid4()
    result = check_invalid_source_references(
        [_section(supporting_topic_ids=[unknown])], _plan(), {known}
    )
    assert not result.passed


def test_invalid_source_references_fails_for_unknown_topic_id_in_plan() -> None:
    known = uuid.uuid4()
    plan = _plan(sections=[{"sequence_number": 0, "supporting_topic_ids": [str(uuid.uuid4())]}])
    result = check_invalid_source_references([], plan, {known})
    assert not result.passed


# --- broken timestamp references -----------------------------------------------------------


def test_broken_timestamp_references_passes_for_valid_chunk_timing() -> None:
    chunk = _chunk(start_ms=1000, end_ms=5000)
    result = check_broken_timestamp_references(
        [_section(supporting_chunk_ids=[chunk.id])], {chunk.id: chunk}
    )
    assert result.passed


def test_broken_timestamp_references_fails_when_start_after_end() -> None:
    chunk = _chunk(start_ms=5000, end_ms=1000)
    result = check_broken_timestamp_references(
        [_section(supporting_chunk_ids=[chunk.id])], {chunk.id: chunk}
    )
    assert not result.passed


def test_broken_timestamp_references_ignores_unknown_chunk_ids() -> None:
    # Already reported by check_source_traceability -- shouldn't double-fail here.
    result = check_broken_timestamp_references([_section(supporting_chunk_ids=[uuid.uuid4()])], {})
    assert result.passed


# --- section count (pathological counts only, not the target range) -----------------------


def test_section_count_passes_within_the_normal_range() -> None:
    # 1. A normal count within the target range passes.
    sections = [_section(sequence_number=i) for i in range(8)]
    result = check_section_count(sections, min_sections=3, max_sections=20)
    assert result.passed


def test_section_count_fails_for_a_pathologically_low_count() -> None:
    # 2. A pathological low count is detected.
    sections = [_section(sequence_number=0)]
    result = check_section_count(sections, min_sections=3, max_sections=20)
    assert not result.passed
    assert "1 section" in result.details


def test_section_count_fails_for_a_pathologically_high_count() -> None:
    # 3. A pathological high count is detected.
    sections = [_section(sequence_number=i) for i in range(25)]
    result = check_section_count(sections, min_sections=3, max_sections=20)
    assert not result.passed
    assert "25 section" in result.details


def test_section_count_boundary_values() -> None:
    # 4. Boundary values: exactly min/max pass (inclusive); one below/above fails.
    at_min = [_section(sequence_number=i) for i in range(3)]
    at_max = [_section(sequence_number=i) for i in range(20)]
    below_min = [_section(sequence_number=i) for i in range(2)]
    above_max = [_section(sequence_number=i) for i in range(21)]

    assert check_section_count(at_min, min_sections=3, max_sections=20).passed
    assert check_section_count(at_max, min_sections=3, max_sections=20).passed
    assert not check_section_count(below_min, min_sections=3, max_sections=20).passed
    assert not check_section_count(above_max, min_sections=3, max_sections=20).passed


def test_section_count_does_not_reject_a_reasonable_article_slightly_outside_the_target_guidance() -> None:
    # A 5- or 11-section article (just outside the planner's 6-10 TARGET
    # guidance, app/ai/prompts.py::planning_prompt) must never be treated
    # as pathological -- the validation bound is deliberately much wider
    # than the target range.
    five_sections = [_section(sequence_number=i) for i in range(5)]
    eleven_sections = [_section(sequence_number=i) for i in range(11)]
    assert check_section_count(five_sections, min_sections=3, max_sections=20).passed
    assert check_section_count(eleven_sections, min_sections=3, max_sections=20).passed


# --- broken provenance chain (section -> its own topics -> their chunks) ----------------


def test_broken_provenance_chain_passes_when_chunk_ids_trace_to_their_own_topic() -> None:
    chunk = _chunk()
    topic = _topic(chunk_ids=[chunk.id])
    result = check_broken_provenance_chain(
        [_section(supporting_chunk_ids=[chunk.id], supporting_topic_ids=[topic.id])],
        {topic.id: topic},
    )
    assert result.passed


def test_broken_provenance_chain_fails_when_a_chunk_id_is_not_traceable_to_its_own_topic() -> None:
    chunk, other_chunk = _chunk(), _chunk()
    topic = _topic(chunk_ids=[other_chunk.id])  # does NOT include chunk.id
    result = check_broken_provenance_chain(
        [_section(sequence_number=0, supporting_chunk_ids=[chunk.id], supporting_topic_ids=[topic.id])],
        {topic.id: topic},
    )
    assert not result.passed
    assert result.severity == FAILURE
    assert result.sections == (0,)


def test_broken_provenance_chain_skips_a_section_with_no_known_topic() -> None:
    # An unknown topic id is check_invalid_source_references' job to
    # report -- this check has nothing to verify the chain against here.
    chunk = _chunk()
    result = check_broken_provenance_chain(
        [_section(supporting_chunk_ids=[chunk.id], supporting_topic_ids=[uuid.uuid4()])], {}
    )
    assert result.passed


# --- excessive exact phrase repetition across sections -----------------------------------


def test_excessive_phrase_repetition_passes_for_distinct_sections() -> None:
    result = check_excessive_phrase_repetition(
        [
            _section(sequence_number=0, content="Section zero discusses an entirely different subject in depth."),
            _section(sequence_number=1, content="Section one covers a completely unrelated area of research."),
        ]
    )
    assert result.passed


def test_excessive_phrase_repetition_does_not_flag_a_name_repeated_many_times() -> None:
    # A short recurring name is nowhere near the 6-word window this check
    # looks for, and must never be flagged on its own.
    result = check_excessive_phrase_repetition(
        [
            _section(sequence_number=0, content="Dr. Smith began the interview describing her background in neuroscience."),
            _section(sequence_number=1, content="Later, Dr. Smith turned to a completely different topic about sleep."),
            _section(sequence_number=2, content="Dr. Smith closed with thoughts on future research directions entirely."),
            _section(sequence_number=3, content="Throughout, Dr. Smith remained focused on practical, everyday advice only."),
        ]
    )
    assert result.passed


def test_excessive_phrase_repetition_warns_when_a_long_phrase_repeats_across_three_plus_sections() -> None:
    phrase = "the researchers found that morning cortisol predicts energy"
    result = check_excessive_phrase_repetition(
        [
            _section(sequence_number=0, content=f"{phrase} in the first study discussed here today."),
            _section(sequence_number=1, content=f"Later on, {phrase} in the second study as well."),
            _section(sequence_number=2, content=f"Finally, {phrase} in a third and separate study too."),
        ]
    )
    assert result.passed  # WARNING, not FAILURE
    assert result.severity == WARNING
    assert result.sections == (0, 1, 2)


# --- repeated section-opening patterns ----------------------------------------------------


def test_repeated_section_openings_passes_for_distinct_openings() -> None:
    result = check_repeated_section_openings(
        [
            _section(sequence_number=0, content="The first section begins with a discussion of the setup."),
            _section(sequence_number=1, content="Turning now to the results, the data reveals a clear trend."),
            _section(sequence_number=2, content="Finally, the implications of these results deserve careful consideration here."),
        ]
    )
    assert result.passed


def test_repeated_section_openings_does_not_flag_two_sections_sharing_an_opening() -> None:
    # Only two sections share this opening -- not "clearly excessive".
    shared_opening = "According to Dr. Smith the researchers observed that"
    result = check_repeated_section_openings(
        [
            _section(sequence_number=0, content=f"{shared_opening} energy levels rose sharply after breakfast each day."),
            _section(sequence_number=1, content=f"{shared_opening} sleep quality improved after the intervention began."),
            _section(sequence_number=2, content="A completely different section with its own unrelated opening entirely."),
        ]
    )
    assert result.passed


def test_repeated_section_openings_warns_when_three_or_more_sections_share_a_template_opening() -> None:
    shared_opening = "According to Dr. Smith the researchers observed that"
    result = check_repeated_section_openings(
        [
            _section(sequence_number=0, content=f"{shared_opening} energy levels rose sharply after breakfast each day."),
            _section(sequence_number=1, content=f"{shared_opening} sleep quality improved after the intervention began."),
            _section(sequence_number=2, content=f"{shared_opening} focus increased noticeably during the afternoon session."),
        ]
    )
    assert result.passed  # WARNING, not FAILURE
    assert result.severity == WARNING
    assert result.sections == (0, 1, 2)


# --- suspiciously similar sections (conservative) -----------------------------------------


def test_similar_sections_passes_for_distinct_sections() -> None:
    result = check_similar_sections(
        [
            _section(sequence_number=0, content=" ".join([f"alpha{i}" for i in range(30)])),
            _section(sequence_number=1, content=" ".join([f"beta{i}" for i in range(30)])),
        ]
    )
    assert result.passed


def test_similar_sections_does_not_flag_topically_related_but_distinct_sections() -> None:
    # A small amount of natural vocabulary overlap (a shared concept) is
    # not the same as substantial duplicated text, and must stay unflagged.
    shared = [f"shared{i}" for i in range(4)]
    result = check_similar_sections(
        [
            _section(sequence_number=0, content=" ".join(shared + [f"uniquea{i}" for i in range(30)])),
            _section(sequence_number=1, content=" ".join(shared + [f"uniqueb{i}" for i in range(30)])),
        ]
    )
    assert result.passed
    assert result.severity == PASS


def test_similar_sections_skips_sections_with_too_few_shingles() -> None:
    result = check_similar_sections(
        [_section(sequence_number=0, content="one two three"), _section(sequence_number=1, content="one two three")]
    )
    assert result.passed


def test_similar_sections_warns_for_substantial_partial_overlap() -> None:
    shared = [f"shared{i}" for i in range(25)]
    result = check_similar_sections(
        [
            _section(sequence_number=0, content=" ".join(shared + [f"uniquea{i}" for i in range(10)])),
            _section(sequence_number=1, content=" ".join(shared + [f"uniqueb{i}" for i in range(10)])),
        ]
    )
    assert result.passed  # WARNING, not FAILURE
    assert result.severity == WARNING
    assert result.sections == (0, 1)


# --- repeated paragraph-opening patterns (article-wide, paragraph granularity) ------------


def test_repeated_paragraph_openings_passes_when_too_few_paragraphs() -> None:
    # Only 3 paragraphs total (below _PARAGRAPH_OPENING_MIN_PARAGRAPHS) --
    # even an identical opening across all of them isn't a meaningful
    # signal yet, so this must PASS regardless of the repetition.
    paragraphs = [
        "That is exactly why the mechanism functions well over long stretches of time.",
        "That is exactly how the training regimen builds durable strength over months.",
        "That is exactly what separates casual practice from deliberate practice routines.",
    ]
    result = check_repeated_paragraph_openings([_section(content="\n\n".join(paragraphs))])
    assert result.passed
    assert result.severity == PASS


def test_repeated_paragraph_openings_passes_for_an_occasional_legitimate_repeat() -> None:
    # 12 paragraphs, only 2 of which share an opening -- well under the
    # scaled threshold, and a normal amount of incidental repetition.
    paragraphs = [
        "That is exactly why the mechanism functions well over long stretches of time.",
        "That is exactly how the training regimen builds durable strength over months.",
        "A different idea appears here about nutrition and its wide-ranging effects.",
        "Another concept touches on sleep quality and measurable daily improvements.",
        "A separate consideration covers circadian rhythms and long-term hormonal balance.",
        "The closing thought ties every earlier idea together for the reader.",
        "Yet another angle considers stress management and its downstream effects.",
        "A further point addresses hydration and its role in recovery.",
        "One more idea concerns morning routines and their lasting impact.",
        "A later paragraph turns to social support and its protective value.",
        "An additional thread examines light exposure and its hormonal signaling.",
        "The final paragraph draws every earlier thread together for the reader.",
    ]
    result = check_repeated_paragraph_openings([_section(content="\n\n".join(paragraphs))])
    assert result.passed
    assert result.severity == PASS


def test_repeated_paragraph_openings_warns_for_an_obvious_repeat() -> None:
    # 8 paragraphs, 4 of which open with a near-identical "That is exactly
    # ..." construction -- the exact "That...", "That...", "That...",
    # "That..." pattern this check exists to catch.
    paragraphs = [
        "That is exactly why the mechanism functions well over long stretches of time.",
        "That is exactly how the training regimen builds durable strength over months.",
        "That is exactly what separates casual practice from deliberate practice routines.",
        "That is exactly the reason the recovery period matters as much as effort.",
        "A different idea appears here about nutrition and its wide-ranging effects.",
        "Another concept touches on sleep quality and measurable daily improvements.",
        "A separate consideration covers circadian rhythms and long-term hormonal balance.",
        "The closing thought ties every earlier idea together for the reader.",
    ]
    result = check_repeated_paragraph_openings([_section(content="\n\n".join(paragraphs))])
    assert result.passed  # WARNING, not FAILURE
    assert result.severity == WARNING
    assert result.metrics["largest_group_size"] == 4


# --- excessive attribution-phrase density (article-wide) ----------------------------------


def test_attribution_phrase_density_passes_for_normal_attribution() -> None:
    content = "Alice says something important. " + " ".join(["filler"] * 296)  # 300 words total
    result = check_attribution_phrase_density([_section(content=content)])
    assert result.passed
    assert result.severity == PASS


def test_attribution_phrase_density_passes_when_article_too_short() -> None:
    # Below _ATTRIBUTION_MIN_ARTICLE_WORDS -- a density estimate isn't
    # meaningful yet, even with heavy repetition.
    content = " ".join(["Alice says something."] * 10)  # 30 words
    result = check_attribution_phrase_density([_section(content=content)])
    assert result.passed
    assert result.severity == PASS


def test_attribution_phrase_density_warns_for_clearly_excessive_density() -> None:
    content = " ".join(["Alice says something specific about it."] * 70)  # 350 words, "says" x70
    result = check_attribution_phrase_density([_section(content=content)])
    assert result.passed  # WARNING, not FAILURE
    assert result.severity == WARNING
    assert result.metrics["total_attribution_occurrences"] >= 70


def test_attribution_phrase_density_never_suggests_removing_attribution() -> None:
    # The WARNING's own details text must never read as an instruction to
    # drop attribution -- only to vary its construction (see
    # app/ai/prompts.py::section_generation_prompt's ATTRIBUTION guidance).
    content = " ".join(["Alice says something specific about it."] * 70)
    result = check_attribution_phrase_density([_section(content=content)])
    assert "remove" not in result.details.lower()
    assert "drop" not in result.details.lower()


# --- run_validation: full integration of all 17 checks -------------------------------------


def test_run_validation_passes_for_a_well_formed_article() -> None:
    chunk = _chunk(start_ms=0, end_ms=5000)
    # chunk_ids=[chunk.id] so the new broken_provenance_chain check's
    # section -> topic -> chunk chain is genuinely intact, matching how
    # app/ai/nodes/planning.py actually derives a section's
    # supporting_chunk_ids (always the union of its topics' chunk_ids).
    topic = _topic(chunk_ids=[chunk.id])
    section = _section(
        sequence_number=0,
        content=" ".join(["word"] * 150),  # above _MIN_ARTICLE_WORDS
        supporting_chunk_ids=[chunk.id],
        supporting_topic_ids=[topic.id],
    )
    plan = _plan(sections=[{"sequence_number": 0, "supporting_topic_ids": [str(topic.id)]}])

    report = run_validation(
        sections=[section],
        plan=plan,
        chunks=[chunk],
        topics=[topic],
        transcript_word_count=10000,
        max_length_ratio=0.4,
        # Permissive on purpose -- this test's single synthetic section
        # exercises the OTHER checks, not section-count pathology.
        min_sections=1,
        max_sections=20,
    )

    assert report.passed
    assert len(report.checks) == 17
    assert all(c.passed for c in report.checks)


def test_run_validation_warns_but_still_passes_outside_the_soft_length_target() -> None:
    chunk = _chunk(start_ms=0, end_ms=5000)
    topic = _topic(chunk_ids=[chunk.id])
    section = _section(
        sequence_number=0,
        content=" ".join(["word"] * 200),  # well under a 5500-6500 soft target
        supporting_chunk_ids=[chunk.id],
        supporting_topic_ids=[topic.id],
    )
    plan = _plan(sections=[{"sequence_number": 0, "supporting_topic_ids": [str(topic.id)]}])

    report = run_validation(
        sections=[section],
        plan=plan,
        chunks=[chunk],
        topics=[topic],
        transcript_word_count=100_000,
        max_length_ratio=0.4,
        min_sections=1,
        max_sections=20,
        target_word_count_min=5500,
        target_word_count_max=6500,
    )

    assert report.passed  # a WARNING alone never fails the report
    length_check = next(c for c in report.checks if c.name == "article_length")
    assert length_check.severity == WARNING
    assert length_check in report.warnings


def test_run_validation_fails_when_provenance_chain_is_broken() -> None:
    # A section citing a chunk that belongs to none of ITS OWN cited
    # topics -- the "silently corrupted during revision" scenario
    # requirement #9 describes -- even though the chunk id and topic id
    # are each independently valid/known.
    real_chunk = _chunk(start_ms=0, end_ms=5000)
    other_chunk = _chunk(start_ms=0, end_ms=5000)
    topic = _topic(chunk_ids=[other_chunk.id])  # does NOT include real_chunk.id
    section = _section(
        sequence_number=0,
        content=" ".join(["word"] * 150),
        supporting_chunk_ids=[real_chunk.id],
        supporting_topic_ids=[topic.id],
    )
    plan = _plan(sections=[{"sequence_number": 0, "supporting_topic_ids": [str(topic.id)]}])

    report = run_validation(
        sections=[section],
        plan=plan,
        chunks=[real_chunk, other_chunk],
        topics=[topic],
        transcript_word_count=10000,
        max_length_ratio=0.4,
        min_sections=1,
        max_sections=20,
    )

    assert not report.passed
    chain_check = next(c for c in report.checks if c.name == "broken_provenance_chain")
    assert chain_check.severity == FAILURE
    assert chain_check.sections == (0,)


def test_run_validation_fails_when_any_single_check_fails() -> None:
    chunk = _chunk()
    section = _section(content="too short")  # fails check_no_empty_sections
    plan = _plan(sections=[{"sequence_number": 0}])

    report = run_validation(
        sections=[section],
        plan=plan,
        chunks=[chunk],
        topics=[],
        transcript_word_count=10000,
        max_length_ratio=0.4,
        min_sections=1,
        max_sections=20,
    )

    assert not report.passed
    failed_names = {c.name for c in report.checks if not c.passed}
    assert "no_empty_sections" in failed_names


def test_validation_report_to_json_round_trips_check_shape() -> None:
    report = run_validation(
        sections=[],
        plan=_plan(),
        chunks=[],
        topics=[],
        transcript_word_count=0,
        max_length_ratio=0.4,
        min_sections=1,
        max_sections=20,
    )
    payload = report.to_json()
    assert len(payload) == 17
    assert all({"name", "passed", "severity", "details", "sections", "metrics"} <= set(entry) for entry in payload)
