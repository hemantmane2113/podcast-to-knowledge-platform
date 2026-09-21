import uuid

from app.models.article_plan import ArticlePlan
from app.models.article_section import ArticleSection
from app.models.chunk import Chunk
from app.models.topic import Topic
from app.services.article_validation import (
    check_article_length,
    check_broken_timestamp_references,
    check_invalid_source_references,
    check_no_duplicate_paragraphs,
    check_no_duplicate_sections,
    check_no_empty_sections,
    check_no_missing_planned_sections,
    check_source_coverage,
    check_source_traceability,
    check_unsupported_content,
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


def _topic(topic_id: uuid.UUID | None = None) -> Topic:
    return Topic(
        id=topic_id or uuid.uuid4(),
        transcript_id=uuid.uuid4(),
        episode_id=uuid.uuid4(),
        sequence_number=0,
        title="t",
        summary="s",
        chunk_ids=[],
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


def test_no_duplicate_paragraphs_fails_across_sections() -> None:
    repeated = "This exact paragraph appears twice across two different sections in the article."
    result = check_no_duplicate_paragraphs(
        [
            _section(sequence_number=0, content=repeated),
            _section(sequence_number=1, content=repeated),
        ]
    )
    assert not result.passed


def test_no_duplicate_paragraphs_ignores_short_transitional_lines() -> None:
    result = check_no_duplicate_paragraphs(
        [_section(sequence_number=0, content="OK."), _section(sequence_number=1, content="OK.")]
    )
    assert result.passed


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


# --- run_validation: full integration of all 10 checks -------------------------------------


def test_run_validation_passes_for_a_well_formed_article() -> None:
    chunk = _chunk(start_ms=0, end_ms=5000)
    topic = _topic()
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
    )

    assert report.passed
    assert len(report.checks) == 10
    assert all(c.passed for c in report.checks)


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
    )

    assert not report.passed
    failed_names = {c.name for c in report.checks if not c.passed}
    assert "no_empty_sections" in failed_names


def test_validation_report_to_json_round_trips_check_shape() -> None:
    report = run_validation(
        sections=[], plan=_plan(), chunks=[], topics=[], transcript_word_count=0, max_length_ratio=0.4
    )
    payload = report.to_json()
    assert len(payload) == 10
    assert all({"name", "passed", "details"} <= set(entry) for entry in payload)
