"""End-to-end test of the LangGraph article pipeline (app/ai/graph.py)
against a real Postgres test DB, with a FakeLLMProvider standing in for
the real inference call -- no network, no real LLM, deterministic and
fast, but exercises the actual persistence path every node writes
through (TopicRepository, ArticlePlanRepository, ArticleRepository,
ValidationResultRepository) exactly like a real run would.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.graph import run_article_pipeline
from app.ai.schemas import (
    ArticlePlanResult,
    GeneratedSection,
    PlannedSection,
    TopicAnalysisResult,
    TopicItem,
    TopicMergeDecision,
)
from app.ai.state import ArticlePipelineState, PipelineDeps
from app.config.settings import Settings
from app.models.chunk import Chunk
from app.models.episode import Episode, ProcessingStatus
from app.models.transcript import Transcript
from tests.fakes import FakeLLMProvider


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, app_env="development", llm_provider="groq", groq_api_key="x", llm_model="m", **overrides)


async def _seed(session: AsyncSession) -> tuple[Episode, Transcript]:
    episode = Episode(
        id=uuid.uuid4(),
        youtube_video_id="abc12345678",
        youtube_url="https://www.youtube.com/watch?v=abc12345678",
        status=ProcessingStatus.CHUNKING,
    )
    session.add(episode)
    transcript = Transcript(id=uuid.uuid4(), episode_id=episode.id, language="en")
    session.add(transcript)
    await session.commit()
    return episode, transcript


def _chunks() -> list[Chunk]:
    # Deliberately not persisted -- source_segment_ids/chunk_ids elsewhere
    # in this pipeline are plain UUID arrays, not FK-constrained, so the
    # graph only ever needs these as in-memory state, same as a real run
    # reading them back from ChunkRepository would provide.
    return [
        Chunk(
            id=uuid.uuid4(),
            transcript_id=uuid.uuid4(),
            episode_id=uuid.uuid4(),
            sequence_number=i,
            text=" ".join(["word"] * 200),
            start_ms=i * 60_000,
            end_ms=(i + 1) * 60_000,
            source_segment_ids=[],
            token_count=200,
        )
        for i in range(2)
    ]


def _good_section(heading: str = "Introduction") -> GeneratedSection:
    return GeneratedSection(heading=heading, content=" ".join(["word"] * 150))


async def test_pipeline_persists_topics_plan_article_and_passing_validation(
    db_session: AsyncSession,
) -> None:
    episode, transcript = await _seed(db_session)
    chunks = _chunks()

    llm = FakeLLMProvider(
        structured_responses=[
            TopicAnalysisResult(topics=[TopicItem(title="Topic A", summary="s", chunk_sequence_numbers=[0, 1])]),
            ArticlePlanResult(
                title="The Article",
                introduction_summary="i",
                sections=[PlannedSection(heading="Introduction", supporting_topic_sequence_numbers=[0])],
                conclusion_summary="c",
            ),
            _good_section(),
        ]
    )
    # section_count_min=1: this fixture's single planned section exercises
    # the OTHER checks, not section-count pathology (app/services/
    # article_validation.py::check_section_count).
    deps = PipelineDeps(session=db_session, llm_provider=llm, settings=_settings(section_count_min=1))

    initial_state: ArticlePipelineState = {
        "episode_id": episode.id,
        "transcript_id": transcript.id,
        "chunks": chunks,
        "transcript_word_count": 100_000,  # keeps the length check trivially satisfied
        "revision_count": 0,
        "max_revision_attempts": 2,
    }

    final_state = await run_article_pipeline(deps, initial_state)

    assert len(final_state["topics"]) == 1
    assert final_state["topics"][0].title == "Topic A"
    assert final_state["plan"].title == "The Article"
    assert len(final_state["article"].sections) == 1
    assert final_state["validation_report"].passed
    assert final_state.get("revision_count", 0) == 0

    refreshed_episode = await db_session.get(Episode, episode.id)
    assert refreshed_episode.status == ProcessingStatus.VERIFYING


async def test_pipeline_revises_a_failing_section_until_it_passes(db_session: AsyncSession) -> None:
    episode, transcript = await _seed(db_session)
    chunks = _chunks()

    llm = FakeLLMProvider(
        structured_responses=[
            TopicAnalysisResult(topics=[TopicItem(title="Topic A", summary="s", chunk_sequence_numbers=[0, 1])]),
            ArticlePlanResult(
                title="The Article",
                introduction_summary="i",
                sections=[PlannedSection(heading="Introduction", supporting_topic_sequence_numbers=[0])],
                conclusion_summary="c",
            ),
            GeneratedSection(heading="Introduction", content="too short"),  # fails no_empty_sections
            _good_section(),  # the revision's regenerated section
        ]
    )
    deps = PipelineDeps(session=db_session, llm_provider=llm, settings=_settings(section_count_min=1))

    initial_state: ArticlePipelineState = {
        "episode_id": episode.id,
        "transcript_id": transcript.id,
        "chunks": chunks,
        "transcript_word_count": 100_000,
        "revision_count": 0,
        "max_revision_attempts": 2,
    }

    final_state = await run_article_pipeline(deps, initial_state)

    assert final_state["validation_report"].passed
    assert final_state["revision_count"] == 1
    assert final_state["article"].sections[0].content == _good_section().content


async def test_pipeline_stops_after_max_revision_attempts_even_if_still_failing(
    db_session: AsyncSession,
) -> None:
    episode, transcript = await _seed(db_session)
    chunks = _chunks()

    always_bad = GeneratedSection(heading="Introduction", content="too short")
    llm = FakeLLMProvider(
        structured_responses=[
            TopicAnalysisResult(topics=[TopicItem(title="Topic A", summary="s", chunk_sequence_numbers=[0, 1])]),
            ArticlePlanResult(
                title="The Article",
                introduction_summary="i",
                sections=[PlannedSection(heading="Introduction", supporting_topic_sequence_numbers=[0])],
                conclusion_summary="c",
            ),
            always_bad,  # initial generation
            always_bad,  # revision 1
            always_bad,  # revision 2 (max_revision_attempts=2)
        ]
    )
    deps = PipelineDeps(session=db_session, llm_provider=llm, settings=_settings())

    initial_state: ArticlePipelineState = {
        "episode_id": episode.id,
        "transcript_id": transcript.id,
        "chunks": chunks,
        "transcript_word_count": 100_000,
        "revision_count": 0,
        "max_revision_attempts": 2,
    }

    final_state = await run_article_pipeline(deps, initial_state)

    assert not final_state["validation_report"].passed
    assert final_state["revision_count"] == 2  # stopped at the bound, not looping forever


async def test_pipeline_passes_narrative_context_and_already_covered_material_to_later_sections(
    db_session: AsyncSession,
) -> None:
    """End-to-end proof of the editorial-rework wiring (Batch 3): a real
    planning_node -> section_generation_node round-trip through actual
    Postgres persistence (not just section_generation_prompt in
    isolation) must carry narrative_purpose/transition_from_previous,
    introduction_summary/conclusion_summary, and both "already covered"
    layers (planned key_ideas + the immediately preceding section's own
    generated tail) into each later section's actual prompt -- and must
    NOT leak a non-immediate section's content (only the immediately
    preceding one)."""
    episode, transcript = await _seed(db_session)
    chunks = _chunks()

    def _section_content(tail_marker: str) -> str:
        # Distinct lead text per section (not just the tail) -- identical
        # filler paragraphs across sections would otherwise legitimately
        # trip check_no_duplicate_paragraphs and force an unplanned
        # revision round this test isn't set up for.
        return f"Lead paragraph {tail_marker}. " + " ".join(["word"] * 140) + f"\n\nClosing thought {tail_marker}."

    llm = FakeLLMProvider(
        structured_responses=[
            TopicAnalysisResult(topics=[TopicItem(title="Topic A", summary="s", chunk_sequence_numbers=[0, 1])]),
            ArticlePlanResult(
                title="The Article",
                introduction_summary="central question X",
                sections=[
                    PlannedSection(
                        heading="Intro",
                        key_ideas=["idea A"],
                        supporting_topic_sequence_numbers=[0],
                        narrative_purpose="purpose A",
                        transition_from_previous="",
                    ),
                    PlannedSection(
                        heading="Body",
                        key_ideas=["idea B"],
                        supporting_topic_sequence_numbers=[0],
                        narrative_purpose="purpose B",
                        transition_from_previous="because of A",
                    ),
                    PlannedSection(
                        heading="Conclusion",
                        key_ideas=["idea C"],
                        supporting_topic_sequence_numbers=[0],
                        narrative_purpose="purpose C",
                        transition_from_previous="because of B",
                    ),
                ],
                conclusion_summary="synthesis Y",
            ),
            GeneratedSection(heading="Intro", content=_section_content("TAIL0")),
            GeneratedSection(heading="Body", content=_section_content("TAIL1")),
            GeneratedSection(heading="Conclusion", content=_section_content("TAIL2")),
        ]
    )
    deps = PipelineDeps(session=db_session, llm_provider=llm, settings=_settings(section_count_min=1))

    initial_state: ArticlePipelineState = {
        "episode_id": episode.id,
        "transcript_id": transcript.id,
        "chunks": chunks,
        "transcript_word_count": 100_000,
        "revision_count": 0,
        "max_revision_attempts": 2,
    }

    final_state = await run_article_pipeline(deps, initial_state)
    assert final_state["validation_report"].passed

    # The persisted plan itself carries the new fields through planning_node.
    plan_sections = final_state["plan"].sections
    assert plan_sections[1]["narrative_purpose"] == "purpose B"
    assert plan_sections[1]["transition_from_previous"] == "because of A"

    # structured_calls: [0]=topic_analysis, [1]=planning, [2..4]=sections 0,1,2.
    def _prompt_text(index: int) -> str:
        messages, system, _ = llm.structured_calls[index]
        return system + messages[0].content

    intro_prompt = _prompt_text(2)
    body_prompt = _prompt_text(3)
    conclusion_prompt = _prompt_text(4)

    # Section 0 (opening): gets the introduction-specific block, and has
    # no "already covered" block at all (nothing precedes it).
    assert "ARTICLE'S OPENING SECTION" in intro_prompt
    assert "central question X" in intro_prompt
    assert "ALREADY COVERED" not in intro_prompt

    # Section 1 (middle): gets its own narrative_purpose/transition, layer-1
    # context from section 0's PLANNED key idea, and layer-2 context from
    # section 0's ACTUAL generated tail -- but nothing from section 2
    # (which doesn't exist yet), and no intro/conclusion block.
    assert "purpose B" in body_prompt
    assert "because of A" in body_prompt
    assert "idea A" in body_prompt
    assert "Closing thought TAIL0" in body_prompt
    assert "idea C" not in body_prompt
    assert "ARTICLE'S OPENING SECTION" not in body_prompt
    assert "ARTICLE'S CLOSING SECTION" not in body_prompt

    # Section 2 (closing): gets the conclusion-specific block, layer-1
    # context from BOTH earlier sections' planned key ideas, but layer-2
    # context ONLY from section 1's tail -- section 0's tail must not leak
    # through (only the immediately preceding section's excerpt is passed).
    assert "ARTICLE'S CLOSING SECTION" in conclusion_prompt
    assert "synthesis Y" in conclusion_prompt
    assert "idea A" in conclusion_prompt
    assert "idea B" in conclusion_prompt
    assert "Closing thought TAIL1" in conclusion_prompt
    assert "Closing thought TAIL0" not in conclusion_prompt
    # Regression: "return to the central question the introduction raised"
    # is an empty instruction without the actual text of what that was --
    # the closing section must also see introduction_summary, not just
    # conclusion_summary.
    assert "central question X" in conclusion_prompt

    # Soft per-section word-count guidance derived from
    # Settings.article_target_word_count_min/max (defaults 5500/6500) split
    # across 3 planned sections.
    assert "1833" in body_prompt
    assert "2166" in body_prompt


async def test_topic_analysis_batches_when_chunks_exceed_token_budget(db_session: AsyncSession) -> None:
    episode, transcript = await _seed(db_session)
    chunks = _chunks()  # 2 chunks, ~50 tokens each (200 words / 4)

    llm = FakeLLMProvider(
        structured_responses=[
            TopicAnalysisResult(topics=[TopicItem(title="Batch 0", summary="s", chunk_sequence_numbers=[0])]),
            TopicAnalysisResult(topics=[TopicItem(title="Batch 1", summary="s", chunk_sequence_numbers=[1])]),
            # The boundary-merge call: both topics touch the batch seam
            # (chunk 0 is batch 0's last chunk, chunk 1 is batch 1's first),
            # so both are sent as candidates -- an empty decision means
            # neither should merge, so both are kept exactly as produced.
            TopicMergeDecision(merges=[]),
            ArticlePlanResult(
                title="T", introduction_summary="i", sections=[], conclusion_summary="c"
            ),
        ]
    )
    # Force each chunk into its own batch.
    deps = PipelineDeps(session=db_session, llm_provider=llm, settings=_settings(topic_analysis_token_budget=10))

    initial_state: ArticlePipelineState = {
        "episode_id": episode.id,
        "transcript_id": transcript.id,
        "chunks": chunks,
        "transcript_word_count": 100_000,
        "revision_count": 0,
        "max_revision_attempts": 2,
    }

    final_state = await run_article_pipeline(deps, initial_state)

    assert len(final_state["topics"]) == 2
    assert {t.title for t in final_state["topics"]} == {"Batch 0", "Batch 1"}
