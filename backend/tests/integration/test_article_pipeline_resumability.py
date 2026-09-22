"""Batch 2B: pipeline resumability / incremental persistence, exercised
against a real Postgres test DB (same infrastructure as
test_article_pipeline_graph.py) so the actual persistence path -- not
just in-memory state -- is what proves a stage was skipped or a section
reused. Every test asserts an exact FakeLLMProvider.structured_calls
count: the point isn't just that the pipeline produces the right output,
but that it demonstrably avoided LLM calls for work already done.
"""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.graph import run_article_pipeline
from app.ai.schemas import ArticlePlanResult, GeneratedSection, PlannedSection, TopicAnalysisResult, TopicItem
from app.ai.state import ArticlePipelineState, PipelineDeps
from app.config.settings import Settings
from app.core.exceptions import LLMProviderTransientError
from app.models.chunk import Chunk
from app.models.episode import Episode, ProcessingStatus
from app.models.transcript import Transcript
from app.repositories.article_plan_repository import ArticlePlanRepository, PlannedSectionData
from app.repositories.article_repository import ArticleRepository, ArticleSectionCandidate
from app.repositories.topic_repository import TopicCandidate, TopicRepository
from tests.fakes import FakeLLMProvider


def _settings(**overrides) -> Settings:
    overrides.setdefault("section_count_min", 1)  # these fixtures use 1-3 sections throughout
    return Settings(_env_file=None, app_env="development", llm_provider="groq", groq_api_key="x", llm_model="m", **overrides)


async def _seed(session: AsyncSession) -> tuple[Episode, Transcript]:
    # A fresh, unique youtube_video_id per call -- episodes.youtube_video_id
    # is unique, and a couple of these tests seed more than one episode.
    video_id = uuid.uuid4().hex[:11]
    episode = Episode(
        id=uuid.uuid4(),
        youtube_video_id=video_id,
        youtube_url=f"https://www.youtube.com/watch?v={video_id}",
        status=ProcessingStatus.CHUNKING,
    )
    session.add(episode)
    transcript = Transcript(id=uuid.uuid4(), episode_id=episode.id, language="en")
    session.add(transcript)
    await session.commit()
    return episode, transcript


def _chunks() -> list[Chunk]:
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


def _initial_state(episode: Episode, transcript: Transcript, chunks: list[Chunk]) -> ArticlePipelineState:
    return {
        "episode_id": episode.id,
        "transcript_id": transcript.id,
        "chunks": chunks,
        "transcript_word_count": 100_000,  # keeps article_length trivially satisfied throughout
        "revision_count": 0,
        "max_revision_attempts": 2,
    }


def _topic_analysis_response(chunks: list[Chunk]) -> TopicAnalysisResult:
    return TopicAnalysisResult(
        topics=[TopicItem(title="Topic A", summary="s", chunk_sequence_numbers=[0, 1])]
    )


def _three_section_plan() -> ArticlePlanResult:
    return ArticlePlanResult(
        title="The Article",
        introduction_summary="i",
        sections=[
            PlannedSection(heading="Intro", supporting_topic_sequence_numbers=[0]),
            PlannedSection(heading="Body", supporting_topic_sequence_numbers=[0]),
            PlannedSection(heading="Conclusion", supporting_topic_sequence_numbers=[0]),
        ],
        conclusion_summary="c",
    )


def _generated_section(heading: str, marker: str) -> GeneratedSection:
    # Distinct content per heading (leading marker word) so
    # check_no_duplicate_sections never trips across these fixtures.
    return GeneratedSection(heading=heading, content=" ".join([marker] + ["word"] * 149))


async def _seed_topics(session: AsyncSession, episode: Episode, transcript: Transcript, chunks: list[Chunk]):
    topics = await TopicRepository(session).replace_all(
        transcript_id=transcript.id,
        episode_id=episode.id,
        candidates=[
            TopicCandidate(
                sequence_number=0, title="Topic A", summary="s", chunk_ids=[chunks[0].id, chunks[1].id]
            )
        ],
    )
    await session.commit()
    return topics


async def _seed_plan(session: AsyncSession, episode: Episode, topic_id: uuid.UUID, chunks: list[Chunk]):
    plan = await ArticlePlanRepository(session).replace(
        episode_id=episode.id,
        title="The Article",
        introduction_summary="i",
        conclusion_summary="c",
        sections=[
            PlannedSectionData(
                sequence_number=i,
                heading=heading,
                key_ideas=[],
                supporting_topic_ids=[str(topic_id)],
                supporting_chunk_ids=[str(c.id) for c in chunks],
            )
            for i, heading in enumerate(["Intro", "Body", "Conclusion"])
        ],
    )
    await session.commit()
    return plan


# --- 1. Fresh job: every stage runs, every missing section generates -----------------


async def test_fresh_job_runs_every_stage_and_generates_every_section(db_session: AsyncSession) -> None:
    episode, transcript = await _seed(db_session)
    chunks = _chunks()

    llm = FakeLLMProvider(
        structured_responses=[
            _topic_analysis_response(chunks),
            _three_section_plan(),
            _generated_section("Intro", "intro"),
            _generated_section("Body", "body"),
            _generated_section("Conclusion", "conclusion"),
        ]
    )
    deps = PipelineDeps(session=db_session, llm_provider=llm, settings=_settings())

    final_state = await run_article_pipeline(deps, _initial_state(episode, transcript, chunks))

    assert len(final_state["topics"]) == 1
    assert final_state["plan"].title == "The Article"
    assert len(final_state["article"].sections) == 3
    assert final_state["validation_report"].passed
    # Exactly 5 calls: 1 topic-analysis + 1 planning + 3 sections. Nothing
    # extra, nothing skipped.
    assert len(llm.structured_calls) == 5


# --- 2. Topic analysis already complete: its LLM call is skipped; planning still runs ---


async def test_resumes_when_topic_analysis_already_complete(db_session: AsyncSession) -> None:
    episode, transcript = await _seed(db_session)
    chunks = _chunks()
    existing_topics = await _seed_topics(db_session, episode, transcript, chunks)

    llm = FakeLLMProvider(
        structured_responses=[
            _three_section_plan(),
            _generated_section("Intro", "intro"),
            _generated_section("Body", "body"),
            _generated_section("Conclusion", "conclusion"),
        ]
    )
    deps = PipelineDeps(session=db_session, llm_provider=llm, settings=_settings())

    final_state = await run_article_pipeline(deps, _initial_state(episode, transcript, chunks))

    assert {t.id for t in final_state["topics"]} == {t.id for t in existing_topics}
    assert len(final_state["article"].sections) == 3
    # Exactly 4 calls: planning + 3 sections -- topic analysis's own LLM
    # call never happened.
    assert len(llm.structured_calls) == 4
    assert all(response_model is not TopicAnalysisResult for _, _, response_model in llm.structured_calls)


# --- 3. Topic analysis + planning already complete: both skipped; sections generate -----


async def test_resumes_when_topic_analysis_and_planning_already_complete(db_session: AsyncSession) -> None:
    episode, transcript = await _seed(db_session)
    chunks = _chunks()
    topics = await _seed_topics(db_session, episode, transcript, chunks)
    plan = await _seed_plan(db_session, episode, topics[0].id, chunks)

    llm = FakeLLMProvider(
        structured_responses=[
            _generated_section("Intro", "intro"),
            _generated_section("Body", "body"),
            _generated_section("Conclusion", "conclusion"),
        ]
    )
    deps = PipelineDeps(session=db_session, llm_provider=llm, settings=_settings())

    final_state = await run_article_pipeline(deps, _initial_state(episode, transcript, chunks))

    assert final_state["plan"].id == plan.id  # reused, not regenerated
    assert len(final_state["article"].sections) == 3
    # Exactly 3 calls: the 3 sections -- neither topic analysis nor
    # planning's LLM call happened.
    assert len(llm.structured_calls) == 3


# --- 4. Some sections already persisted: only the missing ones generate -----------------


async def test_resumes_when_some_sections_already_persisted(db_session: AsyncSession) -> None:
    episode, transcript = await _seed(db_session)
    chunks = _chunks()
    topics = await _seed_topics(db_session, episode, transcript, chunks)
    plan = await _seed_plan(db_session, episode, topics[0].id, chunks)
    await ArticleRepository(db_session).replace(
        episode_id=episode.id,
        article_plan_id=plan.id,
        title=plan.title,
        revision_count=0,
        sections=[
            ArticleSectionCandidate(
                sequence_number=0,
                heading="Intro",
                content=" ".join(["intro"] + ["word"] * 149),
                supporting_chunk_ids=[c.id for c in chunks],
                supporting_topic_ids=[topics[0].id],
            ),
            ArticleSectionCandidate(
                sequence_number=1,
                heading="Body",
                content=" ".join(["body"] + ["word"] * 149),
                supporting_chunk_ids=[c.id for c in chunks],
                supporting_topic_ids=[topics[0].id],
            ),
        ],
    )
    await db_session.commit()

    llm = FakeLLMProvider(structured_responses=[_generated_section("Conclusion", "conclusion")])
    deps = PipelineDeps(session=db_session, llm_provider=llm, settings=_settings())

    final_state = await run_article_pipeline(deps, _initial_state(episode, transcript, chunks))

    assert {s.heading for s in final_state["article"].sections} == {"Intro", "Body", "Conclusion"}
    # Exactly 1 call: only the missing section.
    assert len(llm.structured_calls) == 1


# --- 5. Crash after several sections: persisted ones remain; retry resumes --------------


async def test_crash_after_several_sections_resumes_from_the_missing_ones(db_session: AsyncSession) -> None:
    episode, transcript = await _seed(db_session)
    chunks = _chunks()
    settings = _settings()

    first_llm = FakeLLMProvider(
        structured_responses=[
            _topic_analysis_response(chunks),
            _three_section_plan(),
            _generated_section("Intro", "intro"),
            _generated_section("Body", "body"),
            LLMProviderTransientError("simulated worker crash generating the 3rd section"),
        ]
    )
    deps1 = PipelineDeps(session=db_session, llm_provider=first_llm, settings=settings)
    with pytest.raises(LLMProviderTransientError):
        await run_article_pipeline(deps1, _initial_state(episode, transcript, chunks))

    # Sections generated before the crash must have survived it --
    # committed individually, not lost with the in-flight 3rd section.
    article = await ArticleRepository(db_session).get_by_episode_id(episode.id)
    assert article is not None
    assert {s.heading for s in article.sections} == {"Intro", "Body"}

    # The retry: only the section that never completed is requested.
    second_llm = FakeLLMProvider(structured_responses=[_generated_section("Conclusion", "conclusion")])
    deps2 = PipelineDeps(session=db_session, llm_provider=second_llm, settings=settings)
    final_state = await run_article_pipeline(deps2, _initial_state(episode, transcript, chunks))

    assert {s.heading for s in final_state["article"].sections} == {"Intro", "Body", "Conclusion"}
    assert len(second_llm.structured_calls) == 1
    assert final_state["validation_report"].passed


# --- 6. A failed section is never persisted as successful --------------------------------


async def test_failed_section_is_not_persisted_and_retry_regenerates_it(db_session: AsyncSession) -> None:
    episode, transcript = await _seed(db_session)
    chunks = _chunks()
    settings = _settings()

    first_llm = FakeLLMProvider(
        structured_responses=[
            _topic_analysis_response(chunks),
            ArticlePlanResult(
                title="T",
                introduction_summary="i",
                sections=[PlannedSection(heading="Intro", supporting_topic_sequence_numbers=[0])],
                conclusion_summary="c",
            ),
            LLMProviderTransientError("simulated failure generating the only section"),
        ]
    )
    deps1 = PipelineDeps(session=db_session, llm_provider=first_llm, settings=settings)
    with pytest.raises(LLMProviderTransientError):
        await run_article_pipeline(deps1, _initial_state(episode, transcript, chunks))

    # get_or_create persists the Article row eagerly, but the section
    # itself -- whose generation call raised -- must never be written.
    article = await ArticleRepository(db_session).get_by_episode_id(episode.id)
    assert article is not None
    assert article.sections == []

    second_llm = FakeLLMProvider(structured_responses=[_generated_section("Intro", "intro")])
    deps2 = PipelineDeps(session=db_session, llm_provider=second_llm, settings=settings)
    final_state = await run_article_pipeline(deps2, _initial_state(episode, transcript, chunks))

    assert len(final_state["article"].sections) == 1
    assert len(second_llm.structured_calls) == 1


# --- 7. All sections already exist: zero section-generation calls; validation still runs --


async def test_all_sections_already_exist_generates_nothing_and_validates_correctly(
    db_session: AsyncSession,
) -> None:
    episode, transcript = await _seed(db_session)
    chunks = _chunks()
    topics = await _seed_topics(db_session, episode, transcript, chunks)
    plan = await _seed_plan(db_session, episode, topics[0].id, chunks)
    await ArticleRepository(db_session).replace(
        episode_id=episode.id,
        article_plan_id=plan.id,
        title=plan.title,
        revision_count=0,
        sections=[
            ArticleSectionCandidate(
                sequence_number=i,
                heading=heading,
                content=" ".join([marker] + ["word"] * 149),
                supporting_chunk_ids=[c.id for c in chunks],
                supporting_topic_ids=[topics[0].id],
            )
            for i, (heading, marker) in enumerate(
                [("Intro", "intro"), ("Body", "body"), ("Conclusion", "conclusion")]
            )
        ],
    )
    await db_session.commit()

    llm = FakeLLMProvider(structured_responses=[])  # raises if anything at all is called
    deps = PipelineDeps(session=db_session, llm_provider=llm, settings=_settings())

    final_state = await run_article_pipeline(deps, _initial_state(episode, transcript, chunks))

    assert len(final_state["article"].sections) == 3
    assert final_state["validation_report"].passed
    assert len(llm.structured_calls) == 0


# --- 8. A stale article from a different plan is never reused ----------------------------


async def test_get_or_create_never_reuses_an_article_belonging_to_a_different_plan(
    db_session: AsyncSession,
) -> None:
    # ArticlePlan.episode_id is unique, so two plans can never legitimately
    # coexist for the SAME episode -- by the time a second plan exists for
    # an episode, ArticlePlanRepository.replace's cascade has already
    # deleted any Article tied to the old one. This test instead proves
    # get_or_create's OWN defensive check directly (never trust a caller's
    # article_plan_id blindly), the way it would matter if section_generation
    # _node were ever passed a stale plan by mistake: a second, unrelated
    # episode's real plan id stands in for "some plan id that isn't the one
    # this episode's existing Article actually belongs to".
    episode, _transcript = await _seed(db_session)
    plans = ArticlePlanRepository(db_session)
    plan = await plans.replace(
        episode_id=episode.id,
        title="Plan",
        introduction_summary="i",
        conclusion_summary="c",
        sections=[PlannedSectionData(sequence_number=0, heading="Intro", key_ideas=[], supporting_topic_ids=[], supporting_chunk_ids=[])],
    )
    await db_session.commit()

    articles = ArticleRepository(db_session)
    old_article = await articles.replace(
        episode_id=episode.id,
        article_plan_id=plan.id,
        title="Plan",
        revision_count=0,
        sections=[ArticleSectionCandidate(sequence_number=0, heading="Intro", content="x" * 100)],
    )
    await db_session.commit()

    other_episode, _other_transcript = await _seed(db_session)
    other_plan = await plans.replace(
        episode_id=other_episode.id,
        title="Other Plan",
        introduction_summary="i",
        conclusion_summary="c",
        sections=[PlannedSectionData(sequence_number=0, heading="Other Intro", key_ideas=[], supporting_topic_ids=[], supporting_chunk_ids=[])],
    )
    await db_session.commit()

    result = await articles.get_or_create(
        episode_id=episode.id, article_plan_id=other_plan.id, title="Other Plan", revision_count=0
    )
    await db_session.commit()

    assert result.id != old_article.id
    assert result.article_plan_id == other_plan.id
    assert result.sections == []
