import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.episode import Episode, ProcessingStatus
from app.models.transcript import Transcript
from app.repositories.article_plan_repository import ArticlePlanRepository, PlannedSectionData
from app.repositories.article_repository import ArticleRepository, ArticleSectionCandidate
from app.repositories.topic_repository import TopicCandidate, TopicRepository
from app.repositories.validation_result_repository import ValidationResultRepository


async def _seed_episode_and_transcript(session: AsyncSession) -> tuple[Episode, Transcript]:
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


# --- TopicRepository -----------------------------------------------------------------


async def test_topic_replace_all_persists_topics(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    repo = TopicRepository(db_session)
    chunk_id = uuid.uuid4()

    candidates = [
        TopicCandidate(
            sequence_number=0,
            title="Scaling laws",
            summary="Discussion of scaling laws.",
            chunk_ids=[chunk_id],
            key_claims=[{"text": "bigger is not always better", "speaker": None, "claim_type": "opinion"}],
            subtopics=["compute", "data"],
        )
    ]
    await repo.replace_all(transcript_id=transcript.id, episode_id=episode.id, candidates=candidates)
    await db_session.commit()

    stored = await repo.get_by_transcript_id(transcript.id)
    assert len(stored) == 1
    assert stored[0].title == "Scaling laws"
    assert stored[0].chunk_ids == [chunk_id]
    assert stored[0].key_claims[0]["claim_type"] == "opinion"
    assert stored[0].episode_id == episode.id


async def test_topic_replace_all_is_idempotent(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    repo = TopicRepository(db_session)
    candidates = [TopicCandidate(sequence_number=0, title="A", summary="a", chunk_ids=[])]

    await repo.replace_all(transcript_id=transcript.id, episode_id=episode.id, candidates=candidates)
    await db_session.commit()
    first_ids = {t.id for t in await repo.get_by_transcript_id(transcript.id)}

    await repo.replace_all(transcript_id=transcript.id, episode_id=episode.id, candidates=candidates)
    await db_session.commit()
    second = await repo.get_by_transcript_id(transcript.id)

    assert len(second) == 1
    assert first_ids.isdisjoint({t.id for t in second})


# --- ArticlePlanRepository ------------------------------------------------------------


async def test_article_plan_replace_persists_sections(db_session: AsyncSession) -> None:
    episode, _ = await _seed_episode_and_transcript(db_session)
    repo = ArticlePlanRepository(db_session)

    sections = [
        PlannedSectionData(
            sequence_number=0,
            heading="Introduction",
            key_ideas=["idea one"],
            supporting_topic_ids=[str(uuid.uuid4())],
            supporting_chunk_ids=[str(uuid.uuid4())],
            viewpoints=["host thinks X"],
            narrative_purpose="Establish the central principle.",
            transition_from_previous="",
        )
    ]
    await repo.replace(
        episode_id=episode.id,
        title="The Article",
        introduction_summary="intro",
        conclusion_summary="outro",
        sections=sections,
    )
    await db_session.commit()

    stored = await repo.get_by_episode_id(episode.id)
    assert stored is not None
    assert stored.title == "The Article"
    assert stored.sections[0]["heading"] == "Introduction"
    assert stored.sections[0]["key_ideas"] == ["idea one"]
    assert stored.sections[0]["narrative_purpose"] == "Establish the central principle."
    assert stored.sections[0]["transition_from_previous"] == ""


async def test_article_plan_replace_defaults_narrative_fields_to_blank(db_session: AsyncSession) -> None:
    """A PlannedSectionData built without the two new fields (e.g. older
    call-site code, or a test fixture written before this change) must
    still persist cleanly -- backward-compatible defaults, not a required
    field."""
    episode, _ = await _seed_episode_and_transcript(db_session)
    repo = ArticlePlanRepository(db_session)

    sections = [
        PlannedSectionData(
            sequence_number=0,
            heading="Introduction",
            key_ideas=[],
            supporting_topic_ids=[],
            supporting_chunk_ids=[],
        )
    ]
    await repo.replace(
        episode_id=episode.id, title="T", introduction_summary="i", conclusion_summary="c", sections=sections
    )
    await db_session.commit()

    stored = await repo.get_by_episode_id(episode.id)
    assert stored.sections[0]["narrative_purpose"] == ""
    assert stored.sections[0]["transition_from_previous"] == ""


async def test_article_plan_replace_cascades_to_existing_article(db_session: AsyncSession) -> None:
    episode, _ = await _seed_episode_and_transcript(db_session)
    plan_repo = ArticlePlanRepository(db_session)
    article_repo = ArticleRepository(db_session)

    plan = await plan_repo.replace(
        episode_id=episode.id, title="v1", introduction_summary="i", conclusion_summary="c", sections=[]
    )
    await db_session.commit()
    await article_repo.replace(
        episode_id=episode.id, article_plan_id=plan.id, title="v1 article", revision_count=0, sections=[]
    )
    await db_session.commit()
    assert await article_repo.get_by_episode_id(episode.id) is not None

    # Regenerating the plan invalidates the article built from the old one.
    await plan_repo.replace(
        episode_id=episode.id, title="v2", introduction_summary="i2", conclusion_summary="c2", sections=[]
    )
    await db_session.commit()

    assert await article_repo.get_by_episode_id(episode.id) is None


# --- ArticleRepository -----------------------------------------------------------------


async def test_article_replace_persists_article_and_sections(db_session: AsyncSession) -> None:
    episode, _ = await _seed_episode_and_transcript(db_session)
    plan_repo = ArticlePlanRepository(db_session)
    article_repo = ArticleRepository(db_session)

    plan = await plan_repo.replace(
        episode_id=episode.id, title="p", introduction_summary="i", conclusion_summary="c", sections=[]
    )
    await db_session.commit()

    chunk_id, topic_id = uuid.uuid4(), uuid.uuid4()
    sections = [
        ArticleSectionCandidate(
            sequence_number=0,
            heading="Intro",
            content="Some generated prose.",
            supporting_chunk_ids=[chunk_id],
            supporting_topic_ids=[topic_id],
        )
    ]
    await article_repo.replace(
        episode_id=episode.id, article_plan_id=plan.id, title="Title", revision_count=0, sections=sections
    )
    await db_session.commit()

    stored = await article_repo.get_by_episode_id(episode.id)
    assert stored is not None
    assert stored.title == "Title"
    assert len(stored.sections) == 1
    assert stored.sections[0].content == "Some generated prose."
    assert stored.sections[0].supporting_chunk_ids == [chunk_id]


async def test_article_replace_increments_revision_count(db_session: AsyncSession) -> None:
    episode, _ = await _seed_episode_and_transcript(db_session)
    plan_repo = ArticlePlanRepository(db_session)
    article_repo = ArticleRepository(db_session)
    plan = await plan_repo.replace(
        episode_id=episode.id, title="p", introduction_summary="i", conclusion_summary="c", sections=[]
    )
    await db_session.commit()

    await article_repo.replace(
        episode_id=episode.id, article_plan_id=plan.id, title="t", revision_count=0, sections=[]
    )
    await db_session.commit()
    await article_repo.replace(
        episode_id=episode.id, article_plan_id=plan.id, title="t", revision_count=1, sections=[]
    )
    await db_session.commit()

    stored = await article_repo.get_by_episode_id(episode.id)
    assert stored.revision_count == 1


async def test_article_replace_is_idempotent_on_rerun(db_session: AsyncSession) -> None:
    episode, _ = await _seed_episode_and_transcript(db_session)
    plan_repo = ArticlePlanRepository(db_session)
    article_repo = ArticleRepository(db_session)
    plan = await plan_repo.replace(
        episode_id=episode.id, title="p", introduction_summary="i", conclusion_summary="c", sections=[]
    )
    await db_session.commit()

    sections = [ArticleSectionCandidate(sequence_number=0, heading="H", content="C")]
    await article_repo.replace(
        episode_id=episode.id, article_plan_id=plan.id, title="t", revision_count=0, sections=sections
    )
    await db_session.commit()
    first_id = (await article_repo.get_by_episode_id(episode.id)).id

    await article_repo.replace(
        episode_id=episode.id, article_plan_id=plan.id, title="t", revision_count=0, sections=sections
    )
    await db_session.commit()
    second = await article_repo.get_by_episode_id(episode.id)

    assert second.id != first_id  # delete-then-reinsert, same as Chunk/Topic
    assert len(second.sections) == 1


# --- ValidationResultRepository ---------------------------------------------------------


async def test_validation_result_create_and_get_latest(db_session: AsyncSession) -> None:
    episode, _ = await _seed_episode_and_transcript(db_session)
    plan_repo = ArticlePlanRepository(db_session)
    article_repo = ArticleRepository(db_session)
    validation_repo = ValidationResultRepository(db_session)

    plan = await plan_repo.replace(
        episode_id=episode.id, title="p", introduction_summary="i", conclusion_summary="c", sections=[]
    )
    await db_session.commit()
    article = await article_repo.replace(
        episode_id=episode.id, article_plan_id=plan.id, title="t", revision_count=0, sections=[]
    )
    await db_session.commit()

    validation_repo.create(article_id=article.id, passed=False, checks=[{"name": "x", "passed": False}])
    await db_session.commit()
    validation_repo.create(article_id=article.id, passed=True, checks=[{"name": "x", "passed": True}])
    await db_session.commit()

    latest = await validation_repo.get_latest_by_article_id(article.id)
    assert latest is not None
    assert latest.passed is True

    # Both runs are kept -- not delete-then-replace like the other repos.
    from app.models.validation_result import ValidationResult

    count = await db_session.scalar(
        select(func.count()).select_from(ValidationResult).where(ValidationResult.article_id == article.id)
    )
    assert count == 2


async def test_deleting_article_cascades_to_validation_results(db_session: AsyncSession) -> None:
    episode, _ = await _seed_episode_and_transcript(db_session)
    plan_repo = ArticlePlanRepository(db_session)
    article_repo = ArticleRepository(db_session)
    validation_repo = ValidationResultRepository(db_session)

    plan = await plan_repo.replace(
        episode_id=episode.id, title="p", introduction_summary="i", conclusion_summary="c", sections=[]
    )
    await db_session.commit()
    article = await article_repo.replace(
        episode_id=episode.id, article_plan_id=plan.id, title="t", revision_count=0, sections=[]
    )
    await db_session.commit()
    validation_repo.create(article_id=article.id, passed=True, checks=[])
    await db_session.commit()

    await db_session.delete(article)
    await db_session.commit()

    assert await validation_repo.get_latest_by_article_id(article.id) is None
