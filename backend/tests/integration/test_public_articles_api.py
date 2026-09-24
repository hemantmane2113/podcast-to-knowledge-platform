import uuid

from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_article_service
from app.main import app
from app.models.episode import Episode, ProcessingStatus
from app.models.transcript import Transcript
from app.repositories.article_plan_repository import ArticlePlanRepository
from app.repositories.article_repository import ArticleRepository, ArticleSectionCandidate
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.episode_repository import EpisodeRepository
from app.repositories.validation_result_repository import ValidationResultRepository
from app.services.article_service import ArticleService
from app.services.chunking_service import ChunkCandidate
from tests.fakes import FakeJobQueue


async def _client(db_session: AsyncSession) -> AsyncClient:
    async def override_get_article_service():
        yield ArticleService(db_session, FakeJobQueue())

    app.dependency_overrides[get_article_service] = override_get_article_service
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _seed_episode_and_transcript(session: AsyncSession) -> tuple[Episode, Transcript]:
    # A unique suffix per call -- several tests below seed more than one
    # episode, and youtube_video_id is unique.
    video_id = f"vid{uuid.uuid4().hex[:8]}"
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


async def _seed_article(
    session: AsyncSession, episode: Episode, transcript: Transcript, *, publish: bool
) -> tuple[uuid.UUID, uuid.UUID]:
    """Seeds a chunk, plan, a two-section article, and a passing
    validation result -- optionally publishing it via the same
    ArticleService.publish_article path the API uses, not by hand-setting
    Episode.status, so these tests exercise the real publish lifecycle
    rather than assuming its internals. Returns (chunk_id, article_id)."""
    chunk = (
        await ChunkRepository(session).replace_all(
            transcript_id=transcript.id,
            episode_id=episode.id,
            candidates=[
                ChunkCandidate(
                    sequence_number=0, text="Some source text.", start_ms=1_000, end_ms=5_000,
                    source_segment_ids=[], token_count=5,
                )
            ],
        )
    )[0]
    await session.commit()

    plan = await ArticlePlanRepository(session).replace(
        episode_id=episode.id, title="p", introduction_summary="i", conclusion_summary="c", sections=[]
    )
    await session.commit()
    article = await ArticleRepository(session).replace(
        episode_id=episode.id,
        article_plan_id=plan.id,
        title="The Article",
        revision_count=0,
        sections=[
            ArticleSectionCandidate(
                sequence_number=1, heading="Second", content="Second section prose.", supporting_chunk_ids=[chunk.id]
            ),
            ArticleSectionCandidate(
                sequence_number=0, heading="First", content="First section prose.", supporting_chunk_ids=[chunk.id]
            ),
        ],
    )
    await session.commit()
    ValidationResultRepository(session).create(article_id=article.id, passed=True, checks=[])
    await session.commit()

    if publish:
        await ArticleService(session, FakeJobQueue()).publish_article(episode.id)

    return chunk.id, article.id


# --- GET /articles/{episode_id} ----------------------------------------------------------


async def test_public_article_unknown_episode_returns_404(db_session: AsyncSession) -> None:
    client = await _client(db_session)
    try:
        response = await client.get(f"/api/v1/articles/{uuid.uuid4()}")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    # A genuinely nonexistent episode -- same EPISODE_NOT_FOUND the
    # private GET .../article endpoint already returns for this case.
    assert response.status_code == 404
    assert response.json()["code"] == "EPISODE_NOT_FOUND"


async def test_public_article_not_yet_published_is_not_accessible(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    await _seed_article(db_session, episode, transcript, publish=False)

    client = await _client(db_session)
    try:
        response = await client.get(f"/api/v1/articles/{episode.id}")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    # Same 404/ARTICLE_NOT_FOUND as a genuinely nonexistent article --
    # existence of an unpublished article is never leaked.
    assert response.status_code == 404
    assert response.json()["code"] == "ARTICLE_NOT_FOUND"


async def test_public_article_published_is_accessible(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    chunk_id, _ = await _seed_article(db_session, episode, transcript, publish=True)

    client = await _client(db_session)
    try:
        response = await client.get(f"/api/v1/articles/{episode.id}")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 200
    body = response.json()
    assert body["episode_id"] == str(episode.id)
    assert body["title"] == "The Article"
    assert body["published_at"] is not None
    assert len(body["sections"]) == 2

    # Never exposed: internal article id, validation internals, job data.
    assert "id" not in body
    assert "latest_validation" not in body
    assert "revision_count" not in body

    section = body["sections"][0]
    assert set(section.keys()) == {"sequence_number", "heading", "content", "supporting_sources"}
    assert len(section["supporting_sources"]) == 1
    source = section["supporting_sources"][0]
    assert source["start_ms"] == 1_000
    assert source["end_ms"] == 5_000
    assert "id" not in source  # the raw chunk UUID is never exposed publicly


async def test_public_article_sections_are_returned_in_sequence_order(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    # _seed_article deliberately inserts sequence_number 1 before 0.
    await _seed_article(db_session, episode, transcript, publish=True)

    client = await _client(db_session)
    try:
        response = await client.get(f"/api/v1/articles/{episode.id}")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 200
    sections = response.json()["sections"]
    assert [s["sequence_number"] for s in sections] == [0, 1]
    assert sections[0]["heading"] == "First"
    assert sections[1]["heading"] == "Second"


# --- GET /articles (list) -----------------------------------------------------------------


async def test_public_article_list_is_empty_when_nothing_is_published(db_session: AsyncSession) -> None:
    episode, transcript = await _seed_episode_and_transcript(db_session)
    await _seed_article(db_session, episode, transcript, publish=False)

    client = await _client(db_session)
    try:
        response = await client.get("/api/v1/articles")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 200
    assert response.json()["articles"] == []


async def test_public_article_list_includes_only_published_articles(db_session: AsyncSession) -> None:
    published_episode, published_transcript = await _seed_episode_and_transcript(db_session)
    await _seed_article(db_session, published_episode, published_transcript, publish=True)

    unpublished_episode, unpublished_transcript = await _seed_episode_and_transcript(db_session)
    await _seed_article(db_session, unpublished_episode, unpublished_transcript, publish=False)

    client = await _client(db_session)
    try:
        response = await client.get("/api/v1/articles")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 200
    articles = response.json()["articles"]
    assert len(articles) == 1
    assert articles[0]["episode_id"] == str(published_episode.id)
    assert articles[0]["title"] == "The Article"
    assert articles[0]["published_at"] is not None


async def test_public_article_list_orders_newest_publish_first(db_session: AsyncSession) -> None:
    older_episode, older_transcript = await _seed_episode_and_transcript(db_session)
    await _seed_article(db_session, older_episode, older_transcript, publish=True)

    newer_episode, newer_transcript = await _seed_episode_and_transcript(db_session)
    await _seed_article(db_session, newer_episode, newer_transcript, publish=True)

    # Force a deterministic ordering regardless of how fast the two
    # publish calls above ran (datetime.now(UTC) resolution).
    older = await EpisodeRepository(db_session).get_by_id(older_episode.id)
    newer = await EpisodeRepository(db_session).get_by_id(newer_episode.id)
    assert newer.article_published_at >= older.article_published_at

    client = await _client(db_session)
    try:
        response = await client.get("/api/v1/articles")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 200
    episode_ids = [a["episode_id"] for a in response.json()["articles"]]
    assert episode_ids == [str(newer_episode.id), str(older_episode.id)]
