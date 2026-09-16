from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_episode_service
from app.main import app
from app.services.episode_service import EpisodeService
from tests.fakes import FakeJobQueue

VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
VIDEO_ID = "dQw4w9WgXcQ"


async def _client(db_session: AsyncSession) -> AsyncClient:
    queue = FakeJobQueue()

    async def override_get_episode_service():
        yield EpisodeService(db_session, queue)

    app.dependency_overrides[get_episode_service] = override_get_episode_service
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_create_episode_returns_201_with_ingesting_status(db_session: AsyncSession) -> None:
    client = await _client(db_session)
    try:
        response = await client.post("/api/v1/episodes", json={"youtube_url": VIDEO_URL})
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 201
    body = response.json()
    assert body["youtube_video_id"] == VIDEO_ID
    assert body["status"] == "INGESTING"
    assert "id" in body


async def test_create_episode_while_still_processing_returns_409(db_session: AsyncSession) -> None:
    client = await _client(db_session)
    try:
        first = await client.post("/api/v1/episodes", json={"youtube_url": VIDEO_URL})
        second = await client.post("/api/v1/episodes", json={"youtube_url": VIDEO_URL})
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert first.status_code == 201
    assert second.status_code == 409
    assert second.json()["code"] == "EPISODE_ALREADY_PROCESSING"


async def test_create_episode_after_completion_replays_existing_episode(
    db_session: AsyncSession,
) -> None:
    from app.models.episode import ProcessingStatus
    from app.models.processing_job import JobType
    from app.repositories.episode_repository import EpisodeRepository
    from app.repositories.processing_job_repository import ProcessingJobRepository

    client = await _client(db_session)
    try:
        first = await client.post("/api/v1/episodes", json={"youtube_url": VIDEO_URL})
        episode_id = first.json()["id"]

        # Simulate the worker having finished this episode's ingestion.
        import uuid as uuid_mod

        episodes = EpisodeRepository(db_session)
        jobs = ProcessingJobRepository(db_session)
        episode = await episodes.get_by_id(uuid_mod.UUID(episode_id))
        episodes.set_status(episode, ProcessingStatus.TRANSCRIPT_FETCHED)
        active_job = await jobs.get_active_job(episode.id, JobType.TRANSCRIPT_INGESTION)
        jobs.mark_completed(active_job)
        await db_session.commit()

        second = await client.post("/api/v1/episodes", json={"youtube_url": VIDEO_URL})
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["id"] == episode_id
    assert second.json()["status"] == "TRANSCRIPT_FETCHED"


async def test_create_episode_invalid_url_returns_400(db_session: AsyncSession) -> None:
    client = await _client(db_session)
    try:
        response = await client.post("/api/v1/episodes", json={"youtube_url": "not a url"})
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_YOUTUBE_URL"


async def test_get_episode_not_found_returns_404(db_session: AsyncSession) -> None:
    client = await _client(db_session)
    try:
        response = await client.get("/api/v1/episodes/00000000-0000-0000-0000-000000000000")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 404
    assert response.json()["code"] == "EPISODE_NOT_FOUND"


async def test_get_episode_returns_current_status(db_session: AsyncSession) -> None:
    client = await _client(db_session)
    try:
        created = await client.post("/api/v1/episodes", json={"youtube_url": VIDEO_URL})
        episode_id = created.json()["id"]

        response = await client.get(f"/api/v1/episodes/{episode_id}")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 200
    assert response.json()["id"] == episode_id
    assert response.json()["status"] == "INGESTING"


async def test_get_transcript_not_ready_returns_404(db_session: AsyncSession) -> None:
    client = await _client(db_session)
    try:
        created = await client.post("/api/v1/episodes", json={"youtube_url": VIDEO_URL})
        episode_id = created.json()["id"]

        response = await client.get(f"/api/v1/episodes/{episode_id}/transcript")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 404
    assert response.json()["code"] == "TRANSCRIPT_NOT_FOUND"


async def test_get_transcript_returns_segments_once_ingested(db_session: AsyncSession) -> None:
    from app.models.episode import ProcessingStatus
    from app.repositories.episode_repository import EpisodeRepository
    from app.repositories.transcript_repository import TranscriptRepository
    from app.schemas.transcript import NormalizedSegment, NormalizedTranscript

    client = await _client(db_session)
    try:
        created = await client.post("/api/v1/episodes", json={"youtube_url": VIDEO_URL})
        episode_id = created.json()["id"]

        episodes = EpisodeRepository(db_session)
        transcripts = TranscriptRepository(db_session)
        import uuid as uuid_mod

        episode = await episodes.get_by_id(uuid_mod.UUID(episode_id))
        transcripts.create_with_segments(
            episode.id,
            NormalizedTranscript(
                language="en",
                segments=[
                    NormalizedSegment(text="first", start_ms=0, duration_ms=1000),
                    NormalizedSegment(text="second", start_ms=1000, duration_ms=800, speaker="Speaker 1"),
                ],
            ),
        )
        episodes.set_status(episode, ProcessingStatus.TRANSCRIPT_FETCHED)
        await db_session.commit()

        response = await client.get(f"/api/v1/episodes/{episode_id}/transcript")
    finally:
        app.dependency_overrides.clear()
        await client.aclose()

    assert response.status_code == 200
    body = response.json()
    assert body["episode_id"] == episode_id
    assert body["language"] == "en"
    assert len(body["segments"]) == 2
    assert body["segments"][0] == {
        "sequence_number": 0,
        "text": "first",
        "start_ms": 0,
        "duration_ms": 1000,
        "speaker": None,
    }
    assert body["segments"][1]["speaker"] == "Speaker 1"
