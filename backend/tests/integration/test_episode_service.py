import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import EpisodeAlreadyProcessingError
from app.models.episode import Episode, ProcessingStatus
from app.repositories.episode_repository import EpisodeRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.services.episode_service import EpisodeService
from app.utils.youtube import InvalidYouTubeURLError
from tests.fakes import FakeJobQueue

VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
VIDEO_ID = "dQw4w9WgXcQ"


async def test_create_episode_creates_and_enqueues(db_session: AsyncSession) -> None:
    queue = FakeJobQueue()
    service = EpisodeService(db_session, queue)

    episode, created = await service.create_episode(VIDEO_URL)

    assert created is True
    assert episode.youtube_video_id == VIDEO_ID
    assert episode.status == ProcessingStatus.INGESTING
    assert len(queue.enqueued) == 1
    assert queue.enqueued[0][0] == episode.id


async def test_create_episode_is_idempotent(db_session: AsyncSession) -> None:
    queue = FakeJobQueue()
    service = EpisodeService(db_session, queue)

    first, first_created = await service.create_episode(VIDEO_URL)

    # A second submission of the same video, after the first job finished,
    # must not create a duplicate episode row.
    episodes = EpisodeRepository(db_session)
    jobs = ProcessingJobRepository(db_session)
    episodes.set_status(first, ProcessingStatus.TRANSCRIPT_FETCHED)
    job = queue.enqueued[0]
    job_row = await jobs.get_by_id(job[1])
    assert job_row is not None
    jobs.mark_completed(job_row)
    await db_session.commit()

    second, second_created = await service.create_episode(VIDEO_URL)

    assert second_created is False
    assert second.id == first.id
    # No new job enqueued for the already-completed episode.
    assert len(queue.enqueued) == 1
    assert first_created is True


async def test_create_episode_rejects_while_already_processing(db_session: AsyncSession) -> None:
    queue = FakeJobQueue()
    service = EpisodeService(db_session, queue)

    await service.create_episode(VIDEO_URL)  # status stays INGESTING, job stays PENDING

    with pytest.raises(EpisodeAlreadyProcessingError):
        await service.create_episode(VIDEO_URL)

    # Still only one enqueue call from the first, successful call.
    assert len(queue.enqueued) == 1


async def test_create_episode_rejects_invalid_url(db_session: AsyncSession) -> None:
    queue = FakeJobQueue()
    service = EpisodeService(db_session, queue)

    with pytest.raises(InvalidYouTubeURLError):
        await service.create_episode("https://example.com/not-youtube")

    assert queue.enqueued == []


async def test_create_episode_recovers_from_race_via_db_constraint(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulates two concurrent requests for the same video: one wins and
    commits first: the other's existence check ran before that commit (so
    it also tries to insert), and must recover via the DB's unique
    constraint rather than creating a duplicate episode."""
    queue = FakeJobQueue()
    service = EpisodeService(db_session, queue)

    winner_id = uuid.uuid4()
    db_session.add(
        Episode(
            id=winner_id,
            youtube_video_id=VIDEO_ID,
            youtube_url=VIDEO_URL,
            status=ProcessingStatus.INGESTING,
        )
    )
    await db_session.commit()

    original_get = EpisodeRepository.get_by_youtube_video_id
    call_count = {"n": 0}

    async def fake_get_missing_first_time(self: EpisodeRepository, video_id: str):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return None  # simulate the race: ran before the winner's commit
        return await original_get(self, video_id)

    monkeypatch.setattr(EpisodeRepository, "get_by_youtube_video_id", fake_get_missing_first_time)

    episode, created = await service.create_episode(VIDEO_URL)

    assert created is False
    assert episode.id == winner_id
    assert queue.enqueued == []
