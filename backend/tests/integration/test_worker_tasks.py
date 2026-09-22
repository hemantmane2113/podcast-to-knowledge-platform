import asyncio
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.exceptions import TranscriptProviderAuthError, TranscriptProviderError
from app.models.chunk import Chunk
from app.models.episode import Episode, ProcessingStatus
from app.models.processing_job import JobStatus, JobType, ProcessingJob
from app.repositories.article_repository import ArticleRepository
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.episode_repository import EpisodeRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.repositories.transcript_repository import TranscriptRepository
from app.schemas.transcript import NormalizedSegment, NormalizedTranscript
from app.services.chunking_service import ChunkCandidate
from app.worker.tasks import MAX_TRIES, generate_article, ingest_episode_transcript, process_transcript
from tests.conftest import TEST_DATABASE_URL
from tests.fakes import FakeJobQueue, FakeLLMProvider, StubProvider

VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
VIDEO_ID = "dQw4w9WgXcQ"


async def _seed_episode_and_job(session: AsyncSession):
    episodes = EpisodeRepository(session)
    jobs = ProcessingJobRepository(session)
    episode = episodes.create(youtube_video_id=VIDEO_ID, youtube_url=VIDEO_URL)
    job = jobs.create(episode_id=episode.id, job_type=JobType.TRANSCRIPT_INGESTION)
    await session.commit()
    return episode, job


async def _reload(episode_id, job_id) -> tuple[Episode, ProcessingJob]:
    """The worker task writes through its own session (a separate
    connection to the same database -- see app.worker.tasks). Read back
    through a brand-new engine/session rather than the seeding session's
    identity map, so this actually observes what the worker committed
    instead of the pre-worker in-memory objects."""
    engine = create_async_engine(TEST_DATABASE_URL)
    try:
        async with AsyncSession(bind=engine) as fresh_session:
            episode = await EpisodeRepository(fresh_session).get_by_id(episode_id)
            job = await ProcessingJobRepository(fresh_session).get_by_id(job_id)
            return episode, job
    finally:
        await engine.dispose()


async def test_successful_ingestion(db_session: AsyncSession) -> None:
    episode, job = await _seed_episode_and_job(db_session)
    transcript = NormalizedTranscript(
        language="en", segments=[NormalizedSegment(text="hi", start_ms=0, duration_ms=500)]
    )
    queue = FakeJobQueue()
    ctx = {"job_try": 1, "provider": StubProvider(transcript=transcript), "job_queue": queue}

    await ingest_episode_transcript(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.TRANSCRIPT_FETCHED
    assert refreshed_job.status == JobStatus.COMPLETED

    # Auto-chains into transcript processing (§3A.10) -- a new job was
    # created and enqueued, not left for something else to trigger.
    assert len(queue.enqueued_processing) == 1
    assert queue.enqueued_processing[0][0] == episode.id


async def test_successful_ingestion_without_job_queue_does_not_crash(
    db_session: AsyncSession,
) -> None:
    # A worker ctx missing "job_queue" (shouldn't happen in practice --
    # app.worker.settings always sets it) must degrade to "ingestion
    # succeeded, chaining skipped", never raise.
    episode, job = await _seed_episode_and_job(db_session)
    transcript = NormalizedTranscript(
        language="en", segments=[NormalizedSegment(text="hi", start_ms=0, duration_ms=500)]
    )
    ctx = {"job_try": 1, "provider": StubProvider(transcript=transcript)}

    await ingest_episode_transcript(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.TRANSCRIPT_FETCHED
    assert refreshed_job.status == JobStatus.COMPLETED


async def test_non_retryable_failure_marks_failed_immediately_without_raising(
    db_session: AsyncSession,
) -> None:
    episode, job = await _seed_episode_and_job(db_session)
    ctx = {
        "job_try": 1,
        "provider": StubProvider(transcript=TranscriptProviderAuthError("bad key")),
    }

    # Must NOT raise -- an auth failure is never retried, even on the first attempt.
    await ingest_episode_transcript(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.FAILED
    assert refreshed_episode.last_error == "bad key"
    assert refreshed_job.status == JobStatus.FAILED
    assert refreshed_job.error_message == "bad key"


async def test_retryable_failure_raises_to_let_arq_retry_on_non_final_attempt(
    db_session: AsyncSession,
) -> None:
    episode, job = await _seed_episode_and_job(db_session)
    ctx = {
        "job_try": 1,  # first of MAX_TRIES attempts
        "provider": StubProvider(transcript=TranscriptProviderError("transient failure")),
    }

    with pytest.raises(TranscriptProviderError):
        await ingest_episode_transcript(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    # Not yet terminal -- arq is expected to retry this job.
    assert refreshed_episode.status == ProcessingStatus.INGESTING
    assert refreshed_job.status == JobStatus.RUNNING


async def test_missing_provider_marks_failed_without_raising(db_session: AsyncSession) -> None:
    episode, job = await _seed_episode_and_job(db_session)
    ctx = {"job_try": 1, "provider": None}  # e.g. SUPADATA_API_KEY not configured

    await ingest_episode_transcript(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.FAILED
    assert "not configured" in refreshed_episode.last_error
    assert refreshed_job.status == JobStatus.FAILED


async def test_retryable_failure_marks_failed_on_final_attempt(db_session: AsyncSession) -> None:
    episode, job = await _seed_episode_and_job(db_session)
    ctx = {
        "job_try": MAX_TRIES,  # last attempt
        "provider": StubProvider(transcript=TranscriptProviderError("still failing")),
    }

    # Must NOT raise -- attempts are exhausted, this is the terminal failure.
    await ingest_episode_transcript(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.FAILED
    assert refreshed_job.status == JobStatus.FAILED
    assert refreshed_job.error_message == "still failing"


# --- process_transcript (Phase 3A) ------------------------------------------------


async def _seed_episode_with_transcript(session: AsyncSession):
    from app.models.episode import Episode
    from app.models.transcript import Transcript
    from app.models.transcript_segment import TranscriptSegment

    episode = Episode(
        id=uuid.uuid4(),
        youtube_video_id=VIDEO_ID,
        youtube_url=VIDEO_URL,
        status=ProcessingStatus.TRANSCRIPT_FETCHED,
    )
    session.add(episode)
    transcript = Transcript(id=uuid.uuid4(), episode_id=episode.id, language="en")
    session.add(transcript)
    transcript.segments = [
        TranscriptSegment(
            id=uuid.uuid4(),
            transcript_id=transcript.id,
            sequence_number=i,
            text=f"Sentence number {i}.",
            start_ms=i * 4000,
            duration_ms=3000,
        )
        for i in range(5)
    ]
    jobs = ProcessingJobRepository(session)
    job = jobs.create(episode_id=episode.id, job_type=JobType.TRANSCRIPT_PROCESSING)
    await session.commit()
    return episode, transcript, job


async def _reload_chunks(transcript_id) -> list[Chunk]:
    engine = create_async_engine(TEST_DATABASE_URL)
    try:
        async with AsyncSession(bind=engine) as fresh_session:
            return await ChunkRepository(fresh_session).get_by_transcript_id(transcript_id)
    finally:
        await engine.dispose()


async def test_process_transcript_success_persists_chunks(db_session: AsyncSession) -> None:
    episode, transcript, job = await _seed_episode_with_transcript(db_session)
    ctx = {"job_try": 1}

    await process_transcript(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.CHUNKING
    assert refreshed_job.status == JobStatus.COMPLETED

    chunks = await _reload_chunks(transcript.id)
    assert len(chunks) >= 1
    all_segment_ids = {s.id for s in transcript.segments}
    for chunk in chunks:
        assert set(chunk.source_segment_ids) <= all_segment_ids


async def test_process_transcript_is_idempotent_on_rerun(db_session: AsyncSession) -> None:
    episode, transcript, job = await _seed_episode_with_transcript(db_session)
    ctx = {"job_try": 1}

    await process_transcript(ctx, str(episode.id), str(job.id))
    first_chunks = await _reload_chunks(transcript.id)

    # Re-run against the same transcript (e.g. a manual reprocess).
    jobs = ProcessingJobRepository(db_session)
    second_job = jobs.create(episode_id=episode.id, job_type=JobType.TRANSCRIPT_PROCESSING)
    await db_session.commit()
    await process_transcript({"job_try": 1}, str(episode.id), str(second_job.id))
    second_chunks = await _reload_chunks(transcript.id)

    assert len(second_chunks) == len(first_chunks)
    assert [c.text for c in second_chunks] == [c.text for c in first_chunks]
    assert [c.sequence_number for c in second_chunks] == [c.sequence_number for c in first_chunks]


async def test_process_transcript_missing_transcript_fails_without_raising(
    db_session: AsyncSession,
) -> None:
    episodes = EpisodeRepository(db_session)
    jobs = ProcessingJobRepository(db_session)
    episode = episodes.create(youtube_video_id=VIDEO_ID, youtube_url=VIDEO_URL)
    job = jobs.create(episode_id=episode.id, job_type=JobType.TRANSCRIPT_PROCESSING)
    await db_session.commit()

    # No transcript exists for this episode -- non-retryable (TranscriptNotFoundError).
    await process_transcript({"job_try": 1}, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.FAILED
    assert refreshed_job.status == JobStatus.FAILED


def _fail_if_called(*_args, **_kwargs):
    raise AssertionError("get_llm_provider must not be called for this job")


async def test_ingest_episode_transcript_never_calls_get_llm_provider(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for the lazy-LLM-provider fix: ingestion must not
    construct (or attempt to construct) an LLM provider, regardless of
    whether one is configured."""
    monkeypatch.setattr("app.providers.llm.factory.get_llm_provider", _fail_if_called)
    episode, job = await _seed_episode_and_job(db_session)
    transcript = NormalizedTranscript(
        language="en", segments=[NormalizedSegment(text="hi", start_ms=0, duration_ms=500)]
    )
    ctx = {"job_try": 1, "provider": StubProvider(transcript=transcript), "job_queue": FakeJobQueue()}

    await ingest_episode_transcript(ctx, str(episode.id), str(job.id))  # must not raise

    refreshed_episode, _ = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.TRANSCRIPT_FETCHED


async def test_process_transcript_never_calls_get_llm_provider(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for the lazy-LLM-provider fix: chunking must not
    construct (or attempt to construct) an LLM provider."""
    monkeypatch.setattr("app.providers.llm.factory.get_llm_provider", _fail_if_called)
    episode, transcript, job = await _seed_episode_with_transcript(db_session)

    await process_transcript({"job_try": 1}, str(episode.id), str(job.id))  # must not raise

    refreshed_episode, _ = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.CHUNKING


# --- generate_article (Phase C-H) --------------------------------------------------


async def _seed_episode_with_chunks(session: AsyncSession):
    episode, transcript, _ = await _seed_episode_with_transcript(session)
    chunk_candidates = [
        ChunkCandidate(
            sequence_number=0,
            # Long enough that a 150-word generated section comfortably
            # satisfies article_max_length_ratio (default 0.4) -- a
            # too-short transcript here would make the "good" generated
            # section itself fail check_article_length and trigger an
            # unplanned revision.
            text=" ".join(["word"] * 2000),
            start_ms=0,
            end_ms=10_000,
            source_segment_ids=[s.id for s in transcript.segments],
            token_count=2000,
        )
    ]
    await ChunkRepository(session).replace_all(
        transcript_id=transcript.id, episode_id=episode.id, candidates=chunk_candidates
    )
    jobs = ProcessingJobRepository(session)
    job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    await session.commit()
    return episode, transcript, job


def _fake_llm_provider() -> FakeLLMProvider:
    from app.ai.schemas import ArticlePlanResult, GeneratedSection, PlannedSection, TopicAnalysisResult, TopicItem

    return FakeLLMProvider(
        structured_responses=[
            TopicAnalysisResult(topics=[TopicItem(title="Topic A", summary="s", chunk_sequence_numbers=[0])]),
            ArticlePlanResult(
                title="The Article",
                introduction_summary="i",
                sections=[PlannedSection(heading="Intro", supporting_topic_sequence_numbers=[0])],
                conclusion_summary="c",
            ),
            GeneratedSection(heading="Intro", content=" ".join(["word"] * 150)),
        ]
    )


async def test_generate_article_success_persists_article_and_marks_ready_for_review(
    db_session: AsyncSession,
) -> None:
    episode, transcript, job = await _seed_episode_with_chunks(db_session)
    ctx = {"job_try": 1, "llm_provider": _fake_llm_provider()}

    await generate_article(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.READY_FOR_REVIEW
    assert refreshed_job.status == JobStatus.COMPLETED

    article = await ArticleRepository(db_session).get_by_episode_id(episode.id)
    assert article is not None
    assert article.title == "The Article"
    assert len(article.sections) == 1


async def test_generate_article_missing_llm_provider_fails_without_raising(
    db_session: AsyncSession,
) -> None:
    episode, transcript, job = await _seed_episode_with_chunks(db_session)
    ctx = {"job_try": 1, "llm_provider": None}

    await generate_article(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.FAILED
    assert "not configured" in refreshed_episode.last_error
    assert refreshed_job.status == JobStatus.FAILED


async def test_generate_article_missing_transcript_fails_without_raising(
    db_session: AsyncSession,
) -> None:
    episodes = EpisodeRepository(db_session)
    jobs = ProcessingJobRepository(db_session)
    episode = episodes.create(youtube_video_id=VIDEO_ID, youtube_url=VIDEO_URL)
    job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    await db_session.commit()

    await generate_article({"job_try": 1, "llm_provider": _fake_llm_provider()}, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.FAILED
    assert refreshed_job.status == JobStatus.FAILED


async def test_generate_article_missing_chunks_fails_without_raising(db_session: AsyncSession) -> None:
    episode, transcript, _ = await _seed_episode_with_transcript(db_session)
    jobs = ProcessingJobRepository(db_session)
    job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    await db_session.commit()

    await generate_article({"job_try": 1, "llm_provider": _fake_llm_provider()}, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.FAILED
    assert refreshed_job.status == JobStatus.FAILED


async def test_generate_article_lazily_constructs_the_provider_when_ctx_has_no_override(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for the lazy-LLM-provider fix: with no
    ctx["llm_provider"] override -- matching the real worker ctx, since
    app.worker.settings.startup() no longer sets one -- generate_article
    must call app.providers.llm.factory.get_llm_provider itself, exactly
    when this job runs."""
    episode, transcript, job = await _seed_episode_with_chunks(db_session)
    calls: list = []

    def _fake_get_llm_provider(settings):
        calls.append(settings)
        return _fake_llm_provider()

    monkeypatch.setattr("app.providers.llm.factory.get_llm_provider", _fake_get_llm_provider)

    await generate_article({"job_try": 1}, str(episode.id), str(job.id))  # no "llm_provider" key

    assert len(calls) == 1
    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.READY_FOR_REVIEW
    assert refreshed_job.status == JobStatus.COMPLETED


async def test_generate_article_fails_clearly_when_lazy_construction_raises(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for the original incident: a real, misconfigured
    provider (missing GROQ_API_KEY/LLM_MODEL) must still fail the job
    clearly rather than raise -- now via the lazy construction path
    instead of ctx injection."""
    from app.core.exceptions import LLMProviderAuthError

    episode, transcript, job = await _seed_episode_with_chunks(db_session)

    def _raise(settings):
        raise LLMProviderAuthError("LLM_MODEL is not configured")

    monkeypatch.setattr("app.providers.llm.factory.get_llm_provider", _raise)

    await generate_article({"job_try": 1}, str(episode.id), str(job.id))  # no "llm_provider" key

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.FAILED
    assert "not configured" in refreshed_episode.last_error
    assert refreshed_job.status == JobStatus.FAILED


async def test_generate_article_marks_failed_on_cancellation_instead_of_leaving_it_stuck(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for a real stuck job: arq's own job_timeout (default
    300s) cancels a job that runs too long by raising asyncio.CancelledError
    at the current await point inside run_article_pipeline --
    CancelledError is a BaseException, not an Exception, so it was never
    caught by generate_article's `except Exception`. That left
    ProcessingJob/Episode permanently stuck at RUNNING/ANALYZING even
    though the worker process itself was fine and had already moved on --
    with no way to recover, since a later POST /generate-article just kept
    returning that same permanently "active" job
    (ArticleService.request_generation). Simulates the exact cancellation
    directly rather than waiting a real 300s."""
    episode, transcript, job = await _seed_episode_with_chunks(db_session)

    async def _cancelled(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr("app.worker.tasks.run_article_pipeline", _cancelled)

    with pytest.raises(asyncio.CancelledError):
        await generate_article({"job_try": 1, "llm_provider": _fake_llm_provider()}, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.FAILED
    assert refreshed_job.status == JobStatus.FAILED


async def test_generate_article_is_idempotent_on_rerun(db_session: AsyncSession) -> None:
    episode, transcript, job = await _seed_episode_with_chunks(db_session)
    await generate_article({"job_try": 1, "llm_provider": _fake_llm_provider()}, str(episode.id), str(job.id))
    first_article = await ArticleRepository(db_session).get_by_episode_id(episode.id)

    second_job = ProcessingJobRepository(db_session).create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    await db_session.commit()
    await generate_article(
        {"job_try": 1, "llm_provider": _fake_llm_provider()}, str(episode.id), str(second_job.id)
    )
    second_article = await ArticleRepository(db_session).get_by_episode_id(episode.id)

    assert second_article.id != first_article.id  # delete-then-reinsert, same as Chunk/Topic
    assert second_article.title == first_article.title
