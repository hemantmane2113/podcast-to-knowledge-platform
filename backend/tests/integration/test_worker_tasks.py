import asyncio
import uuid

import pytest
from arq import Retry
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.core.exceptions import (
    LLMProviderTransientError,
    TranscriptProviderAuthError,
    TranscriptProviderError,
)
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
from app.services.transcript_processing_service import TranscriptProcessingService
from app.worker.tasks import (
    MAX_TRIES,
    RETRY_DEFER_SECONDS,
    generate_article,
    ingest_episode_transcript,
    process_transcript,
)
from tests.conftest import TEST_DATABASE_URL
from tests.fakes import FakeJobQueue, FakeLLMProvider, StubProvider


class _RetryableTestError(Exception):
    """A minimal stand-in for a transient failure in code paths (like
    TranscriptProcessingService, purely local computation) that don't
    naturally raise one of app.core.exceptions' own retryable types."""

    retryable = True

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


async def test_retryable_failure_raises_arq_retry_on_attempt_1(
    db_session: AsyncSession,
) -> None:
    """Regression test: this task must raise `arq.Retry`, not a bare
    re-raise of the original exception -- verified directly against the
    installed arq version (0.28.0) that only `arq.Retry` (or
    CancelledError/RetryJob under retry_jobs=True) causes arq to actually
    re-invoke the same job_id. A bare re-raise of an ordinary exception is
    treated by arq as an immediate terminal failure with no retry ever
    happening -- which is exactly how the real incident (job
    5f27dcad-b02a-4872-8a3d-639215ac77ce) ended up ARQ-failed but
    Postgres-RUNNING forever. The old version of this test asserted only
    the DB-state side of that bug and called it correct."""
    episode, job = await _seed_episode_and_job(db_session)
    ctx = {
        "job_try": 1,  # first of MAX_TRIES attempts
        "provider": StubProvider(transcript=TranscriptProviderError("transient failure")),
    }

    with pytest.raises(Retry) as exc_info:
        await ingest_episode_transcript(ctx, str(episode.id), str(job.id))
    assert exc_info.value.defer_score == RETRY_DEFER_SECONDS * 1000

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    # Not yet terminal, and untouched by this attempt -- arq will retry
    # this exact job_id; our own code must not mark it FAILED here.
    assert refreshed_episode.status == ProcessingStatus.INGESTING
    assert refreshed_job.status == JobStatus.RUNNING
    assert refreshed_job.error_message is None


async def test_retryable_failure_raises_arq_retry_on_attempt_2(
    db_session: AsyncSession,
) -> None:
    """Same as attempt 1, but at job_try=2 (the middle of a MAX_TRIES=3
    sequence) -- confirms `ctx["job_try"]` is read correctly at any
    position, not just the first attempt."""
    episode, job = await _seed_episode_and_job(db_session)
    ctx = {
        "job_try": 2,
        "provider": StubProvider(transcript=TranscriptProviderError("still transient")),
    }

    with pytest.raises(Retry):
        await ingest_episode_transcript(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.INGESTING
    assert refreshed_job.status == JobStatus.RUNNING


async def test_ingest_episode_transcript_cancellation_marks_failed(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirrors generate_article's existing cancellation regression test --
    ingestion had no CancelledError handler at all before this fix, so a
    job_timeout during ingestion left the row stuck exactly like the
    original generate_article incident."""
    episode, job = await _seed_episode_and_job(db_session)

    async def _cancelled(*_args, **_kwargs):
        raise asyncio.CancelledError()

    from app.services.ingestion_service import IngestionService

    monkeypatch.setattr(IngestionService, "run", _cancelled)
    ctx = {"job_try": 1, "provider": StubProvider(transcript=Exception("unused"))}

    with pytest.raises(asyncio.CancelledError):
        await ingest_episode_transcript(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.FAILED
    assert refreshed_job.status == JobStatus.FAILED


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


async def test_process_transcript_retryable_failure_raises_arq_retry_on_attempt_1(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode, transcript, job = await _seed_episode_with_transcript(db_session)

    async def _raise(*_args, **_kwargs):
        raise _RetryableTestError("transient DB blip")

    monkeypatch.setattr(TranscriptProcessingService, "run", _raise)

    with pytest.raises(Retry) as exc_info:
        await process_transcript({"job_try": 1}, str(episode.id), str(job.id))
    assert exc_info.value.defer_score == RETRY_DEFER_SECONDS * 1000

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.TRANSCRIPT_FETCHED
    assert refreshed_job.status == JobStatus.RUNNING
    assert refreshed_job.error_message is None


async def test_process_transcript_retryable_failure_raises_arq_retry_on_attempt_2(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode, transcript, job = await _seed_episode_with_transcript(db_session)

    async def _raise(*_args, **_kwargs):
        raise _RetryableTestError("still failing")

    monkeypatch.setattr(TranscriptProcessingService, "run", _raise)

    with pytest.raises(Retry):
        await process_transcript({"job_try": 2}, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.TRANSCRIPT_FETCHED
    assert refreshed_job.status == JobStatus.RUNNING


async def test_process_transcript_retryable_failure_marks_failed_on_final_attempt(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode, transcript, job = await _seed_episode_with_transcript(db_session)

    async def _raise(*_args, **_kwargs):
        raise _RetryableTestError("still failing")

    monkeypatch.setattr(TranscriptProcessingService, "run", _raise)

    # Must NOT raise -- attempts are exhausted, this is the terminal failure.
    await process_transcript({"job_try": MAX_TRIES}, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.FAILED
    assert refreshed_job.status == JobStatus.FAILED
    assert refreshed_job.error_message == "Transcript processing failed. See server logs for details."


async def test_process_transcript_cancellation_marks_failed(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirrors generate_article's existing cancellation regression test --
    process_transcript had no CancelledError handler at all before this
    fix."""
    episode, transcript, job = await _seed_episode_with_transcript(db_session)

    async def _cancelled(*_args, **_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(TranscriptProcessingService, "run", _cancelled)

    with pytest.raises(asyncio.CancelledError):
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

    # Three sections, not one: Settings.section_count_min defaults to 3
    # (app/services/article_validation.py::check_section_count), and a
    # real worker process reads real settings (get_settings(), not
    # overridable per-test here) -- a single-section plan would trigger
    # an extra, unprogrammed revision round in these tests. Distinct
    # content per section (not just heading) so check_no_duplicate_sections
    # doesn't ALSO fail and force yet another revision round.
    return FakeLLMProvider(
        structured_responses=[
            TopicAnalysisResult(topics=[TopicItem(title="Topic A", summary="s", chunk_sequence_numbers=[0])]),
            ArticlePlanResult(
                title="The Article",
                introduction_summary="i",
                sections=[
                    PlannedSection(heading="Intro", supporting_topic_sequence_numbers=[0]),
                    PlannedSection(heading="Body", supporting_topic_sequence_numbers=[0]),
                    PlannedSection(heading="Conclusion", supporting_topic_sequence_numbers=[0]),
                ],
                conclusion_summary="c",
            ),
            GeneratedSection(heading="Intro", paragraphs=[" ".join(["intro"] + ["word"] * 149)]),
            GeneratedSection(heading="Body", paragraphs=[" ".join(["body"] + ["word"] * 149)]),
            GeneratedSection(heading="Conclusion", paragraphs=[" ".join(["conclusion"] + ["word"] * 149)]),
        ]
    )


async def test_generate_article_success_persists_article_and_marks_ready_for_review(
    db_session: AsyncSession,
) -> None:
    episode, transcript, job = await _seed_episode_with_chunks(db_session)
    ctx = {"job_try": 1, "llm_provider": _fake_llm_provider()}

    await generate_article(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    # Episode.status Decoupling: article-generation progress lives on
    # ProcessingJob.status now -- Episode.status must stay exactly as
    # _seed_episode_with_chunks left it (TRANSCRIPT_FETCHED), never
    # advance to READY_FOR_REVIEW.
    assert refreshed_episode.status == ProcessingStatus.TRANSCRIPT_FETCHED
    assert refreshed_job.status == JobStatus.COMPLETED

    article = await ArticleRepository(db_session).get_by_episode_id(episode.id, is_draft=True)
    assert article is not None
    assert article.title == "The Article"
    assert len(article.sections) == 3


async def test_generate_article_missing_llm_provider_fails_without_raising(
    db_session: AsyncSession,
) -> None:
    episode, transcript, job = await _seed_episode_with_chunks(db_session)
    ctx = {"job_try": 1, "llm_provider": None}

    await generate_article(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    # Episode.status Decoupling: this failure happens before any pipeline
    # state exists, but it's still a generate_article failure path, so it
    # must never overwrite Episode.status either -- the error is recorded
    # on ProcessingJob only.
    assert refreshed_episode.status == ProcessingStatus.TRANSCRIPT_FETCHED
    assert "not configured" in refreshed_job.error_message
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
    # Episode.status Decoupling: episode was seeded via a bare
    # episodes.create() (INGESTING), and article generation never writes
    # Episode.status, so it must remain INGESTING here.
    assert refreshed_episode.status == ProcessingStatus.INGESTING
    assert refreshed_job.status == JobStatus.FAILED


async def test_generate_article_missing_chunks_fails_without_raising(db_session: AsyncSession) -> None:
    episode, transcript, _ = await _seed_episode_with_transcript(db_session)
    jobs = ProcessingJobRepository(db_session)
    job = jobs.create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    await db_session.commit()

    await generate_article({"job_try": 1, "llm_provider": _fake_llm_provider()}, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    # Episode.status Decoupling: seeded via _seed_episode_with_transcript
    # (TRANSCRIPT_FETCHED), and article generation never writes
    # Episode.status.
    assert refreshed_episode.status == ProcessingStatus.TRANSCRIPT_FETCHED
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
    assert refreshed_episode.status == ProcessingStatus.TRANSCRIPT_FETCHED
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
    # Episode.status Decoupling: this failure happens before any pipeline
    # state exists, but it's still a generate_article failure path, so it
    # must never overwrite Episode.status either.
    assert refreshed_episode.status == ProcessingStatus.TRANSCRIPT_FETCHED
    assert "not configured" in refreshed_job.error_message
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
    # Episode.status Decoupling: seeded via _seed_episode_with_chunks
    # (TRANSCRIPT_FETCHED), and a cancelled draft generation must never
    # overwrite Episode.status (update_episode_status=False in
    # generate_article's CancelledError handler).
    assert refreshed_episode.status == ProcessingStatus.TRANSCRIPT_FETCHED
    assert refreshed_job.status == JobStatus.FAILED


async def test_generate_article_retryable_failure_raises_arq_retry_on_attempt_1(
    db_session: AsyncSession,
) -> None:
    """This is the exact shape of the real incident (job
    5f27dcad-b02a-4872-8a3d-639215ac77ce): Groq 429 -> OpenAI fallback ->
    OpenAI 60s timeout, mapped by app/providers/llm/_chat_completions.py's
    map_sdk_exception to LLMProviderTransientError(retryable=True), raised
    out of the topic-analysis node's first generate_structured call."""
    episode, transcript, job = await _seed_episode_with_chunks(db_session)
    failing_llm = FakeLLMProvider(structured_responses=[LLMProviderTransientError("Request timed out.")])
    ctx = {"job_try": 1, "llm_provider": failing_llm}

    with pytest.raises(Retry) as exc_info:
        await generate_article(ctx, str(episode.id), str(job.id))
    assert exc_info.value.defer_score == RETRY_DEFER_SECONDS * 1000

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    # Episode.status Decoupling: topic_analysis_node no longer writes
    # Episode.status at all -- article-generation progress lives entirely
    # on ProcessingJob.status now, so this stays exactly as seeded
    # (TRANSCRIPT_FETCHED) regardless of where in the pipeline the retry
    # occurred.
    assert refreshed_episode.status == ProcessingStatus.TRANSCRIPT_FETCHED
    assert refreshed_job.status == JobStatus.RUNNING
    assert refreshed_job.error_message is None
    article = await ArticleRepository(db_session).get_by_episode_id(episode.id, is_draft=True)
    assert article is None


async def test_generate_article_retryable_failure_raises_arq_retry_on_attempt_2(
    db_session: AsyncSession,
) -> None:
    episode, transcript, job = await _seed_episode_with_chunks(db_session)
    failing_llm = FakeLLMProvider(structured_responses=[LLMProviderTransientError("still failing")])
    ctx = {"job_try": 2, "llm_provider": failing_llm}

    with pytest.raises(Retry):
        await generate_article(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.TRANSCRIPT_FETCHED
    assert refreshed_job.status == JobStatus.RUNNING


async def test_generate_article_retryable_failure_marks_failed_on_final_attempt(
    db_session: AsyncSession,
) -> None:
    episode, transcript, job = await _seed_episode_with_chunks(db_session)
    failing_llm = FakeLLMProvider(structured_responses=[LLMProviderTransientError("still failing")])
    ctx = {"job_try": MAX_TRIES, "llm_provider": failing_llm}

    # Must NOT raise -- attempts are exhausted, this is the terminal failure.
    await generate_article(ctx, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.TRANSCRIPT_FETCHED
    assert refreshed_job.status == JobStatus.FAILED
    assert refreshed_job.error_message == "still failing"


async def test_generate_article_succeeds_on_a_genuine_retry_after_attempt_1_fails(
    db_session: AsyncSession,
) -> None:
    """Simulates the exact sequence arq itself performs once Retry is
    raised: the SAME job_id is re-invoked later with job_try incremented.
    Attempt 1 fails transiently (no article, no persisted work at all --
    the failure is in topic analysis, the very first LLM call); attempt 2,
    against a fully-succeeding provider, completes normally."""
    episode, transcript, job = await _seed_episode_with_chunks(db_session)
    failing_llm = FakeLLMProvider(structured_responses=[LLMProviderTransientError("transient")])

    with pytest.raises(Retry):
        await generate_article({"job_try": 1, "llm_provider": failing_llm}, str(episode.id), str(job.id))

    succeeding_llm = _fake_llm_provider()
    await generate_article({"job_try": 2, "llm_provider": succeeding_llm}, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.TRANSCRIPT_FETCHED
    assert refreshed_job.status == JobStatus.COMPLETED
    article = await ArticleRepository(db_session).get_by_episode_id(episode.id, is_draft=True)
    assert article is not None
    assert len(article.sections) == 3


async def test_generate_article_retry_resumes_from_already_persisted_sections(
    db_session: AsyncSession,
) -> None:
    """Resumability (Batch 2B) interacting with a genuine arq retry
    (rather than a separately-submitted job, which is all the existing
    resumability integration tests exercise): attempt 1 completes topic
    analysis, planning, and the first section, then fails transiently
    while generating the second section. Attempt 2 (job_try=2, same
    job_id -- the real arq retry shape) must reuse everything already
    persisted rather than regenerating it, and must complete successfully."""
    episode, transcript, job = await _seed_episode_with_chunks(db_session)
    from app.ai.schemas import ArticlePlanResult, GeneratedSection, PlannedSection, TopicAnalysisResult, TopicItem

    attempt_1_llm = FakeLLMProvider(
        structured_responses=[
            TopicAnalysisResult(topics=[TopicItem(title="Topic A", summary="s", chunk_sequence_numbers=[0])]),
            ArticlePlanResult(
                title="The Article",
                introduction_summary="i",
                sections=[
                    PlannedSection(heading="Intro", supporting_topic_sequence_numbers=[0]),
                    PlannedSection(heading="Body", supporting_topic_sequence_numbers=[0]),
                    PlannedSection(heading="Conclusion", supporting_topic_sequence_numbers=[0]),
                ],
                conclusion_summary="c",
            ),
            GeneratedSection(heading="Intro", paragraphs=[" ".join(["intro"] + ["word"] * 149)]),
            LLMProviderTransientError("transient failure generating Body"),
        ]
    )

    with pytest.raises(Retry):
        await generate_article({"job_try": 1, "llm_provider": attempt_1_llm}, str(episode.id), str(job.id))

    # Confirms the premise: attempt 1 really did persist topic analysis,
    # planning, and exactly one section before failing.
    article_after_attempt_1 = await ArticleRepository(db_session).get_by_episode_id(episode.id, is_draft=True)
    assert article_after_attempt_1 is not None
    assert len(article_after_attempt_1.sections) == 1
    assert article_after_attempt_1.sections[0].heading == "Intro"

    attempt_2_llm = FakeLLMProvider(
        structured_responses=[
            GeneratedSection(heading="Body", paragraphs=[" ".join(["body"] + ["word"] * 149)]),
            GeneratedSection(heading="Conclusion", paragraphs=[" ".join(["conclusion"] + ["word"] * 149)]),
        ]
    )
    await generate_article({"job_try": 2, "llm_provider": attempt_2_llm}, str(episode.id), str(job.id))

    refreshed_episode, refreshed_job = await _reload(episode.id, job.id)
    assert refreshed_episode.status == ProcessingStatus.TRANSCRIPT_FETCHED
    assert refreshed_job.status == JobStatus.COMPLETED

    # A fresh session/engine, not db_session -- db_session's identity map
    # already holds article_after_attempt_1's Article (id and all), loaded
    # BEFORE attempt 2 committed its writes through generate_article's own,
    # separate session; with expire_on_commit=False, re-querying through
    # db_session would just hand back that same, now-stale Python object
    # instead of observing what attempt 2 actually persisted (the same
    # reasoning _reload's own docstring documents for episode/job).
    engine = create_async_engine(TEST_DATABASE_URL)
    try:
        async with AsyncSession(bind=engine) as fresh_session:
            final_article = await ArticleRepository(fresh_session).get_by_episode_id(episode.id, is_draft=True)
    finally:
        await engine.dispose()

    assert final_article.id == article_after_attempt_1.id
    assert len(final_article.sections) == 3
    # Attempt 2 only had to pay for the two missing sections -- topic
    # analysis, planning, and the already-persisted "Intro" section cost
    # it nothing.
    assert len(attempt_2_llm.structured_calls) == 2


async def test_generate_article_two_historical_jobs_for_the_same_episode_do_not_interfere(
    db_session: AsyncSession,
) -> None:
    """A second, separately-created ProcessingJob for the same episode
    (e.g. a user retriggering generation after reviewing the first result)
    must operate correctly regardless of the first job's historical
    row -- ProcessingJobRepository has no uniqueness constraint tying an
    episode to a single job row, by design (see
    ProcessingJobRepository.get_active_job's docstring)."""
    episode, transcript, job = await _seed_episode_with_chunks(db_session)
    await generate_article({"job_try": 1, "llm_provider": _fake_llm_provider()}, str(episode.id), str(job.id))

    second_job = ProcessingJobRepository(db_session).create(
        episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION
    )
    await db_session.commit()
    await generate_article(
        {"job_try": 1, "llm_provider": _fake_llm_provider()}, str(episode.id), str(second_job.id)
    )

    refreshed_episode, refreshed_first_job = await _reload(episode.id, job.id)
    _, refreshed_second_job = await _reload(episode.id, second_job.id)
    assert refreshed_episode.status == ProcessingStatus.TRANSCRIPT_FETCHED
    assert refreshed_first_job.status == JobStatus.COMPLETED
    assert refreshed_second_job.status == JobStatus.COMPLETED


async def test_generate_article_is_idempotent_on_rerun(db_session: AsyncSession) -> None:
    episode, transcript, job = await _seed_episode_with_chunks(db_session)
    await generate_article({"job_try": 1, "llm_provider": _fake_llm_provider()}, str(episode.id), str(job.id))
    first_article = await ArticleRepository(db_session).get_by_episode_id(episode.id, is_draft=True)

    second_job = ProcessingJobRepository(db_session).create(episode_id=episode.id, job_type=JobType.ARTICLE_GENERATION)
    await db_session.commit()
    second_llm = _fake_llm_provider()
    await generate_article(
        {"job_try": 1, "llm_provider": second_llm}, str(episode.id), str(second_job.id)
    )
    second_article = await ArticleRepository(db_session).get_by_episode_id(episode.id, is_draft=True)

    # Batch 2B (resumability): the first run already produced a complete,
    # valid article, so the second run reuses it entirely -- same row, zero
    # LLM calls -- rather than blindly regenerating from scratch. Stronger
    # than the pre-Batch-2B "delete-then-reinsert" behavior this replaces:
    # that was idempotent in name only (same *content*, wastefully
    # recomputed); this is idempotent in the sense that actually matters
    # for a paid pipeline -- no repeated work at all.
    assert second_article.id == first_article.id
    assert second_article.title == first_article.title
    assert len(second_llm.structured_calls) == 0
