import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy import Row

from app.core.exceptions import (
    ArticleNotFoundError,
    ArticleValidationNotPassedError,
    DraftNotReadyError,
    EpisodeNotFoundError,
)
from app.models.article import Article
from app.models.article_plan import ArticlePlan
from app.models.article_section import ArticleSection
from app.models.chunk import Chunk
from app.models.episode import Episode, ProcessingStatus
from app.models.processing_job import JobStatus, JobType, ProcessingJob
from app.models.validation_result import ValidationResult
from app.repositories.article_plan_repository import ArticlePlanRepository
from app.repositories.article_repository import ArticleRepository
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.episode_repository import EpisodeRepository
from app.repositories.processing_job_repository import ProcessingJobRepository
from app.repositories.transcript_repository import TranscriptRepository
from app.repositories.validation_result_repository import ValidationResultRepository
from app.services.job_queue import JobQueue

# The model classes ArticlePlanRepository.delete_by_episode_id's cascade
# can remove -- see request_regeneration's targeted (not session-wide)
# identity-map expiry below.
_PLAN_CASCADE_MODELS = (ArticlePlan, Article, ArticleSection, ValidationResult)


class ArticleService:
    """API -> ArticleService -> repositories/job queue (Phase I: the
    review surface). Mirrors EpisodeService's shape -- reads assemble
    what a reviewer needs to see, `request_generation` only ever creates
    a job and enqueues it, never runs the AI pipeline inline (that's the
    worker's job, app/worker/tasks.py::generate_article).
    """

    def __init__(self, session: AsyncSession, job_queue: JobQueue):
        self._session = session
        self._job_queue = job_queue
        self._episodes = EpisodeRepository(session)
        self._articles = ArticleRepository(session)
        self._article_plans = ArticlePlanRepository(session)
        self._transcripts = TranscriptRepository(session)
        self._chunks = ChunkRepository(session)
        self._jobs = ProcessingJobRepository(session)
        self._validation_results = ValidationResultRepository(session)

    async def get_article_with_chunks(
        self, episode_id: uuid.UUID, *, is_draft: bool = False
    ) -> tuple[Episode, Article, list[Chunk]]:
        """`is_draft` defaults to False (the live article) -- GET
        /episodes/{id}/article's existing behavior is unchanged for every
        caller that doesn't explicitly ask for the draft. Passing
        is_draft=True is how a reviewer retrieves the in-progress/awaiting-
        review draft (GET /episodes/{id}/article?draft=true) without
        touching the public-facing live article at all (Live/Draft
        Article Workflow)."""
        episode = await self._episodes.get_by_id(episode_id)
        if episode is None:
            raise EpisodeNotFoundError(f"Episode {episode_id} not found")

        article = await self._articles.get_by_episode_id(episode_id, is_draft=is_draft)
        if article is None:
            message = (
                f"Episode {episode_id} has no draft article"
                if is_draft
                else f"Episode {episode_id} has no generated article yet"
            )
            raise ArticleNotFoundError(message)

        transcript = await self._transcripts.get_by_episode_id(episode_id)
        chunks = await self._chunks.get_by_transcript_id(transcript.id) if transcript else []
        return episode, article, chunks

    async def request_generation(self, episode_id: uuid.UUID) -> ProcessingJob:
        """Idempotent per episode, mirroring EpisodeService.create_episode's
        shape:
        - an active (PENDING/RUNNING) job is returned as-is -- never
          duplicated, so two callers racing this endpoint don't enqueue two
          concurrent pipelines for the same episode.
        - a previous attempt that FAILED (or no attempt at all yet) always
          gets a fresh job: a failed job must never permanently block
          retrying generation.
        - once an article has actually been generated (the most recent job
          COMPLETED), a bare call here must not silently re-run the
          (paid) pipeline -- the existing completed job is returned
          instead. This method's behavior never changes for that case;
          see request_regeneration() below for the explicit, deliberate
          "regenerate anyway" entry point instead.

        Live/Draft Article Workflow: every generation run (this method and
        request_regeneration() both) always targets the episode's DRAFT
        row -- app/worker/tasks.py::generate_article's initial_state
        always sets is_draft=True, unconditionally. The live row is only
        ever replaced by an explicit promote_draft() call. So "has this
        already been generated" below must check for EITHER a draft or a
        live article -- before any promotion happens, the draft is the
        only evidence a completed generation exists; after a promotion,
        the draft row is gone (it became the live row) but the article
        obviously still exists. Checking only one of the two would make
        this method silently re-run the (paid) pipeline in the other
        case.
        """
        episode = await self._episodes.get_by_id(episode_id)
        if episode is None:
            raise EpisodeNotFoundError(f"Episode {episode_id} not found")

        active_job = await self._jobs.get_active_job(episode_id, JobType.ARTICLE_GENERATION)
        if active_job is not None:
            return active_job

        has_draft = await self._articles.get_by_episode_id(episode_id, is_draft=True) is not None
        has_live = await self._articles.get_by_episode_id(episode_id, is_draft=False) is not None
        if has_draft or has_live:
            latest_job = await self._jobs.get_latest_job(episode_id, JobType.ARTICLE_GENERATION)
            if latest_job is not None:
                return latest_job

        job = self._jobs.create(episode_id=episode_id, job_type=JobType.ARTICLE_GENERATION)
        await self._session.commit()
        await self._job_queue.enqueue_article_generation(episode_id=episode_id, job_id=job.id)
        return job

    async def get_latest_generation_job(self, episode_id: uuid.UUID) -> ProcessingJob | None:
        """The latest ARTICLE_GENERATION job for this episode, if any --
        Episode.status Decoupling's reviewer-facing progress source (see
        ArticleGenerationStatusResponse). Does not require the episode to
        exist; a nonexistent episode simply has no jobs, same as one that
        exists but hasn't started generation yet."""
        return await self._jobs.get_latest_job(episode_id, JobType.ARTICLE_GENERATION)

    async def publish_article(self, episode_id: uuid.UUID) -> Episode:
        """The explicit, deliberate last step of generate -> validate ->
        human review -> explicit publish -- never automatic after
        generation (nothing else in this codebase calls this). Reuses
        the existing validation model as the sole source of truth for
        "has this article passed validation": ValidationResultRepository
        .get_latest_by_article_id's `passed` column (already deterministic
        -- see app/services/article_validation.py -- and already
        WARNING-tolerant, since a WARNING-severity check still counts as
        passed there). No new validation mechanism.

        Idempotent: an already-PUBLISHED episode is returned unchanged --
        article_published_at is stamped once, on the transition into
        PUBLISHED, and never bumped forward by a repeat call, since
        there's no separate "publication record" to create here
        (Episode.status/article_published_at ARE the record).

        Touches only the Episode row -- ArticlePlan, Article,
        ArticleSections, Topics, and Chunks are never modified.
        """
        episode = await self._episodes.get_by_id(episode_id)
        if episode is None:
            raise EpisodeNotFoundError(f"Episode {episode_id} not found")

        if episode.status == ProcessingStatus.PUBLISHED:
            return episode

        article = await self._articles.get_by_episode_id(episode_id)
        if article is None:
            raise ArticleNotFoundError(f"Episode {episode_id} has no generated article yet")

        latest_validation = await self._validation_results.get_latest_by_article_id(article.id)
        if latest_validation is None or not latest_validation.passed:
            raise ArticleValidationNotPassedError(
                f"Episode {episode_id}'s article has not passed validation yet"
            )

        self._episodes.mark_published(episode)
        await self._session.commit()
        # updated_at (app/models/base.py::TimestampMixin) has a
        # server-side onupdate=func.now() -- after an UPDATE, SQLAlchemy
        # marks it as needing a fresh read regardless of session-wide
        # expire_on_commit settings, since the actual computed value isn't
        # known until fetched. EpisodeResponse.model_validate() is a
        # synchronous Pydantic call, so a lazy-load triggered from inside
        # it can't run (MissingGreenlet) -- refresh explicitly first, the
        # same async-safe pattern used elsewhere in this service for the
        # analogous stale/expired-attribute risk.
        await self._session.refresh(episode)
        return episode

    async def promote_draft(self, episode_id: uuid.UUID) -> Episode:
        """Live/Draft Article Workflow: makes the episode's current DRAFT
        Article/ArticlePlan the new LIVE one, and publishes the episode.
        This is the only way a draft's content ever reaches the live row
        or the public-facing endpoints (app/api/v1/public_articles.py) --
        generation itself (request_generation/request_regeneration) never
        touches the live row, by design.

        Rejects (each with the same 409 ArticleValidationNotPassedError
        publish_article already uses, or the analogous 404/409 for a
        draft that isn't in a promotable state -- never a bespoke set of
        rules):
        - no draft article exists at all (404 ArticleNotFoundError)
        - the draft's most recent ARTICLE_GENERATION job hasn't completed
          yet (still PENDING/RUNNING) or FAILED outright (409
          DraftNotReadyError)
        - the draft has no ValidationResult yet, or its latest one didn't
          pass (409 ArticleValidationNotPassedError -- exactly
          publish_article's own validation-gate check, reused verbatim
          against the draft article instead of the live one)

        Locks the episode row (SELECT ... FOR UPDATE) for the whole
        operation, so two concurrent promote_draft calls for the same
        episode are serialized rather than interleaved -- the second
        call sees the first one's already-promoted (or already-rejected)
        state once it acquires the lock, never a half-applied one.

        The actual swap is demote-then-promote-then-delete, deliberately
        never delete-then-promote: the live row is never removed until
        AFTER the draft has safely taken over the "live" slot, so a
        failure at any point in this transaction leaves the ORIGINAL live
        Article/ArticlePlan/ArticleSections/ValidationResults completely
        untouched (the whole transaction rolls back) -- this is the exact
        opposite of the old request_regeneration behavior (delete the
        live plan up front, hope the new pipeline run succeeds) that this
        whole redesign exists to fix. Only the two partial unique indexes
        (`... WHERE NOT is_draft` on articles/article_plans) are ever at
        risk of a transient conflict, and only among is_draft=False rows
        -- so the live row must be flipped to is_draft=True (freeing that
        slot) and flushed BEFORE the draft row is flipped to
        is_draft=False, never the other order.
        """
        episode = await self._episodes.get_by_id_for_update(episode_id)
        if episode is None:
            raise EpisodeNotFoundError(f"Episode {episode_id} not found")

        draft = await self._articles.get_by_episode_id(episode_id, is_draft=True)
        if draft is None:
            raise ArticleNotFoundError(f"Episode {episode_id} has no draft article to promote")

        latest_job = await self._jobs.get_latest_job(episode_id, JobType.ARTICLE_GENERATION)
        if latest_job is None or latest_job.status in (JobStatus.PENDING, JobStatus.RUNNING):
            raise DraftNotReadyError(f"Episode {episode_id}'s draft generation is still in progress")
        if latest_job.status == JobStatus.FAILED:
            raise DraftNotReadyError(f"Episode {episode_id}'s draft generation failed")

        latest_validation = await self._validation_results.get_latest_by_article_id(draft.id)
        if latest_validation is None or not latest_validation.passed:
            raise ArticleValidationNotPassedError(
                f"Episode {episode_id}'s draft article has not passed validation"
            )

        draft_plan = await self._article_plans.get_by_episode_id(episode_id, is_draft=True)
        old_live = await self._articles.get_by_episode_id(episode_id, is_draft=False)
        old_live_plan = await self._article_plans.get_by_episode_id(episode_id, is_draft=False)

        if old_live is not None:
            old_live.is_draft = True
            if old_live_plan is not None:
                old_live_plan.is_draft = True
            # Immediately runs the UPDATE(s) so the live slot is freed
            # (partial unique index re-checked) before the draft below
            # tries to take it -- without this, both statements would be
            # batched into the same flush and could be sent in either
            # order, risking a spurious unique-violation on the draft's
            # own flip.
            await self._session.flush()

        draft.is_draft = False
        if draft_plan is not None:
            draft_plan.is_draft = False
        await self._session.flush()

        # Only now, with the draft safely occupying the "live" slot, is
        # the demoted old row removed -- ArticlePlan's ON DELETE CASCADE
        # (Article.article_plan_id -> ArticleSection/ValidationResult)
        # removes the old Article/ArticleSections/ValidationResults in
        # the same statement, so deleting the plan alone is sufficient.
        if old_live_plan is not None:
            await self._session.delete(old_live_plan)
        elif old_live is not None:
            await self._session.delete(old_live)

        if old_live is not None:
            # Same identity-map staleness risk request_regeneration's own
            # docstring/expiry loop guards against: the DB-level ON
            # DELETE CASCADE above removes old_live (and its sections/
            # validation results) without SQLAlchemy ever issuing a
            # DELETE for those specific rows, so an already-loaded object
            # like old_live is never automatically marked deleted or
            # expired by the ORM. Without this, a caller that still holds
            # `old_live` after this call would keep silently reading its
            # stale pre-delete attributes forever, with no DB round trip
            # and no error. Scoped to exactly the rows this delete can
            # affect, not session.expire_all() (see request_regeneration).
            for obj in list(self._session.identity_map.values()):
                if isinstance(obj, _PLAN_CASCADE_MODELS):
                    self._session.expire(obj)

        self._episodes.mark_published(episode)
        await self._session.commit()
        await self._session.refresh(episode)
        return episode

    async def get_published_article(self, episode_id: uuid.UUID) -> tuple[Episode, Article, list[Chunk]]:
        """The public counterpart to get_article_with_chunks: an
        identical lookup, but only ever returns an article whose episode
        is actually PUBLISHED. An article that exists but isn't
        published yet raises the SAME ArticleNotFoundError "doesn't
        exist" already uses -- a public caller must never be able to
        distinguish "no article" from "not published yet" (see
        ArticleNotFoundError's docstring)."""
        episode, article, chunks = await self.get_article_with_chunks(episode_id)
        if episode.status != ProcessingStatus.PUBLISHED:
            raise ArticleNotFoundError(f"Episode {episode_id} has no published article")
        return episode, article, chunks

    async def list_published_articles(self) -> list[Row]:
        """PUBLISHED-only blog listing (episode_id, title,
        article_published_at), newest first -- see
        ArticleRepository.list_published_summaries."""
        return await self._articles.list_published_summaries()

    async def request_regeneration(self, episode_id: uuid.UUID) -> ProcessingJob:
        """The explicit "regenerate" entry point request_generation's own
        docstring points to -- never called from request_generation, and
        does not change its behavior. Differs from it in exactly one way:
        an existing COMPLETED article/job is never a reason to return
        early, since re-running deliberately is the entire point.

        - An active (PENDING/RUNNING) job is still respected exactly as
          request_generation does -- this must never race a pipeline
          that's already in flight for this episode.
        - The episode's existing DRAFT ArticlePlan (if any) is deleted
          before the new job is created -- never the LIVE one (Live/Draft
          Article Workflow: a regeneration must never touch a currently-
          published episode's live Article/ArticlePlan/ArticleSections/
          ValidationResults, so a reviewer can keep reading/serving the
          live article for the entire duration of a regeneration run,
          including if it fails). This is the one thing that actually
          matters: app/ai/nodes/planning.py's resumability check
          (`_plan_is_complete`) only ever reuses a plan that still
          exists, so removing the stale draft plan is what forces the
          next pipeline run to call the planner again instead of
          silently reusing an old draft plan/article/sections left over
          from a previous, already-terminal draft run (see the lifecycle
          audit -- without this, a bare re-enqueue over a completed or
          failed draft is a no-op). The DB's own ON DELETE CASCADE
          (ArticlePlan -> Article -> ArticleSection/ValidationResult)
          removes everything generated from the old draft plan in the
          same operation; Topic and Chunk are untouched -- neither is
          reached by that cascade, and the new run is free to reuse the
          existing Topics exactly as topic_analysis.py's own, separate
          resumability check already does today.
        - Same commit/enqueue shape as request_generation: one commit
          covering both the delete and the new job, then enqueue.
        """
        episode = await self._episodes.get_by_id(episode_id)
        if episode is None:
            raise EpisodeNotFoundError(f"Episode {episode_id} not found")

        active_job = await self._jobs.get_active_job(episode_id, JobType.ARTICLE_GENERATION)
        if active_job is not None:
            return active_job

        await self._article_plans.delete_by_episode_id(episode_id, is_draft=True)
        # The delete above is a Core-level statement (see
        # ArticlePlanRepository.delete_by_episode_id) -- it does not
        # update any Article/ArticlePlan/ArticleSection/ValidationResult
        # objects that happen to already be loaded in THIS session's
        # identity map (e.g. from an earlier call in the same request).
        # Expire exactly those, not the whole session (expire_all() would
        # also expire completely unrelated objects already loaded in this
        # session -- Episode, ProcessingJob -- forcing an unnecessary,
        # and in a sync attribute-access context actually unsafe, reload
        # of state that was never touched by this delete). This is the
        # same class of stale-identity-map bug this codebase has hit
        # before with a Core DELETE; the fix here is scoped to only the
        # rows this delete can actually affect.
        for obj in list(self._session.identity_map.values()):
            if isinstance(obj, _PLAN_CASCADE_MODELS):
                self._session.expire(obj)

        job = self._jobs.create(episode_id=episode_id, job_type=JobType.ARTICLE_GENERATION)
        await self._session.commit()
        await self._job_queue.enqueue_article_generation(episode_id=episode_id, job_id=job.id)
        return job
