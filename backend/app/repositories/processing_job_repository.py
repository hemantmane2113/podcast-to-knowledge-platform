import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.processing_job import JobStatus, JobType, ProcessingJob


class ProcessingJobRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_by_id(self, job_id: uuid.UUID) -> ProcessingJob | None:
        return await self._session.get(ProcessingJob, job_id)

    async def get_active_job(
        self, episode_id: uuid.UUID, job_type: JobType
    ) -> ProcessingJob | None:
        result = await self._session.execute(
            select(ProcessingJob).where(
                ProcessingJob.episode_id == episode_id,
                ProcessingJob.job_type == job_type,
                ProcessingJob.status.in_((JobStatus.PENDING, JobStatus.RUNNING)),
            )
        )
        return result.scalar_one_or_none()

    async def get_latest_job(
        self, episode_id: uuid.UUID, job_type: JobType
    ) -> ProcessingJob | None:
        result = await self._session.execute(
            select(ProcessingJob)
            .where(ProcessingJob.episode_id == episode_id, ProcessingJob.job_type == job_type)
            .order_by(ProcessingJob.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    def create(self, episode_id: uuid.UUID, job_type: JobType) -> ProcessingJob:
        job = ProcessingJob(
            id=uuid.uuid4(),
            episode_id=episode_id,
            job_type=job_type,
            status=JobStatus.PENDING,
        )
        self._session.add(job)
        return job

    def mark_running(self, job: ProcessingJob) -> None:
        job.status = JobStatus.RUNNING
        if job.started_at is None:
            # Preserve the original start time across retry attempts.
            job.started_at = datetime.now(UTC)

    def mark_completed(self, job: ProcessingJob) -> None:
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now(UTC)

    def mark_failed(self, job: ProcessingJob, error_message: str) -> None:
        job.status = JobStatus.FAILED
        job.completed_at = datetime.now(UTC)
        job.error_message = error_message
