from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db_session
from app.services.episode_service import EpisodeService
from app.services.job_queue import JobQueue, get_job_queue

DbSession = Annotated[AsyncSession, Depends(get_db_session)]
Queue = Annotated[JobQueue, Depends(get_job_queue)]


async def get_episode_service(session: DbSession, job_queue: Queue) -> AsyncGenerator[EpisodeService, None]:
    yield EpisodeService(session, job_queue)


EpisodeServiceDep = Annotated[EpisodeService, Depends(get_episode_service)]
