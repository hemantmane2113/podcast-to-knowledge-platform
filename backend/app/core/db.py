from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings


@lru_cache
def get_engine() -> AsyncEngine:
    """Process-wide async engine.

    Importable from the API process and the arq worker process alike —
    deliberately not a FastAPI dependency itself, so DB access isn't
    coupled to the request lifecycle (see ARCHITECTURE.md).
    """
    settings = get_settings()
    return create_async_engine(settings.database_url, pool_pre_ping=True)


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=get_engine(), expire_on_commit=False)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: one session per request, committed on success."""
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        yield session
