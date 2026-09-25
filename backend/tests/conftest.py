import os

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core import db as db_module

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/podcast_to_knowledge_test",
    ),
)

_TABLES_IN_FK_ORDER = ("transcript_segments", "transcripts", "processing_jobs", "episodes")


@pytest_asyncio.fixture(autouse=True)
async def _reset_process_wide_db_engine():
    """app.core.db.get_engine()/get_sessionmaker() are @lru_cache'd for
    production (one engine per process). pytest-asyncio gives each async
    test its own event loop by default, and an asyncpg connection pool
    can't cross event loops -- so any code under test that goes through
    those cached accessors (the worker tasks) needs a fresh engine per
    test, bound to that test's loop.
    """
    db_module.get_engine.cache_clear()
    db_module.get_sessionmaker.cache_clear()
    yield
    if db_module.get_engine.cache_info().currsize > 0:
        await db_module.get_engine().dispose()
    db_module.get_engine.cache_clear()
    db_module.get_sessionmaker.cache_clear()


@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    """A session against a real Postgres test database (schema created via
    `alembic upgrade head` ahead of time — see README). Tables are
    truncated before each test for isolation; this isn't a rollback-only
    fixture because service code commits internally (one transaction per
    unit of work is the thing under test)."""
    engine = create_async_engine(TEST_DATABASE_URL)

    async with engine.begin() as conn:
        for table in _TABLES_IN_FK_ORDER:
            await conn.execute(text(f'TRUNCATE TABLE "{table}" RESTART IDENTITY CASCADE'))

    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()
