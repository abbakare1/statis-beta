"""Database engine, session management, and schema creation."""

from __future__ import annotations

from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config.settings import get_settings


def _build_url(db_path: Path) -> str:
    return f"sqlite+aiosqlite:///{db_path.as_posix()}"


_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            _build_url(settings.DB_PATH),
            echo=settings.DEBUG,
            future=True,
        )
    return _engine


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def get_db_session() -> AsyncSession:
    """FastAPI dependency: yield a session and close it after the request."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        await session.close()


async def init_db():
    """Create all tables defined in the models module."""
    from app.models import Base  # noqa: F811
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Ensure plan_json column exists on analysis_plans (added in Phase 2J)
        try:
            await conn.execute(sa.text(
                "ALTER TABLE analysis_plans ADD COLUMN plan_json TEXT"
            ))
        except Exception:
            pass  # Column already exists


async def close_db():
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
