"""Async SQLAlchemy engine factory for the agent042 database."""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from shared.config import get_settings

logger = logging.getLogger(__name__)

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine():
    """Return the shared async engine (created lazily)."""
    global _engine
    if _engine is None:
        settings = get_settings()
        if not settings.agent042_db_url:
            raise RuntimeError(
                "GATEWAY_AGENT042_DB_URL is not configured. Set it to a postgresql+asyncpg:// URL."
            )
        _engine = create_async_engine(
            settings.agent042_db_url,
            echo=False,
            pool_size=5,
            max_overflow=10,
        )
        logger.info("Async SQLAlchemy engine created")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the shared session factory (created lazily)."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def close_engine() -> None:
    """Dispose of the engine and its connection pool."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("SQLAlchemy engine disposed")
