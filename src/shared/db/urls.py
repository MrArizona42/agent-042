"""Centralized Postgres URL normalization for sync and async SQLAlchemy engines."""

from __future__ import annotations

_ASYNC_PREFIX = "postgresql+asyncpg://"
_SYNC_PREFIX = "postgresql+psycopg2://"


def require_db_url(db_url: str | None, *, purpose: str) -> str:
    """Raise a clear error if *db_url* is unset; otherwise return it unchanged."""
    if not db_url:
        raise ValueError(f"{purpose} requires a database URL")
    return db_url


def to_sync_url(db_url: str) -> str:
    """Return the `postgresql+psycopg2://` form of a Postgres connection URL."""
    if db_url.startswith(_ASYNC_PREFIX):
        return _SYNC_PREFIX + db_url.removeprefix(_ASYNC_PREFIX)
    return db_url


def to_async_url(db_url: str) -> str:
    """Return the `postgresql+asyncpg://` form of a Postgres connection URL."""
    if db_url.startswith(_SYNC_PREFIX):
        return _ASYNC_PREFIX + db_url.removeprefix(_SYNC_PREFIX)
    return db_url
