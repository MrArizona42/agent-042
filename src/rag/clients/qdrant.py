"""Qdrant client construction, shared across rag/ and dags/ entrypoints.

Connection-level config (timeouts, API keys, TLS) only needs to change here.

Use the single-client factories when a caller may already have one half of
the pair injected (e.g. a sync client in tests) -- they preserve short-circuit
laziness, so the other half is never constructed unless actually needed. Use
create_qdrant_clients() only when a fresh pair is wanted unconditionally.
"""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient, QdrantClient


def create_qdrant_client(*, host: str, port: int) -> QdrantClient:
    """Construct a sync Qdrant client for one host/port."""
    return QdrantClient(host=host, port=port)


def create_async_qdrant_client(*, host: str, port: int) -> AsyncQdrantClient:
    """Construct an async Qdrant client for one host/port."""
    return AsyncQdrantClient(host=host, port=port)


def create_qdrant_clients(*, host: str, port: int) -> tuple[QdrantClient, AsyncQdrantClient]:
    """Construct a fresh, matched sync/async Qdrant client pair for one host/port."""
    return (
        create_qdrant_client(host=host, port=port),
        create_async_qdrant_client(host=host, port=port),
    )
