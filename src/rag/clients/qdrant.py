"""Qdrant client construction, shared by the collection manager and the runtime.

Both construction sites build a matched sync/async pair from the same
host/port -- factored out here so that connection-level config (timeouts,
API keys, TLS) only needs to change in one place.
"""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient, QdrantClient


def create_qdrant_clients(*, host: str, port: int) -> tuple[QdrantClient, AsyncQdrantClient]:
    """Construct a matched sync/async Qdrant client pair for one host/port."""
    return QdrantClient(host=host, port=port), AsyncQdrantClient(host=host, port=port)
