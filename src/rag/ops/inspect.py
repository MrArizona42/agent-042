"""Inspection helpers for collection and alias state."""

from __future__ import annotations

from typing import Any

from rag.ops.meta import read_collection_meta
from rag.vector_store import QdrantVectorStore
from shared.config import get_settings


def list_alias_mappings(
    *,
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
    kb_name: str | None = None,
) -> list[dict[str, str]]:
    """List alias-to-collection mappings visible in Qdrant."""
    settings = get_settings()
    qdrant_host = qdrant_host or settings.platform.qdrant_host
    qdrant_port = qdrant_port or settings.platform.qdrant_port
    vector_store = QdrantVectorStore(
        host=qdrant_host,
        port=qdrant_port,
        collection_name="_alias_inspector",
    )
    aliases = vector_store.list_aliases()
    if kb_name is None:
        return aliases
    prefix = f"{kb_name}_"
    return [alias for alias in aliases if alias["alias_name"].startswith(prefix)]


def inspect_collection(
    *,
    collection_name: str,
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
) -> dict[str, Any]:
    """Inspect a concrete collection and its metadata."""
    settings = get_settings()
    qdrant_host = qdrant_host or settings.platform.qdrant_host
    qdrant_port = qdrant_port or settings.platform.qdrant_port
    vector_store = QdrantVectorStore(
        host=qdrant_host,
        port=qdrant_port,
        collection_name=collection_name,
    )
    info = vector_store.get_collection_info()
    meta = read_collection_meta(vector_store, context=collection_name)
    return {
        "collection_name": collection_name,
        "collection_info": info,
        "meta": meta.to_payload(),
    }


def inspect_alias(
    *,
    kb_name: str,
    alias: str,
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
) -> dict[str, Any]:
    """Inspect an alias, its resolved collection, and metadata."""
    settings = get_settings()
    qdrant_host = qdrant_host or settings.platform.qdrant_host
    qdrant_port = qdrant_port or settings.platform.qdrant_port
    alias_name = f"{kb_name}_{alias}"
    vector_store = QdrantVectorStore(
        host=qdrant_host,
        port=qdrant_port,
        collection_name=alias_name,
    )
    resolved_collection = vector_store.resolve_alias(alias_name) or alias_name
    target_store = QdrantVectorStore(
        host=qdrant_host,
        port=qdrant_port,
        collection_name=resolved_collection,
    )
    return {
        "alias_name": alias_name,
        "resolved_collection": resolved_collection,
        "collection_info": target_store.get_collection_info(),
        "meta": read_collection_meta(target_store, context=resolved_collection).to_payload(),
    }
