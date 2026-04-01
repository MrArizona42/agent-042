"""Alias assignment and promotion helpers."""

from __future__ import annotations

from typing import Any

from rag.ops.meta import read_collection_meta
from rag.vector_store import QdrantVectorStore
from shared.config import get_kb_config, get_knowledge_bases, get_settings


def _available_kbs() -> list[str]:
    registry = get_knowledge_bases()
    return [kb.name for task_cfg in registry.values() for kb in task_cfg.knowledge_bases]


def _validate_kb_alias(kb: str, alias: str) -> None:
    kb_cfg = get_kb_config(kb)
    if kb_cfg is None:
        raise ValueError(
            f"Knowledge base '{kb}' not found. Available: {', '.join(_available_kbs()) or '(none)'}"
        )
    if alias not in kb_cfg.aliases:
        raise ValueError(f"Alias '{alias}' is not allowed for knowledge base '{kb}'")


def assign_alias_to_collection(
    *,
    kb: str,
    alias: str,
    collection_name: str,
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
) -> dict[str, Any]:
    """Attach an alias to an existing validated collection."""
    _validate_kb_alias(kb, alias)
    settings = get_settings()
    qdrant_host = qdrant_host or settings.qdrant_host
    qdrant_port = qdrant_port or settings.qdrant_port

    target_store = QdrantVectorStore(
        host=qdrant_host,
        port=qdrant_port,
        collection_name=collection_name,
    )
    if not target_store.collection_exists():
        raise RuntimeError(f"Collection '{collection_name}' does not exist")

    meta = read_collection_meta(target_store, context=collection_name)
    if meta.kb_name != kb:
        raise RuntimeError(
            f"Collection '{collection_name}' belongs to '{meta.kb_name}', not '{kb}'"
        )

    alias_name = f"{kb}_{alias}"
    target_store.update_alias(alias_name, collection_name)
    return {
        "alias_name": alias_name,
        "collection_name": collection_name,
        "meta": meta.to_payload(),
    }


def promote_alias(
    *,
    kb: str,
    from_alias: str,
    to_alias: str,
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
) -> dict[str, Any]:
    """Re-point one alias to the collection behind another alias."""
    _validate_kb_alias(kb, from_alias)
    _validate_kb_alias(kb, to_alias)
    settings = get_settings()
    qdrant_host = qdrant_host or settings.qdrant_host
    qdrant_port = qdrant_port or settings.qdrant_port

    source_alias_name = f"{kb}_{from_alias}"
    source_store = QdrantVectorStore(
        host=qdrant_host,
        port=qdrant_port,
        collection_name=source_alias_name,
    )
    collection_name = source_store.resolve_alias(source_alias_name)
    if collection_name is None:
        raise RuntimeError(f"Alias '{source_alias_name}' does not resolve to a collection")

    result = assign_alias_to_collection(
        kb=kb,
        alias=to_alias,
        collection_name=collection_name,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
    )
    result["source_alias_name"] = source_alias_name
    return result


def detach_alias(
    *,
    kb: str,
    alias: str,
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
) -> dict[str, str]:
    """Delete an alias mapping."""
    _validate_kb_alias(kb, alias)
    settings = get_settings()
    qdrant_host = qdrant_host or settings.qdrant_host
    qdrant_port = qdrant_port or settings.qdrant_port
    alias_name = f"{kb}_{alias}"
    vector_store = QdrantVectorStore(
        host=qdrant_host,
        port=qdrant_port,
        collection_name=alias_name,
    )
    vector_store.delete_alias(alias_name)
    return {"alias_name": alias_name}
