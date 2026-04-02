"""Refresh PyTorch docs by building a successor collection from stored metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.chunking import get_chunker
from rag.embeddings import EmbeddingService
from rag.ops.materialize import (
    batch_embed_and_upsert,
    create_collection_with_meta,
    make_collection_name,
)
from rag.ops.meta import build_collection_meta, read_collection_meta
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


def update_pytorch_docs_collection(
    *,
    pytorch_docs_file: str = "assets/rag_data/pytorch_docs/pytorch_docs.json",
    kb: str = "pytorch_docs",
    alias: str | None = None,
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
    embeddings_url: str | None = None,
) -> dict[str, Any]:
    """Refresh PyTorch docs production collection from `_meta`."""
    settings = get_settings()
    alias = alias or settings.default_alias
    _validate_kb_alias(kb, alias)
    qdrant_host = qdrant_host or settings.qdrant_host
    qdrant_port = qdrant_port or settings.qdrant_port

    docs_path = Path(pytorch_docs_file)
    if not docs_path.exists():
        raise FileNotFoundError(f"PyTorch docs file not found: {docs_path}")

    qdrant_alias = f"{kb}_{alias}"
    staging_alias = f"{qdrant_alias}_staging"
    alias_store = QdrantVectorStore(
        host=qdrant_host,
        port=qdrant_port,
        collection_name=qdrant_alias,
    )
    if not alias_store.collection_exists():
        raise RuntimeError(
            f"Alias '{qdrant_alias}' is missing. Bootstrap the collection before running updates."
        )

    current_target = alias_store.resolve_alias(qdrant_alias)
    if current_target is None:
        collections = alias_store.client.get_collections().collections
        if any(collection.name == qdrant_alias for collection in collections):
            current_target = qdrant_alias
        else:
            raise RuntimeError(f"Alias '{qdrant_alias}' does not resolve to a collection")

    current_store = QdrantVectorStore(
        host=qdrant_host,
        port=qdrant_port,
        collection_name=current_target,
    )
    current_meta = read_collection_meta(current_store, context=current_target)
    build_config = current_meta.build_config

    with open(docs_path, encoding="utf-8") as file_handle:
        docs = json.load(file_handle)

    embedding_service = EmbeddingService(
        model_name=build_config.embedding_model,
        embeddings_url=embeddings_url,
    )
    chunker = get_chunker(
        strategy=build_config.chunking_strategy,
        chunk_size=build_config.chunk_size,
        chunk_overlap=build_config.chunk_overlap,
    )

    successor_collection = make_collection_name(kb)
    successor_meta = build_collection_meta(
        kb_name=kb,
        build_config=build_config,
        implementation=current_meta.implementation,
    )
    successor_store = create_collection_with_meta(
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        collection_name=successor_collection,
        dimension=embedding_service.dimension,
        meta=successor_meta,
    )
    successor_store.update_alias(staging_alias, successor_collection)

    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for doc in docs:
        chunks = chunker.chunk(doc["content"])
        for chunk in chunks:
            documents.append(chunk)
            metadatas.append(
                {
                    "task": "code",
                    "source": "pytorch_docs",
                    "url": doc["url"],
                    "title": doc["title"],
                    "scraped_at": doc["scraped_at"],
                }
            )

    batch_embed_and_upsert(
        vector_store=successor_store,
        embedding_service=embedding_service,
        documents=documents,
        metadatas=metadatas,
    )

    if current_target == qdrant_alias:
        collections = successor_store.client.get_collections().collections
        if any(collection.name == qdrant_alias for collection in collections):
            successor_store.delete_collection(qdrant_alias)

    successor_store.update_alias(qdrant_alias, successor_collection)

    info = successor_store.get_collection_info()
    return {
        "alias_name": qdrant_alias,
        "staging_alias": staging_alias,
        "old_collection_name": current_target,
        "collection_name": successor_collection,
        "meta": successor_meta.to_payload(),
        "points_count": info.get("points_count", 0),
    }
