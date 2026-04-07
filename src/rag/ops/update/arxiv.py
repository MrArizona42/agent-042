"""Refresh an existing ArXiv collection from stored metadata."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from rag.chunking import get_chunker
from rag.embeddings import EmbeddingService
from rag.ops.materialize import batch_embed_and_upsert
from rag.ops.update.common import load_update_collection_meta
from rag.vector_store import QdrantVectorStore
from shared.config import get_kb_config, get_settings, validate_kb_alias

_POINT_ID_NS = uuid.UUID("b8c9d0e1-f2a3-4b5c-6d7e-8f9a0b1c2d3e")


def update_arxiv_collection(
    *,
    arxiv_file: str = "assets/rag_data/arxiv/arxiv_papers.json",
    kb: str = "arxiv",
    alias: str | None = None,
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
    embeddings_url: str | None = None,
) -> dict[str, Any]:
    """Refresh an existing ArXiv production collection from `_meta`."""
    settings = get_settings()
    alias = alias or get_kb_config(kb).default_alias
    validate_kb_alias(kb, alias)
    qdrant_host = qdrant_host or settings.qdrant_host
    qdrant_port = qdrant_port or settings.qdrant_port

    arxiv_path = Path(arxiv_file)
    if not arxiv_path.exists():
        raise FileNotFoundError(f"ArXiv file not found: {arxiv_path}")

    qdrant_alias = f"{kb}_{alias}"
    alias_store = QdrantVectorStore(
        host=qdrant_host,
        port=qdrant_port,
        collection_name=qdrant_alias,
    )
    if not alias_store.collection_exists():
        raise RuntimeError(
            f"Alias '{qdrant_alias}' is missing. Bootstrap the collection before running updates."
        )

    target_collection_name = alias_store.resolve_alias(qdrant_alias) or qdrant_alias
    target_store = QdrantVectorStore(
        host=qdrant_host,
        port=qdrant_port,
        collection_name=target_collection_name,
    )
    meta = load_update_collection_meta(
        vector_store=target_store,
        alias_name=qdrant_alias,
        collection_name=target_collection_name,
        kb_name=kb,
    )
    build_config = meta.build_config

    with open(arxiv_path, encoding="utf-8") as file_handle:
        papers = json.load(file_handle)

    embedding_service = EmbeddingService(
        model_name=build_config.embedding_model,
        embeddings_url=embeddings_url,
    )
    chunker = get_chunker(
        strategy=build_config.chunking_strategy,
        chunk_size=build_config.chunk_size,
        chunk_overlap=build_config.chunk_overlap,
    )

    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []
    for paper in papers:
        full_text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
        chunks = chunker.chunk(full_text)
        for chunk_idx, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append(
                {
                    "task": "chat",
                    "source": "arxiv",
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "primary_category": paper["primary_category"],
                    "published": paper["published"],
                }
            )
            ids.append(str(uuid.uuid5(_POINT_ID_NS, f"arxiv:{paper['arxiv_id']}:{chunk_idx}")))

    batch_embed_and_upsert(
        vector_store=target_store,
        embedding_service=embedding_service,
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    info = target_store.get_collection_info()
    return {
        "alias_name": qdrant_alias,
        "collection_name": target_collection_name,
        "meta": meta.to_payload(),
        "points_count": info.get("points_count", 0),
    }
