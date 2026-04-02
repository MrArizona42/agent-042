"""Shared helpers for fresh collection materialization."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Sequence

from rag.embeddings import EmbeddingService
from rag.ops.meta import CollectionMeta, write_collection_meta
from rag.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


def make_collection_name(kb_name: str) -> str:
    """Generate a timestamped collection name for a KB."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{kb_name}_{timestamp}"


def create_collection_with_meta(
    *,
    qdrant_host: str,
    qdrant_port: int,
    collection_name: str,
    dimension: int,
    meta: CollectionMeta,
) -> QdrantVectorStore:
    """Create a fresh collection and write validated metadata."""
    vector_store = QdrantVectorStore(
        host=qdrant_host,
        port=qdrant_port,
        collection_name=collection_name,
    )
    vector_store.create_collection(dimension=dimension)
    write_collection_meta(vector_store, meta, dimension=dimension)
    return vector_store


def batch_embed_and_upsert(
    *,
    vector_store: QdrantVectorStore,
    embedding_service: EmbeddingService,
    documents: Sequence[str],
    metadatas: Sequence[dict[str, Any]],
    ids: Sequence[str] | None = None,
    batch_size: int = 32,
) -> None:
    """Embed texts in batches and upsert them into Qdrant."""
    if len(documents) != len(metadatas):
        raise ValueError("documents and metadatas must have the same length")
    if ids is not None and len(documents) != len(ids):
        raise ValueError("documents and ids must have the same length")

    if not documents:
        logger.info("No documents to materialize into '%s'", vector_store.collection_name)
        return

    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        batch_documents = list(documents[start:end])
        batch_metadatas = list(metadatas[start:end])
        batch_ids = list(ids[start:end]) if ids is not None else None
        embeddings = embedding_service.embed_documents(batch_documents)
        vector_store.add_documents(
            documents=batch_documents,
            embeddings=embeddings,
            metadatas=batch_metadatas,
            ids=batch_ids,
        )
