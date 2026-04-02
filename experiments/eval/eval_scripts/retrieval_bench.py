"""Temporary collection builder for retrieval-only evaluations.

Reads the build config from the ``_meta`` sentinel of a production collection,
then indexes the benchmark corpus into a temporary Qdrant collection using
the same embedding model and chunking configuration.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def read_build_config(
    *,
    kb_name: str,
    rag_alias: str,
    qdrant_host: str,
    qdrant_port: int,
) -> dict[str, Any] | None:
    """Read the ``_meta`` sentinel from the production collection.

    The alias ``{kb_name}_{rag_alias}`` is resolved to obtain the build
    config (embedding model, chunking strategy, etc.).

    Returns:
        Meta payload dict, or ``None`` if not found.
    """
    from rag.ops.meta import read_build_config_for_alias

    try:
        return read_build_config_for_alias(
            kb_name=kb_name,
            rag_alias=rag_alias,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
        )
    except RuntimeError:
        logger.warning("Cannot read validated build config for alias '%s_%s'", kb_name, rag_alias)
        return None


def build_temp_collection(
    *,
    kb_name: str,
    dataset_name: str,
    rag_alias: str,
    corpus: list[dict[str, Any]],
    build_config: Any,
    qdrant_host: str,
    qdrant_port: int,
    embeddings_url: str,
) -> str:
    """Build a temporary Qdrant collection from benchmark corpus.

    Uses the build config extracted from the production collection's
    ``_meta`` point to replicate the same embedding / chunking setup.

    Args:
        kb_name: Knowledge base name (e.g. ``arxiv``).
        dataset_name: Benchmark name (e.g. ``beir_scifact``).
        rag_alias: Alias role being evaluated.
        corpus: List of ``{"doc_id": ..., "text": ...}`` dicts.
        build_config: Config from ``read_build_config``.
        qdrant_host: Qdrant host.
        qdrant_port: Qdrant port.
        embeddings_url: URL of the embeddings microservice.

    Returns:
        Name of the temporary collection.
    """
    from rag.chunking import get_chunker
    from rag.embeddings import EmbeddingService
    from rag.vector_store import QdrantVectorStore

    collection_name = f"eval_{kb_name}_{dataset_name}_{rag_alias}_{_timestamp()}"
    logger.info("Building temporary collection: %s", collection_name)

    embedding_model = build_config.embedding_model
    chunk_size = build_config.chunk_size
    chunk_overlap = build_config.chunk_overlap
    chunking_strategy = build_config.chunking_strategy

    # Use chunking strategy directly with get_chunker
    emb_service = EmbeddingService(
        model_name=embedding_model,
        embeddings_url=embeddings_url,
    )
    chunker = get_chunker(
        strategy=chunking_strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Chunk corpus
    all_texts: list[str] = []
    all_ids: list[str] = []
    all_meta: list[dict] = []
    for doc in corpus:
        chunks = chunker.chunk(doc["text"])
        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            chunk_key = f"{collection_name}:{doc['doc_id']}:{i}"
            all_ids.append(str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_key)))
            all_meta.append({"source": doc.get("doc_id", ""), "kb": kb_name})

    if not all_texts:
        logger.warning("No chunks produced from corpus")
        return collection_name

    # Embed
    embeddings = emb_service.embed_documents(all_texts)
    dimension = len(embeddings[0])

    # Create collection and add documents
    vs = QdrantVectorStore(host=qdrant_host, port=qdrant_port, collection_name=collection_name)
    vs.create_collection(dimension=dimension)
    vs.add_documents(documents=all_texts, embeddings=embeddings, metadatas=all_meta, ids=all_ids)

    logger.info("Built temporary collection '%s' with %d chunks", collection_name, len(all_texts))
    return collection_name


def delete_temp_collection(
    collection_name: str,
    *,
    qdrant_host: str,
    qdrant_port: int,
) -> None:
    """Delete a temporary evaluation collection."""
    from rag.vector_store import QdrantVectorStore

    vs = QdrantVectorStore(host=qdrant_host, port=qdrant_port, collection_name=collection_name)
    vs.delete_collection(collection_name)
    logger.info("Deleted temporary collection: %s", collection_name)
