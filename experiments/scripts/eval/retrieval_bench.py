"""Temporary collection builder for retrieval-only evaluations.

Reads the build config from the ``_meta`` sentinel of a production collection,
then indexes the benchmark corpus into a temporary Qdrant collection using
the same embedding model and chunking configuration.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def read_build_config(
    *,
    kb_name: str,
    rag_alias: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
) -> dict[str, Any] | None:
    """Read the ``_meta`` sentinel from the production collection.

    The alias ``{kb_name}_{rag_alias}`` is resolved to obtain the build
    config (embedding model, chunking strategy, etc.).

    Returns:
        Meta payload dict, or ``None`` if not found.
    """
    from rag.vector_store import QdrantVectorStore

    alias_name = f"{kb_name}_{rag_alias}"
    vs = QdrantVectorStore(host=qdrant_host, port=qdrant_port, collection_name=alias_name)
    meta = vs.read_meta()
    if meta is None:
        logger.warning("No _meta point in alias '%s'", alias_name)
    return meta


def build_temp_collection(
    *,
    kb_name: str,
    dataset_name: str,
    rag_alias: str,
    corpus: list[dict[str, Any]],
    build_config: dict[str, Any],
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    embeddings_url: str = "http://localhost:8100",
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

    embedding_model = build_config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size = build_config.get("chunk_size", 512)
    chunk_overlap = build_config.get("chunk_overlap", 50)
    task = build_config.get("task", "chat")

    emb_service = EmbeddingService(model_name=embedding_model)
    chunker = get_chunker(task=task, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Chunk corpus
    all_texts: list[str] = []
    all_ids: list[str] = []
    all_meta: list[dict] = []
    for doc in corpus:
        chunks = chunker.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_ids.append(f"{doc['doc_id']}_{i}")
            all_meta.append({"source": doc.get("doc_id", ""), "task": task})

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

    logger.info(
        "Built temporary collection '%s' with %d chunks", collection_name, len(all_texts)
    )
    return collection_name


def delete_temp_collection(
    collection_name: str,
    *,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
) -> None:
    """Delete a temporary evaluation collection."""
    from rag.vector_store import QdrantVectorStore

    vs = QdrantVectorStore(host=qdrant_host, port=qdrant_port, collection_name=collection_name)
    vs.delete_collection(collection_name)
    logger.info("Deleted temporary collection: %s", collection_name)
