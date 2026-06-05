"""Temporary collection builder for retrieval-only evaluations.

Reads the build config from the ``_meta`` sentinel of a production collection,
then indexes the benchmark corpus into a temporary Qdrant collection using
the same embedding model, chunking configuration, and retrieval capability.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from rag.domain import RetrievalCapability
from rag.domain.manifests import attestation_from_payload
from rag.sources.materialize import qdrant_alias_name

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalBuildConfig:
    """Minimal production-index shape needed to mirror retrieval evals."""

    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    sparse_encoder: str | None
    retrieval_capability: RetrievalCapability

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable eval metadata payload."""
        return {
            "chunking_strategy": self.chunking_strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "sparse_encoder": self.sparse_encoder,
            "retrieval_capability": self.retrieval_capability.value,
        }


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def read_build_config(
    *,
    kb_name: str,
    rag_alias: str,
    qdrant_host: str,
    qdrant_port: int,
) -> EvalBuildConfig | None:
    """Read current collection attestation for a production KB alias.

    The source lifecycle now stores the full build manifest as an artifact and
    writes only compact attestation metadata into Qdrant. Retrieval evals only
    need the vector-leg setup plus a chunker for temporary benchmark corpora,
    so this returns the attested embedding/capability and conservative chunking
    defaults until eval jobs get first-class manifest artifact access.

    Returns:
        Minimal build config, or ``None`` if the alias is not attested.
    """
    from rag.vector_store import QdrantVectorStore

    try:
        alias_name = qdrant_alias_name(kb_id=kb_name, alias=rag_alias)
        alias_store = QdrantVectorStore(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=alias_name,
        )
        collection_name = alias_store.resolve_alias(alias_name)
        if collection_name is None:
            logger.warning("Cannot resolve RAG alias '%s'", alias_name)
            return None

        collection_store = QdrantVectorStore(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name,
        )
        payload = collection_store.read_meta()
        if payload is None:
            logger.warning("Collection '%s' has no attestation metadata", collection_name)
            return None

        attestation = attestation_from_payload(payload)
        return EvalBuildConfig(
            chunking_strategy="recursive",
            chunk_size=512,
            chunk_overlap=64,
            embedding_model=attestation.embedding_model,
            sparse_encoder=attestation.sparse_encoder,
            retrieval_capability=attestation.retrieval_capability,
        )
    except Exception:
        logger.warning("Cannot read attested build config for alias '%s/%s'", kb_name, rag_alias)
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
    from rag.sparse_encoder import SparseEncoderService
    from rag.vector_store import QdrantVectorStore

    collection_name = f"eval_{kb_name}_{dataset_name}_{rag_alias}_{_timestamp()}"
    logger.info("Building temporary collection: %s", collection_name)

    embedding_model = build_config.embedding_model
    chunk_size = build_config.chunk_size
    chunk_overlap = build_config.chunk_overlap
    chunking_strategy = build_config.chunking_strategy

    retrieval_capability = build_config.retrieval_capability
    has_dense_leg = retrieval_capability in {"dense", "hybrid"}
    has_sparse_leg = retrieval_capability in {"hybrid", "sparse"}

    emb_service = None
    if has_dense_leg:
        emb_service = EmbeddingService(
            model_name=embedding_model,
            embeddings_url=embeddings_url,
        )

    sparse_encoder = None
    if has_sparse_leg:
        sparse_encoder = SparseEncoderService(embeddings_url=embeddings_url)

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

    try:
        embeddings = emb_service.embed_documents(all_texts) if emb_service is not None else None
        sparse_vectors = (
            sparse_encoder.encode_documents(all_texts) if sparse_encoder is not None else None
        )
        dimension = len(embeddings[0]) if embeddings is not None else 1

        # Create collection and add documents using the same vector legs as production.
        vs = QdrantVectorStore(host=qdrant_host, port=qdrant_port, collection_name=collection_name)
        vs.create_collection(
            dimension=dimension,
            retrieval_capability=retrieval_capability,
        )
        vs.add_documents(
            documents=all_texts,
            embeddings=embeddings,
            metadatas=all_meta,
            ids=all_ids,
            sparse_vectors=sparse_vectors,
        )
    finally:
        if emb_service is not None:
            emb_service.close()
        if sparse_encoder is not None:
            sparse_encoder.close()

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
