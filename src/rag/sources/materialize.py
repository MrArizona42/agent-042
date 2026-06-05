"""Materialize source chunk bundles into Qdrant collections."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client.models import SparseVector

from rag.domain import IndexManifest, RetrievalCapability
from rag.domain.manifests import (
    attestation_from_payload,
    attestation_payload,
    manifest_path,
    write_index_manifest,
)
from rag.sources.bundles import SourceChunkBundle
from rag.sources.chunks import LLAMAINDEX_SENTENCE_SPLITTER

RetrievalStrategy = Literal["dense", "hybrid"]
SourceRetrievalCapability = Literal["dense", "hybrid"]

_POINT_ID_NS = uuid.UUID("46fe6fc7-dbd6-4934-9a73-2cc6ccfbef28")


class EmbeddingClient(Protocol):
    """Dense embedding client contract used by source materialization."""

    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts."""
        ...


class SparseEmbeddingClient(Protocol):
    """Sparse embedding client contract used by hybrid materialization."""

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        """Encode document texts as sparse vectors."""
        ...


class SourceVectorStore(Protocol):
    """Qdrant-like vector store contract used by source materialization."""

    collection_name: str

    def create_collection(
        self,
        dimension: int,
        retrieval_capability: str = "dense",
        force_recreate: bool = False,
    ) -> None:
        """Create the backing collection."""
        ...

    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        sparse_vectors: list[SparseVector] | None = None,
        upsert_batch_size: int = 500,
    ) -> None:
        """Add documents to the backing collection."""
        ...

    def write_meta(self, payload: dict, dimension: int) -> None:
        """Write collection metadata."""
        ...

    def read_meta(self) -> dict | None:
        """Read collection metadata."""
        ...

    def collection_exists(self) -> bool:
        """Return whether the collection exists."""
        ...

    def update_alias(self, alias_name: str, collection_name: str) -> None:
        """Create or update an alias."""
        ...


class MaterializationSummary(BaseModel):
    """Summary for one Qdrant materialization."""

    model_config = ConfigDict(extra="forbid")

    kb_id: str
    collection_name: str
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    retrieval_capability: RetrievalCapability
    vector_size: int = Field(gt=0)
    sparse_enabled: bool
    qdrant_upsert_batch_size: int = Field(gt=0)


class MaterializationResult(BaseModel):
    """Materialization summary plus the written manifest."""

    model_config = ConfigDict(extra="forbid")

    summary: MaterializationSummary
    manifest: IndexManifest
    manifest_path: str


class AliasPromotionResult(BaseModel):
    """Result of promoting a collection behind an alias."""

    model_config = ConfigDict(extra="forbid")

    alias_name: str
    collection_name: str
    manifest_id: str


def collection_name_for_build(
    *,
    kb_id: str,
    created_at: datetime | None = None,
) -> str:
    """Return a conventional timestamped physical collection name."""
    created_at = created_at or datetime.now(tz=UTC)
    stamp = created_at.strftime("%Y%m%d_%H%M%S")
    return f"rag__{kb_id}__{stamp}"


def qdrant_alias_name(*, kb_id: str, alias: str) -> str:
    """Return the conventional Qdrant alias name for a KB alias."""
    return f"rag__{kb_id}__{alias}"


def validate_strategy_supported(
    *,
    retrieval_strategy: RetrievalStrategy,
    retrieval_capability: SourceRetrievalCapability,
) -> None:
    """Validate that a query strategy is supported by collection capability."""
    if retrieval_strategy == "dense":
        return
    if retrieval_strategy == "hybrid" and retrieval_capability == "hybrid":
        return
    raise ValueError(
        f"retrieval_strategy '{retrieval_strategy}' is not supported by "
        f"retrieval_capability '{retrieval_capability}'"
    )


def retrieval_capability_for_strategy(
    retrieval_strategy: RetrievalStrategy,
) -> SourceRetrievalCapability:
    """Return the minimum physical collection capability for a retrieval strategy."""
    if retrieval_strategy == "hybrid":
        return "hybrid"
    return "dense"


def source_snapshot_id(bundles: list[SourceChunkBundle]) -> str:
    """Hash chunk artifact identities and checksums into a source snapshot id."""
    payload = [
        {
            "kb_id": bundle.kb_id,
            "source_instance_id": bundle.source_instance_id,
            "chunk_artifact_checksums": bundle.chunk_artifact_checksums,
        }
        for bundle in bundles
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_ID_NS, chunk_id))


def _all_chunks(bundles: list[SourceChunkBundle]):
    for bundle in bundles:
        yield from bundle.chunks


def _chunking_config(bundles: list[SourceChunkBundle]) -> dict[str, object]:
    return {
        "strategy": LLAMAINDEX_SENTENCE_SPLITTER,
        "source_instance_ids": [bundle.source_instance_id for bundle in bundles],
    }


def materialize_kb_collection(
    *,
    kb_id: str,
    collection_name: str,
    bundles: list[SourceChunkBundle],
    vector_store: SourceVectorStore,
    embedding_client: EmbeddingClient,
    embedding_model: str,
    retrieval_capability: SourceRetrievalCapability,
    rag_data_root: Path | str,
    target_alias: str | None = None,
    sparse_encoder_model: str | None = None,
    sparse_encoder_client: SparseEmbeddingClient | None = None,
    qdrant_upsert_batch_size: int = 128,
    force_recreate: bool = False,
    build_config_ref: str | None = None,
    created_at: datetime | None = None,
) -> MaterializationResult:
    """Materialize chunk bundles into a Qdrant collection and write its manifest."""
    if not bundles:
        raise ValueError("at least one source chunk bundle is required")
    if any(bundle.kb_id != kb_id for bundle in bundles):
        raise ValueError("all source chunk bundles must belong to the target kb_id")
    if retrieval_capability == "hybrid" and sparse_encoder_client is None:
        raise ValueError("hybrid materialization requires a sparse_encoder_client")
    if retrieval_capability == "hybrid" and not sparse_encoder_model:
        raise ValueError("hybrid materialization requires sparse_encoder_model")

    chunks = list(_all_chunks(bundles))
    if not chunks:
        raise ValueError("at least one chunk is required for materialization")
    texts = [chunk.text for chunk in chunks]
    embeddings = embedding_client.embed_documents(texts)
    if len(embeddings) != len(chunks):
        raise ValueError("embedding_client returned a vector count that does not match chunks")
    sparse_vectors = (
        sparse_encoder_client.encode_documents(texts)
        if retrieval_capability == "hybrid" and sparse_encoder_client is not None
        else None
    )
    if sparse_vectors is not None and len(sparse_vectors) != len(chunks):
        raise ValueError("sparse_encoder_client returned a vector count that does not match chunks")
    metadatas = [
        {
            **chunk.metadata,
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "source_document_id": chunk.source_document_id,
            "section_title": chunk.section_title,
            "ordinal": chunk.ordinal,
            "token_count": chunk.token_count,
        }
        for chunk in chunks
    ]
    point_ids = [_point_id(chunk.id) for chunk in chunks]

    vector_store.create_collection(
        dimension=embedding_client.dimension,
        retrieval_capability=retrieval_capability,
        force_recreate=force_recreate,
    )
    vector_store.add_documents(
        texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=point_ids,
        sparse_vectors=sparse_vectors,
        upsert_batch_size=qdrant_upsert_batch_size,
    )

    created_at = created_at or datetime.now(tz=UTC)
    manifest = IndexManifest(
        kb_id=kb_id,
        collection_name=collection_name,
        alias=target_alias,
        source_snapshot_id=source_snapshot_id(bundles),
        document_count=sum(bundle.document_count for bundle in bundles),
        chunk_count=len(chunks),
        embedding_model=embedding_model,
        sparse_encoder=sparse_encoder_model if retrieval_capability == "hybrid" else None,
        retrieval_capability=RetrievalCapability(retrieval_capability),
        chunking_config=_chunking_config(bundles),
        extraction_config={},
        build_config_ref=build_config_ref,
        created_at=created_at,
    )
    path = manifest_path(
        rag_data_root=rag_data_root,
        kb_id=kb_id,
        collection_name=collection_name,
    )
    manifest = write_index_manifest(path, manifest)
    vector_store.write_meta(
        attestation_payload(manifest.to_attestation()),
        embedding_client.dimension,
    )

    summary = MaterializationSummary(
        kb_id=kb_id,
        collection_name=collection_name,
        document_count=sum(bundle.document_count for bundle in bundles),
        chunk_count=len(chunks),
        retrieval_capability=RetrievalCapability(retrieval_capability),
        vector_size=embedding_client.dimension,
        sparse_enabled=retrieval_capability == "hybrid",
        qdrant_upsert_batch_size=qdrant_upsert_batch_size,
    )
    return MaterializationResult(
        summary=summary,
        manifest=manifest,
        manifest_path=path.as_posix(),
    )


def promote_materialized_alias(
    *,
    kb_id: str,
    alias: str,
    collection_name: str,
    vector_store: SourceVectorStore,
) -> AliasPromotionResult:
    """Point a conventional KB alias at an attested materialized collection."""
    if not vector_store.collection_exists():
        raise RuntimeError(f"Collection '{collection_name}' does not exist")
    payload = vector_store.read_meta()
    if payload is None:
        raise RuntimeError(f"Collection '{collection_name}' has no attestation metadata")
    attestation = attestation_from_payload(payload)
    if attestation.kb_id != kb_id:
        raise RuntimeError(
            f"Collection '{collection_name}' belongs to '{attestation.kb_id}', not '{kb_id}'"
        )
    if attestation.collection_name != collection_name:
        raise RuntimeError(
            f"Collection attestation names '{attestation.collection_name}', not '{collection_name}'"
        )

    alias_name = qdrant_alias_name(kb_id=kb_id, alias=alias)
    vector_store.update_alias(alias_name, collection_name)
    return AliasPromotionResult(
        alias_name=alias_name,
        collection_name=collection_name,
        manifest_id=attestation.manifest_id,
    )
