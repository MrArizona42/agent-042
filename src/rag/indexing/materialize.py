"""Materialize source nodes through LlamaIndex and promote attested collections."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client.models import SparseVector

from rag.contracts import CollectionAttestation, IndexManifest, RetrievalCapability
from rag.contracts.manifests import (
    manifest_path,
    write_index_manifest,
)
from rag.indexing.llamaindex_embeddings import ProjectEmbedding, ProjectSparseEncoder
from rag.sources.bundles import SourceNodeBundle
from rag.sources.chunks import LLAMAINDEX_SENTENCE_SPLITTER, read_chunk_artifact

RetrievalStrategy = Literal["dense", "hybrid", "sparse"]
SourceRetrievalCapability = Literal["dense", "hybrid"]

_logger = logging.getLogger(__name__)


class EmbeddingClient(Protocol):
    """Dense embedding client contract used by source materialization."""

    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed one query."""
        ...


class SparseEmbeddingClient(Protocol):
    """Sparse embedding client contract used by hybrid materialization."""

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        """Encode document texts as sparse vectors."""
        ...


class CollectionManager(Protocol):
    """Project-owned collection metadata and alias operations."""

    collection_name: str

    def prepare_new_collection(
        self,
        *,
        force_recreate: bool,
    ) -> None:
        """Ensure a clean physical collection can be created."""
        ...

    def vector_store(
        self,
        *,
        vector_size: int,
        batch_size: int,
        enable_hybrid: bool,
        sparse_encoder: ProjectSparseEncoder | None,
    ) -> QdrantVectorStore:
        """Return the LlamaIndex store for this collection."""
        ...

    def write_attestation(self, attestation: CollectionAttestation) -> None:
        """Write collection-level attestation metadata."""
        ...

    def read_attestation(self) -> CollectionAttestation | None:
        """Read collection-level attestation metadata."""
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
    if retrieval_strategy in {"hybrid", "sparse"} and retrieval_capability == "hybrid":
        return
    raise ValueError(
        f"retrieval_strategy '{retrieval_strategy}' is not supported by "
        f"retrieval_capability '{retrieval_capability}'"
    )


def retrieval_capability_for_strategy(
    retrieval_strategy: RetrievalStrategy,
) -> SourceRetrievalCapability:
    """Return the minimum physical collection capability for a retrieval strategy."""
    if retrieval_strategy in {"hybrid", "sparse"}:
        return "hybrid"
    return "dense"


def source_snapshot_id(bundles: list[SourceNodeBundle]) -> str:
    """Hash node artifact identities and checksums into a source snapshot id."""
    payload = [
        {
            "kb_id": bundle.kb_id,
            "source_instance_id": bundle.source_instance_id,
            "node_artifact_checksums": bundle.node_artifact_checksums,
        }
        for bundle in bundles
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _all_nodes(bundles: list[SourceNodeBundle]):
    for bundle in bundles:
        yield from bundle.nodes


def _chunking_config(bundles: list[SourceNodeBundle]) -> dict[str, object]:
    config: dict[str, object] = {
        "strategy": LLAMAINDEX_SENTENCE_SPLITTER,
        "source_instance_ids": [bundle.source_instance_id for bundle in bundles],
    }
    for bundle in bundles:
        for artifact_path in bundle.node_artifact_paths:
            path = Path(artifact_path)
            if path.exists():
                try:
                    artifact = read_chunk_artifact(path)
                    config["chunk_size"] = artifact.chunking.chunk_size
                    config["chunk_overlap"] = artifact.chunking.chunk_overlap
                    config["method"] = artifact.chunking.method
                except Exception as exc:
                    _logger.debug("Could not read chunking config from %s: %s", path, exc)
                return config
    return config


def materialize_kb_collection_llamaindex(
    *,
    kb_id: str,
    collection_name: str,
    bundles: list[SourceNodeBundle],
    collection_manager: CollectionManager,
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
    build_config_digest: str | None = None,
    build_profile_digest: str | None = None,
    source_adapter_versions: dict[str, str] | None = None,
    source_manifest_digests: dict[str, str] | None = None,
    benchmark_scope: str | None = None,
    created_at: datetime | None = None,
) -> MaterializationResult:
    """Materialize native nodes through LlamaIndex and attest the collection."""
    if not bundles:
        raise ValueError("at least one source node bundle is required")
    if any(bundle.kb_id != kb_id for bundle in bundles):
        raise ValueError("all source node bundles must belong to the target kb_id")
    if collection_manager.collection_name != collection_name:
        raise ValueError(
            "collection manager name does not match requested physical collection "
            f"('{collection_manager.collection_name}' != '{collection_name}')"
        )
    if retrieval_capability == "hybrid" and sparse_encoder_client is None:
        raise ValueError("hybrid materialization requires a sparse_encoder_client")
    if retrieval_capability == "hybrid" and not sparse_encoder_model:
        raise ValueError("hybrid materialization requires sparse_encoder_model")

    nodes = list(_all_nodes(bundles))
    if not nodes:
        raise ValueError("at least one node is required for materialization")
    if len({node.id_ for node in nodes}) != len(nodes):
        raise ValueError("node ids must be unique within one materialization")

    collection_manager.prepare_new_collection(force_recreate=force_recreate)
    project_embedding = ProjectEmbedding(
        embedding_client=embedding_client,
        model_name=embedding_model,
    )
    sparse_encoder = (
        ProjectSparseEncoder(sparse_encoder_client)
        if retrieval_capability == "hybrid" and sparse_encoder_client is not None
        else None
    )
    vector_store = collection_manager.vector_store(
        vector_size=embedding_client.dimension,
        batch_size=qdrant_upsert_batch_size,
        enable_hybrid=retrieval_capability == "hybrid",
        sparse_encoder=sparse_encoder,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=project_embedding,
        insert_batch_size=qdrant_upsert_batch_size,
    )

    created_at = created_at or datetime.now(tz=UTC)
    manifest = IndexManifest(
        kb_id=kb_id,
        collection_name=collection_name,
        alias=target_alias,
        source_snapshot_id=source_snapshot_id(bundles),
        source_manifest_digests=source_manifest_digests or {},
        source_adapter_versions=source_adapter_versions or {},
        document_count=sum(bundle.document_count for bundle in bundles),
        chunk_count=len(nodes),
        embedding_model=embedding_model,
        vector_dimension=embedding_client.dimension,
        sparse_encoder=sparse_encoder_model if retrieval_capability == "hybrid" else None,
        retrieval_capability=RetrievalCapability(retrieval_capability),
        chunking_config=_chunking_config(bundles),
        extraction_config={},
        build_config_ref=build_config_ref,
        build_config_digest=build_config_digest,
        build_profile_digest=build_profile_digest,
        benchmark_scope=benchmark_scope,
        created_at=created_at,
    )
    path = manifest_path(
        rag_data_root=rag_data_root,
        kb_id=kb_id,
        collection_name=collection_name,
    )
    manifest = write_index_manifest(path, manifest)
    collection_manager.write_attestation(manifest.to_attestation())

    summary = MaterializationSummary(
        kb_id=kb_id,
        collection_name=collection_name,
        document_count=sum(bundle.document_count for bundle in bundles),
        chunk_count=len(nodes),
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
    collection_manager: CollectionManager,
) -> AliasPromotionResult:
    """Point a conventional KB alias at an attested materialized collection."""
    if not collection_manager.collection_exists():
        raise RuntimeError(f"Collection '{collection_name}' does not exist")
    attestation = collection_manager.read_attestation()
    if attestation is None:
        raise RuntimeError(f"Collection '{collection_name}' has no attestation metadata")
    if attestation.kb_id != kb_id:
        raise RuntimeError(
            f"Collection '{collection_name}' belongs to '{attestation.kb_id}', not '{kb_id}'"
        )
    if attestation.collection_name != collection_name:
        raise RuntimeError(
            f"Collection attestation names '{attestation.collection_name}', not '{collection_name}'"
        )

    alias_name = qdrant_alias_name(kb_id=kb_id, alias=alias)
    collection_manager.update_alias(alias_name, collection_name)
    return AliasPromotionResult(
        alias_name=alias_name,
        collection_name=collection_name,
        manifest_id=attestation.manifest_id,
    )
