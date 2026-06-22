"""Materialize source nodes as immutable LlamaIndex-backed releases."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.models import SparseVector

from app_config.catalog.schema import AliasBuildConfig
from rag.contracts import ReleaseAttestation
from rag.contracts.manifests import (
    release_manifest_path,
    release_to_attestation,
    with_release_manifest_id,
    write_release_manifest,
)
from rag.control_plane.models import RagRelease
from rag.indexing.llamaindex_embeddings import ProjectEmbedding, ProjectSparseEncoder
from rag.sources.bundles import SourceNodeBundle

RetrievalStrategy = Literal["dense", "hybrid", "sparse"]
SourceRetrievalCapability = Literal["dense", "hybrid"]


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

    def write_release_attestation(self, attestation: ReleaseAttestation) -> None:
        """Write schema-version-2 release attestation metadata."""
        ...

    def read_release_attestation(self) -> ReleaseAttestation | None:
        """Read schema-version-2 release attestation metadata."""
        ...

    def collection_exists(self) -> bool:
        """Return whether the collection exists."""
        ...

    def update_alias(self, alias_name: str, collection_name: str) -> None:
        """Create or update an alias."""
        ...


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


def _all_nodes(bundles: list[SourceNodeBundle]):
    for bundle in bundles:
        yield from bundle.nodes


def _validate_bundles(
    *,
    kb_id: str,
    collection_name: str,
    bundles: list[SourceNodeBundle],
    collection_manager: CollectionManager,
    retrieval_capability: SourceRetrievalCapability,
    sparse_encoder_client: SparseEmbeddingClient | None,
    sparse_encoder_model: str | None,
) -> list:
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
    return nodes


def _write_nodes_to_qdrant(
    *,
    nodes: list,
    collection_manager: CollectionManager,
    embedding_client: EmbeddingClient,
    embedding_model: str,
    retrieval_capability: SourceRetrievalCapability,
    sparse_encoder_client: SparseEmbeddingClient | None,
    qdrant_upsert_batch_size: int,
    force_recreate: bool,
) -> None:
    """Create the physical collection and index *nodes* into it via LlamaIndex."""
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


def materialize_release_collection(
    *,
    kb_id: str,
    release_id: str,
    collection_name: str,
    release_fingerprint: str,
    catalog_digest: str,
    build_config_digest: str,
    source_declaration_digest: str,
    source_snapshot_id: str,
    build_config: AliasBuildConfig,
    bundles: list[SourceNodeBundle],
    collection_manager: CollectionManager,
    embedding_client: EmbeddingClient,
    rag_data_root: Path | str,
    source_adapter_versions: dict[str, str],
    source_manifest_digests: dict[str, str],
    sparse_encoder_client: SparseEmbeddingClient | None = None,
    qdrant_upsert_batch_size: int = 128,
    force_recreate: bool = False,
    created_at: datetime | None = None,
) -> RagRelease:
    """Materialize an immutable, content-identified release through LlamaIndex.

    Release identity is computed by the caller from the resolved alias build
    configuration before this function runs, since it determines the physical
    collection name. This function writes the Qdrant collection, immutable
    release manifest, and schema-version-2 attestation.
    """
    retrieval_capability: SourceRetrievalCapability = (
        "hybrid" if build_config.sparse_encoder is not None else "dense"
    )
    nodes = _validate_bundles(
        kb_id=kb_id,
        collection_name=collection_name,
        bundles=bundles,
        collection_manager=collection_manager,
        retrieval_capability=retrieval_capability,
        sparse_encoder_client=sparse_encoder_client,
        sparse_encoder_model=(
            build_config.sparse_encoder.model if build_config.sparse_encoder else None
        ),
    )
    _write_nodes_to_qdrant(
        nodes=nodes,
        collection_manager=collection_manager,
        embedding_client=embedding_client,
        embedding_model=build_config.dense_encoder.model,
        retrieval_capability=retrieval_capability,
        sparse_encoder_client=sparse_encoder_client,
        qdrant_upsert_batch_size=qdrant_upsert_batch_size,
        force_recreate=force_recreate,
    )

    release = RagRelease(
        id=release_id,
        kb_id=kb_id,
        collection_name=collection_name,
        manifest_id="",
        release_fingerprint=release_fingerprint,
        catalog_digest=catalog_digest,
        build_config_digest=build_config_digest,
        source_declaration_digest=source_declaration_digest,
        source_snapshot_id=source_snapshot_id,
        build_config=build_config,
        source_manifest_digests=source_manifest_digests,
        source_adapter_versions=source_adapter_versions,
        document_count=sum(bundle.document_count for bundle in bundles),
        chunk_count=len(nodes),
        created_at=created_at or datetime.now(tz=UTC),
    )
    release = with_release_manifest_id(release)
    path = release_manifest_path(rag_data_root=rag_data_root, kb_id=kb_id, release_id=release_id)
    release = write_release_manifest(path, release)
    collection_manager.write_release_attestation(release_to_attestation(release))
    return release
