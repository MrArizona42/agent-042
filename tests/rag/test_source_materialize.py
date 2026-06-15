from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client.models import SparseVector

from rag.domain import Chunk, RetrievalCapability
from rag.domain.manifests import attestation_from_payload, read_index_manifest
from rag.sources.bundles import SourceChunkBundle
from rag.sources.materialize import (
    collection_name_for_build,
    materialize_kb_collection,
    promote_materialized_alias,
    qdrant_alias_name,
    retrieval_capability_for_strategy,
    validate_strategy_supported,
)


class _EmbeddingClient:
    dimension = 3

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), 0.0, 1.0] for index, _ in enumerate(texts)]


class _SparseClient:
    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector(indices=[index + 1], values=[1.0]) for index, _ in enumerate(texts)]


class _VectorStore:
    def __init__(self, collection_name: str = "rag__pytorch_reference__test"):
        self.collection_name = collection_name
        self.created: dict[str, object] | None = None
        self.added: list[dict[str, object]] = []
        self.meta: dict | None = None
        self.meta_dimension: int | None = None
        self.aliases: dict[str, str] = {}
        self.exists = True

    def create_collection(
        self,
        dimension: int,
        retrieval_capability: str = "dense",
        force_recreate: bool = False,
    ) -> None:
        self.created = {
            "dimension": dimension,
            "retrieval_capability": retrieval_capability,
            "force_recreate": force_recreate,
        }

    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        sparse_vectors: list[SparseVector] | None = None,
        upsert_batch_size: int = 500,
    ) -> None:
        self.added.append(
            {
                "documents": documents,
                "embeddings": embeddings,
                "metadatas": metadatas,
                "ids": ids,
                "sparse_vectors": sparse_vectors,
                "upsert_batch_size": upsert_batch_size,
            }
        )

    def write_meta(self, payload: dict, dimension: int) -> None:
        self.meta_dimension = dimension
        self.meta = payload

    def read_meta(self) -> dict | None:
        return self.meta

    def collection_exists(self) -> bool:
        return self.exists

    def update_alias(self, alias_name: str, collection_name: str) -> None:
        self.aliases[alias_name] = collection_name


def _bundle() -> SourceChunkBundle:
    chunks = [
        Chunk(
            id="html:tensors:chunk:0000",
            document_id="html:tensors",
            source_document_id="html:tensors",
            text="Tensor text.",
            section_title="Overview",
            ordinal=0,
            token_count=2,
            metadata={"kb_id": "pytorch_reference", "source_type": "html_docs"},
        ),
        Chunk(
            id="html:torch:chunk:0000",
            document_id="html:torch",
            source_document_id="html:torch",
            text="Torch text.",
            section_title="Overview",
            ordinal=1,
            token_count=2,
            metadata={"kb_id": "pytorch_reference", "source_type": "html_docs"},
        ),
    ]
    return SourceChunkBundle(
        kb_id="pytorch_reference",
        source_instance_id="docs",
        source_types=["html_docs"],
        chunk_artifact_paths=["chunks/html_tensors.json"],
        chunk_artifact_checksums={"chunks/html_tensors.json": "sha256:abc"},
        chunks=chunks,
        document_count=2,
        chunk_count=2,
    )


def test_strategy_capability_rules_are_explicit() -> None:
    assert retrieval_capability_for_strategy("dense") == "dense"
    assert retrieval_capability_for_strategy("hybrid") == "hybrid"
    validate_strategy_supported(retrieval_strategy="dense", retrieval_capability="dense")
    validate_strategy_supported(retrieval_strategy="dense", retrieval_capability="hybrid")
    validate_strategy_supported(retrieval_strategy="hybrid", retrieval_capability="hybrid")

    with pytest.raises(ValueError, match="not supported"):
        validate_strategy_supported(retrieval_strategy="hybrid", retrieval_capability="dense")


def test_materialize_dense_collection_writes_manifest_and_attestation(tmp_path: Path) -> None:
    store = _VectorStore()

    result = materialize_kb_collection(
        kb_id="pytorch_reference",
        collection_name=store.collection_name,
        bundles=[_bundle()],
        vector_store=store,
        embedding_client=_EmbeddingClient(),
        embedding_model="test-embedding",
        retrieval_capability="dense",
        rag_data_root=tmp_path,
        target_alias="challenger",
        qdrant_upsert_batch_size=64,
        build_config_ref="catalog.toml",
    )
    manifest = read_index_manifest(result.manifest_path)
    attestation = attestation_from_payload(store.meta or {})

    assert store.created == {
        "dimension": 3,
        "retrieval_capability": "dense",
        "force_recreate": False,
    }
    assert store.added[0]["upsert_batch_size"] == 64
    assert store.added[0]["sparse_vectors"] is None
    assert manifest.embedding_model == "test-embedding"
    assert manifest.retrieval_capability == RetrievalCapability.DENSE
    assert manifest.chunk_count == 2
    assert manifest.manifest_id == attestation.manifest_id
    assert result.summary.sparse_enabled is False


def test_materialize_hybrid_requires_and_writes_sparse_vectors(tmp_path: Path) -> None:
    store = _VectorStore()

    with pytest.raises(ValueError, match="sparse_encoder_client"):
        materialize_kb_collection(
            kb_id="pytorch_reference",
            collection_name=store.collection_name,
            bundles=[_bundle()],
            vector_store=store,
            embedding_client=_EmbeddingClient(),
            embedding_model="test-embedding",
            retrieval_capability="hybrid",
            rag_data_root=tmp_path,
            sparse_encoder_model="Qdrant/bm25",
        )

    result = materialize_kb_collection(
        kb_id="pytorch_reference",
        collection_name=store.collection_name,
        bundles=[_bundle()],
        vector_store=store,
        embedding_client=_EmbeddingClient(),
        embedding_model="test-embedding",
        retrieval_capability="hybrid",
        rag_data_root=tmp_path,
        sparse_encoder_model="Qdrant/bm25",
        sparse_encoder_client=_SparseClient(),
    )

    assert store.created["retrieval_capability"] == "hybrid"
    assert store.added[-1]["sparse_vectors"] is not None
    assert result.summary.sparse_enabled is True
    assert result.manifest.sparse_encoder == "Qdrant/bm25"


def test_promote_materialized_alias_validates_attestation(tmp_path: Path) -> None:
    store = _VectorStore()
    materialize_kb_collection(
        kb_id="pytorch_reference",
        collection_name=store.collection_name,
        bundles=[_bundle()],
        vector_store=store,
        embedding_client=_EmbeddingClient(),
        embedding_model="test-embedding",
        retrieval_capability="dense",
        rag_data_root=tmp_path,
    )

    result = promote_materialized_alias(
        kb_id="pytorch_reference",
        alias="challenger",
        collection_name=store.collection_name,
        vector_store=store,
    )

    assert result.alias_name == qdrant_alias_name(
        kb_id="pytorch_reference",
        alias="challenger",
    )
    assert store.aliases[result.alias_name] == store.collection_name
    assert result.manifest_id.startswith("sha256:")


def test_collection_name_for_build_is_timestamped() -> None:
    name = collection_name_for_build(
        kb_id="pytorch_reference",
    )

    assert name.startswith("rag__pytorch_reference__")
    assert "__challenger__" not in name
    assert "__champion__" not in name
