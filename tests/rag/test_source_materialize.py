from __future__ import annotations

from pathlib import Path

import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from rag.contracts import RetrievalCapability
from rag.contracts.manifests import read_index_manifest
from rag.contracts.metadata import node_id_for_chunk
from rag.indexing.llamaindex_embeddings import ProjectEmbedding
from rag.indexing.llamaindex_qdrant import QdrantCollectionManager
from rag.indexing.materialize import (
    collection_name_for_build,
    materialize_kb_collection_llamaindex,
    promote_materialized_alias,
    qdrant_alias_name,
    retrieval_capability_for_strategy,
    validate_strategy_supported,
)
from rag.sources.bundles import SourceNodeBundle


class _EmbeddingClient:
    dimension = 3

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [1.0, 0.0, 0.0] if "tensor" in text.lower() else [0.0, 1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class _SparseClient:
    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        return [
            SparseVector(
                indices=[1] if "tensor" in text.lower() else [2],
                values=[1.0],
            )
            for text in texts
        ]


def _node(*, document_id: str, text: str, ordinal: int) -> TextNode:
    chunk_id = f"{document_id}:chunk:{ordinal:04d}"
    metadata = {
        "kb_id": "pytorch_reference",
        "source_instance_id": "pytorch_reference.docs",
        "source_document_id": document_id,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "title": document_id,
        "source_uri": f"https://docs.test/{document_id}",
        "section_title": "Overview",
        "section_ordinal": 0,
        "section_level": 1,
        "ordinal": ordinal,
        "token_count": len(text.split()),
        "adapter_id": "generic.http_html",
        "adapter_version": "1",
    }
    return TextNode(
        id_=node_id_for_chunk(chunk_id),
        text=text,
        metadata=metadata,
        excluded_embed_metadata_keys=list(metadata),
    )


def _bundle() -> SourceNodeBundle:
    nodes = [
        _node(document_id="docs:tensors", text="Tensor text.", ordinal=0),
        _node(document_id="docs:torch", text="Torch text.", ordinal=0),
    ]
    return SourceNodeBundle(
        kb_id="pytorch_reference",
        source_instance_id="pytorch_reference.docs",
        node_artifact_paths=["chunks/docs_tensors.json"],
        node_artifact_checksums={"chunks/docs_tensors.json": "sha256:abc"},
        nodes=nodes,
        document_count=2,
        node_count=2,
    )


@pytest.fixture()
def qdrant_client() -> QdrantClient:
    return QdrantClient(":memory:")


def _manager(client: QdrantClient) -> QdrantCollectionManager:
    return QdrantCollectionManager(
        client=client,
        collection_name="rag__pytorch_reference__test",
    )


def test_strategy_capability_rules_are_explicit() -> None:
    assert retrieval_capability_for_strategy("dense") == "dense"
    assert retrieval_capability_for_strategy("hybrid") == "hybrid"
    assert retrieval_capability_for_strategy("sparse") == "hybrid"
    validate_strategy_supported(retrieval_strategy="dense", retrieval_capability="dense")
    validate_strategy_supported(retrieval_strategy="dense", retrieval_capability="hybrid")
    validate_strategy_supported(retrieval_strategy="hybrid", retrieval_capability="hybrid")
    validate_strategy_supported(retrieval_strategy="sparse", retrieval_capability="hybrid")

    with pytest.raises(ValueError, match="not supported"):
        validate_strategy_supported(retrieval_strategy="hybrid", retrieval_capability="dense")
    with pytest.raises(ValueError, match="not supported"):
        validate_strategy_supported(retrieval_strategy="sparse", retrieval_capability="dense")


def test_dense_materialization_retrieves_and_writes_collection_attestation(
    tmp_path: Path,
    qdrant_client: QdrantClient,
) -> None:
    manager = _manager(qdrant_client)
    embedding_client = _EmbeddingClient()
    result = materialize_kb_collection_llamaindex(
        kb_id="pytorch_reference",
        collection_name=manager.collection_name,
        bundles=[_bundle()],
        collection_manager=manager,
        embedding_client=embedding_client,
        embedding_model="test-embedding",
        retrieval_capability="dense",
        rag_data_root=tmp_path,
        target_alias="challenger",
        qdrant_upsert_batch_size=1,
        build_config_ref="catalog.toml",
        build_config_digest="sha256:catalog",
        build_profile_digest="sha256:profile",
    )

    manifest = read_index_manifest(result.manifest_path)
    attestation = manager.read_attestation()
    assert attestation is not None
    assert manifest.manifest_id == attestation.manifest_id
    assert manifest.retrieval_capability == RetrievalCapability.DENSE
    assert manifest.chunk_count == 2
    assert result.summary.sparse_enabled is False

    info = qdrant_client.get_collection(manager.collection_name)
    assert info.config.metadata == {
        "attestation": attestation.model_dump(mode="json", exclude_none=True)
    }
    points, _ = qdrant_client.scroll(manager.collection_name, limit=10, with_payload=True)
    assert len(points) == 2
    assert all(point.payload and point.payload.get("type") != "collection_meta" for point in points)

    vector_store = manager.vector_store(
        vector_size=embedding_client.dimension,
        batch_size=1,
        enable_hybrid=False,
        sparse_encoder=None,
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=ProjectEmbedding(embedding_client=embedding_client),
    )
    retrieved = index.as_retriever(similarity_top_k=1).retrieve("tensor")
    assert retrieved[0].node.metadata["document_id"] == "docs:tensors"


def test_hybrid_materialization_builds_named_vectors_and_retrieves(
    tmp_path: Path,
    qdrant_client: QdrantClient,
) -> None:
    manager = _manager(qdrant_client)
    embedding_client = _EmbeddingClient()
    sparse_client = _SparseClient()

    result = materialize_kb_collection_llamaindex(
        kb_id="pytorch_reference",
        collection_name=manager.collection_name,
        bundles=[_bundle()],
        collection_manager=manager,
        embedding_client=embedding_client,
        embedding_model="test-embedding",
        retrieval_capability="hybrid",
        rag_data_root=tmp_path,
        sparse_encoder_model="Qdrant/bm25",
        sparse_encoder_client=sparse_client,
    )

    info = qdrant_client.get_collection(manager.collection_name)
    assert set(info.config.params.vectors) == {"dense"}
    assert set(info.config.params.sparse_vectors or {}) == {"sparse"}
    assert result.manifest.sparse_encoder == "Qdrant/bm25"
    assert result.summary.sparse_enabled is True

    from rag.indexing.llamaindex_embeddings import ProjectSparseEncoder

    vector_store = manager.vector_store(
        vector_size=embedding_client.dimension,
        batch_size=2,
        enable_hybrid=True,
        sparse_encoder=ProjectSparseEncoder(sparse_client),
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=ProjectEmbedding(embedding_client=embedding_client),
    )
    retrieved = index.as_retriever(
        similarity_top_k=1,
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
    ).retrieve("tensor")
    assert retrieved[0].node.metadata["document_id"] == "docs:tensors"


def test_materialization_requires_clean_collection_unless_forced(
    tmp_path: Path,
    qdrant_client: QdrantClient,
) -> None:
    manager = _manager(qdrant_client)
    kwargs = {
        "kb_id": "pytorch_reference",
        "collection_name": manager.collection_name,
        "bundles": [_bundle()],
        "collection_manager": manager,
        "embedding_client": _EmbeddingClient(),
        "embedding_model": "test-embedding",
        "retrieval_capability": "dense",
        "rag_data_root": tmp_path,
    }
    materialize_kb_collection_llamaindex(**kwargs)

    with pytest.raises(RuntimeError, match="already exists"):
        materialize_kb_collection_llamaindex(**kwargs)

    materialize_kb_collection_llamaindex(**kwargs, force_recreate=True)
    assert qdrant_client.get_collection(manager.collection_name).points_count == 2


def test_promote_alias_validates_collection_metadata(
    tmp_path: Path,
    qdrant_client: QdrantClient,
) -> None:
    manager = _manager(qdrant_client)
    materialize_kb_collection_llamaindex(
        kb_id="pytorch_reference",
        collection_name=manager.collection_name,
        bundles=[_bundle()],
        collection_manager=manager,
        embedding_client=_EmbeddingClient(),
        embedding_model="test-embedding",
        retrieval_capability="dense",
        rag_data_root=tmp_path,
    )

    result = promote_materialized_alias(
        kb_id="pytorch_reference",
        alias="challenger",
        collection_name=manager.collection_name,
        collection_manager=manager,
    )

    assert result.alias_name == qdrant_alias_name(
        kb_id="pytorch_reference",
        alias="challenger",
    )
    aliases = {
        alias.alias_name: alias.collection_name for alias in qdrant_client.get_aliases().aliases
    }
    assert aliases[result.alias_name] == manager.collection_name


def test_collection_name_for_build_is_timestamped() -> None:
    name = collection_name_for_build(kb_id="pytorch_reference")
    assert name.startswith("rag__pytorch_reference__")
    assert "__challenger__" not in name
