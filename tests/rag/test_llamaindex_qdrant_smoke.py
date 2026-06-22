"""Integration smoke test (Phase 1): LlamaIndex + Qdrant + project adapters.

Confirms a LlamaIndex `VectorStoreIndex` can be built and queried against a
Qdrant vector store in both dense and hybrid configurations, using
`ProjectEmbedding`/`ProjectSparseEncoder` instead of network-backed model
calls. Uses qdrant_client's in-memory mode so no running Qdrant server or
network access is required.
"""

from __future__ import annotations

import uuid

import pytest
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore as LIQdrantVectorStore
from qdrant_client import QdrantClient

from rag.indexing.llamaindex_embeddings import ProjectEmbedding, ProjectSparseEncoder


class _FakeEmbeddingClient:
    dimension = 4

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text) % 4), 0.0, 0.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 4), 0.0, 0.0, 0.0]


class _FakeSparseVector:
    def __init__(self, indices: list[int], values: list[float]) -> None:
        self.indices = indices
        self.values = values


class _FakeSparseEncoderClient:
    def encode_documents(self, texts: list[str]) -> list[_FakeSparseVector]:
        return [_FakeSparseVector(indices=[1, 2], values=[0.5, 0.5]) for _ in texts]


def _nodes() -> list[TextNode]:
    return [
        TextNode(id_=str(uuid.uuid4()), text="hello world"),
        TextNode(id_=str(uuid.uuid4()), text="goodbye world"),
    ]


@pytest.fixture()
def qdrant_client() -> QdrantClient:
    return QdrantClient(":memory:")


@pytest.fixture()
def embed_model() -> ProjectEmbedding:
    return ProjectEmbedding(embedding_client=_FakeEmbeddingClient())


def test_dense_vector_store_index_builds_and_retrieves(
    qdrant_client: QdrantClient,
    embed_model: ProjectEmbedding,
) -> None:
    vector_store = LIQdrantVectorStore(collection_name="dense_smoke", client=qdrant_client)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes=_nodes(),
        storage_context=storage_context,
        embed_model=embed_model,
    )

    collection_info = qdrant_client.get_collection("dense_smoke")
    assert collection_info.config.params.vectors.size == 4
    assert not collection_info.config.params.sparse_vectors

    retriever = index.as_retriever(similarity_top_k=1)
    results = retriever.retrieve("hello")
    assert len(results) == 1


def test_hybrid_vector_store_index_builds_with_dense_and_sparse_vectors(
    qdrant_client: QdrantClient,
    embed_model: ProjectEmbedding,
) -> None:
    sparse_encoder = ProjectSparseEncoder(_FakeSparseEncoderClient())
    vector_store = LIQdrantVectorStore(
        collection_name="hybrid_smoke",
        client=qdrant_client,
        enable_hybrid=True,
        sparse_doc_fn=sparse_encoder.sparse_doc_fn,
        sparse_query_fn=sparse_encoder.sparse_query_fn,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex(
        nodes=_nodes(),
        storage_context=storage_context,
        embed_model=embed_model,
    )

    collection_info = qdrant_client.get_collection("hybrid_smoke")
    assert collection_info.config.params.vectors
    assert collection_info.config.params.sparse_vectors
