"""Integration tests for QdrantVectorStore hybrid path — Phase 6.

Uses qdrant_client's in-memory client so no running Qdrant server is needed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from qdrant_client import QdrantClient as RealQdrantClient
from qdrant_client.models import SparseVector


@pytest.fixture()
def dense_store():
    from rag.vector_store import QdrantVectorStore

    in_memory = RealQdrantClient(":memory:")
    with patch("rag.vector_store.QdrantClient", return_value=in_memory):
        store = QdrantVectorStore(host="localhost", port=6333, collection_name="test_dense")
    return store


@pytest.fixture()
def hybrid_store():
    from rag.vector_store import QdrantVectorStore

    in_memory = RealQdrantClient(":memory:")
    with patch("rag.vector_store.QdrantClient", return_value=in_memory):
        store = QdrantVectorStore(host="localhost", port=6333, collection_name="test_hybrid")
    return store


class TestCreateCollection:
    def test_dense_collection_has_named_dense_vector(self, dense_store):
        dense_store.create_collection(dimension=4, retrieval_capability="dense")

        info = dense_store.client.get_collection("test_dense")
        assert "dense" in info.config.params.vectors
        assert not info.config.params.sparse_vectors

    def test_hybrid_collection_has_both_dense_and_sparse_vectors(self, hybrid_store):
        hybrid_store.create_collection(dimension=4, retrieval_capability="hybrid")

        info = hybrid_store.client.get_collection("test_hybrid")
        assert "dense" in info.config.params.vectors
        assert "sparse" in info.config.params.sparse_vectors


class TestAddDocuments:
    def test_add_dense_documents(self, dense_store):
        dense_store.create_collection(dimension=4, retrieval_capability="dense")
        dense_store.add_documents(
            documents=["neural networks"],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            metadatas=[{"source": "test"}],
        )

        info = dense_store.client.get_collection("test_dense")
        assert info.points_count == 1

    def test_add_hybrid_documents_with_sparse_vectors(self, hybrid_store):
        hybrid_store.create_collection(dimension=4, retrieval_capability="hybrid")
        hybrid_store.add_documents(
            documents=["machine learning"],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            metadatas=[{"source": "test"}],
            sparse_vectors=[SparseVector(indices=[0, 3], values=[0.6, 0.4])],
        )

        info = hybrid_store.client.get_collection("test_hybrid")
        assert info.points_count == 1


class TestSearch:
    def test_dense_search_on_named_vector_collection(self, dense_store):
        dense_store.create_collection(dimension=4, retrieval_capability="dense")
        dense_store.add_documents(
            documents=["deep learning research"],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            metadatas=[{"source": "arxiv"}],
        )

        results = dense_store.search(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            top_k=5,
            score_threshold=0.0,
        )

        assert len(results) == 1
        assert results[0].content == "deep learning research"
        assert results[0].metadata["source"] == "arxiv"

    def test_dense_search_on_hybrid_collection(self, hybrid_store):
        hybrid_store.create_collection(dimension=4, retrieval_capability="hybrid")
        hybrid_store.add_documents(
            documents=["transformer architecture"],
            embeddings=[[0.5, 0.5, 0.5, 0.5]],
            metadatas=[{"source": "paper"}],
            sparse_vectors=[SparseVector(indices=[1, 4], values=[0.8, 0.2])],
        )

        results = hybrid_store.search(
            query_embedding=[0.5, 0.5, 0.5, 0.5],
            top_k=5,
            score_threshold=0.0,
        )

        assert len(results) == 1
        assert results[0].content == "transformer architecture"

    def test_dense_search_with_none_score_threshold(self, dense_store):
        dense_store.create_collection(dimension=4, retrieval_capability="dense")
        dense_store.add_documents(
            documents=["attention mechanism"],
            embeddings=[[0.1, 0.2, 0.3, 0.4]],
            metadatas=[{"source": "test"}],
        )

        results = dense_store.search(
            query_embedding=[0.1, 0.2, 0.3, 0.4],
            top_k=5,
            score_threshold=None,
        )

        assert len(results) == 1

    def test_hybrid_search_dbsf_fusion(self, hybrid_store):
        hybrid_store.create_collection(dimension=4, retrieval_capability="hybrid")
        hybrid_store.add_documents(
            documents=["gradient descent optimisation"],
            embeddings=[[0.3, 0.4, 0.5, 0.6]],
            metadatas=[{"source": "book"}],
            sparse_vectors=[SparseVector(indices=[2, 7], values=[0.7, 0.3])],
        )

        results = hybrid_store.search(
            query_embedding=[0.3, 0.4, 0.5, 0.6],
            top_k=5,
            score_threshold=0.0,
            strategy="hybrid",
            sparse_query=SparseVector(indices=[2, 7], values=[0.7, 0.3]),
        )

        assert len(results) == 1
        assert results[0].content == "gradient descent optimisation"
