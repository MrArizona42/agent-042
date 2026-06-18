"""Unit tests for LlamaIndex embedding/sparse-encoder adapters (Phase 1).

Uses fake clients so no network call ever happens.
"""

from __future__ import annotations

from rag.indexing.llamaindex_embeddings import ProjectEmbedding, ProjectSparseEncoder


class _FakeSparseVector:
    def __init__(self, indices: list[int], values: list[float]) -> None:
        self.indices = indices
        self.values = values


class _FakeEmbeddingClient:
    dimension = 3

    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(texts)
        return [[float(len(text))] * 3 for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [float(len(text))] * 3


class _FakeSparseEncoderClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode_documents(self, texts: list[str]) -> list[_FakeSparseVector]:
        self.calls.append(texts)
        return [_FakeSparseVector(indices=[i], values=[1.0]) for i, _ in enumerate(texts)]


class TestProjectEmbedding:
    def test_class_name(self) -> None:
        assert ProjectEmbedding.class_name() == "ProjectEmbedding"

    def test_get_query_embedding_delegates_to_client(self) -> None:
        client = _FakeEmbeddingClient()
        embedding = ProjectEmbedding(embedding_client=client)

        vector = embedding.get_query_embedding("hello world")

        assert vector == [11.0, 11.0, 11.0]
        assert client.query_calls == ["hello world"]

    def test_get_text_embedding_uses_single_item_batch(self) -> None:
        client = _FakeEmbeddingClient()
        embedding = ProjectEmbedding(embedding_client=client)

        vector = embedding.get_text_embedding("abcde")

        assert vector == [5.0, 5.0, 5.0]
        assert client.document_calls == [["abcde"]]

    def test_get_text_embedding_batch_uses_one_request(self) -> None:
        client = _FakeEmbeddingClient()
        embedding = ProjectEmbedding(embedding_client=client)

        vectors = embedding.get_text_embedding_batch(["a", "bb", "ccc"])

        assert vectors == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
        assert client.document_calls == [["a", "bb", "ccc"]]


class TestProjectSparseEncoder:
    def test_sparse_doc_fn_returns_indices_and_values(self) -> None:
        client = _FakeSparseEncoderClient()
        encoder = ProjectSparseEncoder(client)

        indices, values = encoder.sparse_doc_fn(["a", "b"])

        assert indices == [[0], [1]]
        assert values == [[1.0], [1.0]]
        assert client.calls == [["a", "b"]]

    def test_sparse_query_fn_matches_doc_fn_behavior(self) -> None:
        client = _FakeSparseEncoderClient()
        encoder = ProjectSparseEncoder(client)

        indices, values = encoder.sparse_query_fn(["query text"])

        assert indices == [[0]]
        assert values == [[1.0]]
        assert client.calls == [["query text"]]
