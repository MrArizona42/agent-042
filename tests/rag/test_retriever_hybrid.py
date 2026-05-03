"""Tests for Retriever hybrid path and reranker integration — Phase 6."""

from __future__ import annotations

from unittest.mock import MagicMock

from qdrant_client.models import SparseVector

from rag.vector_store import Document


def _doc(content: str, score: float = 0.5) -> Document:
    return Document(content=content, metadata={}, score=score)


def _make_retriever(*, with_reranker: bool = False, reranker_multiplier: int = 1):
    """Return a (retriever, mock_vs, mock_sparse, mock_reranker) tuple."""
    from rag.retriever import Retriever

    mock_embedding_svc = MagicMock()
    mock_embedding_svc.embed_query.return_value = [0.1] * 8

    mock_vs = MagicMock()
    mock_vs.search.return_value = [_doc("doc1", 0.8), _doc("doc2", 0.6), _doc("doc3", 0.4)]

    mock_sparse = MagicMock()
    mock_sparse.encode_query.return_value = SparseVector(indices=[0, 1], values=[0.5, 0.5])

    mock_reranker = None
    if with_reranker:
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            _doc("doc1", 0.9),
            _doc("doc2", 0.7),
            _doc("doc3", 0.3),
        ]

    retriever = Retriever(
        embedding_service=mock_embedding_svc,
        vector_store=mock_vs,
        reranker=mock_reranker,
        sparse_encoder_service=mock_sparse,
        reranker_multiplier=reranker_multiplier,
    )
    return retriever, mock_vs, mock_sparse, mock_reranker


class TestRetrieverHybridPath:
    def test_hybrid_encodes_sparse_query_and_passes_to_search(self):
        retriever, mock_vs, mock_sparse, _ = _make_retriever()

        retriever.retrieve(query="test query", top_k=2, score_threshold=0.0, strategy="hybrid")

        mock_sparse.encode_query.assert_called_once_with("test query")
        kwargs = mock_vs.search.call_args.kwargs
        assert kwargs["strategy"] == "hybrid"
        assert isinstance(kwargs["sparse_query"], SparseVector)

    def test_dense_path_does_not_encode_sparse_query(self):
        retriever, mock_vs, mock_sparse, _ = _make_retriever()

        retriever.retrieve(query="test query", top_k=2, score_threshold=0.0, strategy="dense")

        mock_sparse.encode_query.assert_not_called()
        kwargs = mock_vs.search.call_args.kwargs
        assert "strategy" not in kwargs
        assert "sparse_query" not in kwargs

    def test_dense_path_truncates_to_top_k(self):
        retriever, mock_vs, _, _ = _make_retriever()
        mock_vs.search.return_value = [_doc(f"d{i}", 0.9 - i * 0.1) for i in range(5)]

        result = retriever.retrieve(query="q", top_k=2, score_threshold=0.0)

        assert len(result) == 2

    def test_empty_query_returns_empty_without_search(self):
        retriever, mock_vs, mock_sparse, _ = _make_retriever()

        result = retriever.retrieve(query="  ", top_k=5, score_threshold=0.5)

        assert result == []
        mock_vs.search.assert_not_called()
        mock_sparse.encode_query.assert_not_called()


class TestRetrieverRerankerIntegration:
    def test_reranker_expands_fetch_k_by_multiplier(self):
        retriever, mock_vs, _, mock_reranker = _make_retriever(
            with_reranker=True, reranker_multiplier=4
        )
        mock_vs.search.return_value = [_doc(f"d{i}") for i in range(20)]
        mock_reranker.rerank.return_value = [_doc(f"d{i}", 0.9 - i * 0.05) for i in range(20)]

        retriever.retrieve(query="q", top_k=5, score_threshold=0.0)

        kwargs = mock_vs.search.call_args.kwargs
        assert kwargs["top_k"] == 20  # 5 * 4

    def test_reranker_first_stage_passes_none_score_threshold(self):
        retriever, mock_vs, _, mock_reranker = _make_retriever(
            with_reranker=True, reranker_multiplier=1
        )
        mock_vs.search.return_value = []
        mock_reranker.rerank.return_value = []

        retriever.retrieve(query="q", top_k=3, score_threshold=0.9)

        kwargs = mock_vs.search.call_args.kwargs
        assert kwargs["score_threshold"] is None

    def test_reranker_score_threshold_applied_after_rerank(self):
        retriever, mock_vs, _, mock_reranker = _make_retriever(
            with_reranker=True, reranker_multiplier=1
        )
        mock_vs.search.return_value = [_doc("a"), _doc("b")]
        mock_reranker.rerank.return_value = [_doc("a", 0.8), _doc("b", 0.2)]

        result = retriever.retrieve(query="q", top_k=5, score_threshold=0.5)

        assert len(result) == 1
        assert result[0].content == "a"

    def test_reranker_result_truncated_to_top_k(self):
        retriever, mock_vs, _, mock_reranker = _make_retriever(
            with_reranker=True, reranker_multiplier=3
        )
        mock_vs.search.return_value = [_doc(f"d{i}") for i in range(9)]
        mock_reranker.rerank.return_value = [_doc(f"d{i}", 0.9 - i * 0.1) for i in range(9)]

        result = retriever.retrieve(query="q", top_k=3, score_threshold=0.0)

        assert len(result) == 3

    def test_no_reranker_passes_score_threshold_to_search(self):
        retriever, mock_vs, _, _ = _make_retriever()

        retriever.retrieve(query="q", top_k=5, score_threshold=0.7)

        kwargs = mock_vs.search.call_args.kwargs
        assert kwargs["score_threshold"] == 0.7
