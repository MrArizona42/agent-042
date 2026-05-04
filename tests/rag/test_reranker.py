"""Tests for CrossEncoderReranker — Phase 6."""

from __future__ import annotations

from unittest.mock import MagicMock

from rag.vector_store import Document


def _doc(content: str, score: float = 0.5) -> Document:
    return Document(content=content, metadata={}, score=score)


class TestCrossEncoderReranker:
    def test_rerank_replaces_scores_and_sorts_descending(self):
        from rag.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._client = MagicMock()
        reranker._client.post.return_value.json.return_value = {
            "scores": [0.2, 0.9, 0.4],
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        }

        docs = [_doc("a", 0.8), _doc("b", 0.7), _doc("c", 0.6)]
        result = reranker.rerank("query", docs, top_k=3)

        assert [d.content for d in result] == ["b", "c", "a"]
        assert result[0].score == 0.9
        assert result[1].score == 0.4
        assert result[2].score == 0.2

    def test_rerank_empty_docs_returns_empty_without_http_call(self):
        from rag.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._client = MagicMock()

        result = reranker.rerank("query", [], top_k=5)

        assert result == []
        reranker._client.post.assert_not_called()

    def test_rerank_posts_correct_payload(self):
        from rag.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._client = MagicMock()
        reranker._client.post.return_value.json.return_value = {
            "scores": [0.5],
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        }

        reranker.rerank("find me docs", [_doc("hello world")], top_k=1)

        reranker._client.post.assert_called_once_with(
            "/v1/rerank",
            json={"query": "find me docs", "passages": ["hello world"]},
        )

    def test_rerank_all_docs_receive_new_score(self):
        from rag.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._client = MagicMock()
        scores = [0.1, 0.8, 0.5]
        reranker._client.post.return_value.json.return_value = {
            "scores": scores,
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        }

        docs = [_doc(f"doc{i}", 0.9) for i in range(3)]
        result = reranker.rerank("q", docs, top_k=3)

        returned_scores = sorted([d.score for d in result], reverse=True)
        assert returned_scores == sorted(scores, reverse=True)
