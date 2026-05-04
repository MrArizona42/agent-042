"""Tests for SparseEncoderService — Phase 6."""

from __future__ import annotations

from unittest.mock import MagicMock

from qdrant_client.models import SparseVector


class TestSparseEncoderService:
    """SparseEncoderService batches requests and returns SparseVectors."""

    @staticmethod
    def _make_response(pairs: list[tuple[list, list]]) -> dict:
        return {
            "data": [
                {"indices": idxs, "values": vals, "index": i}
                for i, (idxs, vals) in enumerate(pairs)
            ],
            "model": "Qdrant/bm25",
        }

    def test_encode_documents_returns_sparse_vectors(self):
        from rag.sparse_encoder import SparseEncoderService

        svc = SparseEncoderService.__new__(SparseEncoderService)
        svc._client = MagicMock()
        svc._client.post.return_value.json.return_value = self._make_response(
            [([0, 5, 12], [0.3, 0.5, 0.8]), ([1, 7], [0.4, 0.6])]
        )

        result = svc.encode_documents(["hello world", "foo bar"])

        assert len(result) == 2
        assert isinstance(result[0], SparseVector)
        assert result[0].indices == [0, 5, 12]
        assert result[0].values == [0.3, 0.5, 0.8]

    def test_encode_documents_empty_input_returns_empty(self):
        from rag.sparse_encoder import SparseEncoderService

        svc = SparseEncoderService.__new__(SparseEncoderService)
        svc._client = MagicMock()

        result = svc.encode_documents([])

        assert result == []
        svc._client.post.assert_not_called()

    def test_encode_documents_batches_at_limit(self):
        from rag.sparse_encoder import SparseEncoderService

        batch_size = SparseEncoderService._ENCODE_BATCH
        svc = SparseEncoderService.__new__(SparseEncoderService)
        svc._client = MagicMock()
        svc._client.post.return_value.json.side_effect = [
            self._make_response([([0], [0.5])] * batch_size),
            self._make_response([([0], [0.5])]),
        ]

        result = svc.encode_documents(["text"] * (batch_size + 1))

        assert svc._client.post.call_count == 2
        assert len(result) == batch_size + 1

    def test_encode_query_returns_single_sparse_vector(self):
        from rag.sparse_encoder import SparseEncoderService

        svc = SparseEncoderService.__new__(SparseEncoderService)
        svc._client = MagicMock()
        svc._client.post.return_value.json.return_value = self._make_response(
            [([2, 9], [0.7, 0.9])]
        )

        result = svc.encode_query("what is gradient descent?")

        assert isinstance(result, SparseVector)
        assert result.indices == [2, 9]
        assert result.values == [0.7, 0.9]
        svc._client.post.assert_called_once_with(
            "/v1/sparse-embeddings",
            json={"input": ["what is gradient descent?"]},
        )
