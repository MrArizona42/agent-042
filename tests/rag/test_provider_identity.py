"""Tests for provider identity reporting and validation (declarative alias workflow, phase 1).

Covers `EmbeddingService`/`SparseEncoderService`/`CrossEncoderReranker` parsing
the `/v1/info` identity response, and the `validate_*_identity` functions
raising rather than silently substituting a mismatched catalog-declared
identity for the live provider's actual identity.
"""

from __future__ import annotations

import pytest


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeHttpxClient:
    def __init__(self, *, info: dict, **_kwargs):
        self._info = info

    def get(self, path: str):
        assert path in ("/v1/info", "/v1/dimension")
        return _FakeResponse(self._info)

    def post(self, path: str, json: dict):  # noqa: A002 - matches httpx.Client signature
        raise NotImplementedError

    def close(self) -> None:
        return None


class TestEmbeddingServiceIdentity:
    def test_init_captures_model_and_dimension_from_info_endpoint(self, monkeypatch):
        import rag.embeddings as embeddings_module

        monkeypatch.setattr(
            embeddings_module.httpx,
            "Client",
            lambda **kwargs: _FakeHttpxClient(
                info={"dense_model": "minilm", "dense_dimension": 384, "sparse_model": "bm25"},
                **kwargs,
            ),
        )

        client = embeddings_module.EmbeddingService()

        assert client.model == "minilm"
        assert client.dimension == 384

    def test_validate_dense_encoder_identity_passes_on_match(self, monkeypatch):
        import rag.embeddings as embeddings_module

        monkeypatch.setattr(
            embeddings_module.httpx,
            "Client",
            lambda **kwargs: _FakeHttpxClient(
                info={"dense_model": "minilm", "dense_dimension": 384, "sparse_model": "bm25"},
                **kwargs,
            ),
        )
        client = embeddings_module.EmbeddingService()

        embeddings_module.validate_dense_encoder_identity(
            client, expected_model="minilm", expected_dimension=384
        )

    def test_validate_dense_encoder_identity_raises_on_model_mismatch(self, monkeypatch):
        import rag.embeddings as embeddings_module

        monkeypatch.setattr(
            embeddings_module.httpx,
            "Client",
            lambda **kwargs: _FakeHttpxClient(
                info={"dense_model": "minilm", "dense_dimension": 384, "sparse_model": "bm25"},
                **kwargs,
            ),
        )
        client = embeddings_module.EmbeddingService()

        with pytest.raises(embeddings_module.EmbeddingIdentityMismatch, match="other-model"):
            embeddings_module.validate_dense_encoder_identity(
                client, expected_model="other-model", expected_dimension=384
            )

    def test_validate_dense_encoder_identity_raises_on_dimension_mismatch(self, monkeypatch):
        import rag.embeddings as embeddings_module

        monkeypatch.setattr(
            embeddings_module.httpx,
            "Client",
            lambda **kwargs: _FakeHttpxClient(
                info={"dense_model": "minilm", "dense_dimension": 384, "sparse_model": "bm25"},
                **kwargs,
            ),
        )
        client = embeddings_module.EmbeddingService()

        with pytest.raises(embeddings_module.EmbeddingIdentityMismatch, match="768"):
            embeddings_module.validate_dense_encoder_identity(
                client, expected_model="minilm", expected_dimension=768
            )


class TestSparseEncoderServiceIdentity:
    def test_init_captures_model_from_info_endpoint(self, monkeypatch):
        import rag.sparse_encoder as sparse_module

        monkeypatch.setattr(
            sparse_module.httpx,
            "Client",
            lambda **kwargs: _FakeHttpxClient(
                info={"dense_model": "minilm", "dense_dimension": 384, "sparse_model": "bm25"},
                **kwargs,
            ),
        )

        client = sparse_module.SparseEncoderService()

        assert client.model == "bm25"

    def test_validate_sparse_encoder_identity_passes_on_match(self, monkeypatch):
        import rag.sparse_encoder as sparse_module

        monkeypatch.setattr(
            sparse_module.httpx,
            "Client",
            lambda **kwargs: _FakeHttpxClient(
                info={"dense_model": "minilm", "dense_dimension": 384, "sparse_model": "bm25"},
                **kwargs,
            ),
        )
        client = sparse_module.SparseEncoderService()

        sparse_module.validate_sparse_encoder_identity(client, expected_model="bm25")

    def test_validate_sparse_encoder_identity_raises_on_mismatch(self, monkeypatch):
        import rag.sparse_encoder as sparse_module

        monkeypatch.setattr(
            sparse_module.httpx,
            "Client",
            lambda **kwargs: _FakeHttpxClient(
                info={"dense_model": "minilm", "dense_dimension": 384, "sparse_model": "bm25"},
                **kwargs,
            ),
        )
        client = sparse_module.SparseEncoderService()

        with pytest.raises(sparse_module.SparseEncoderIdentityMismatch, match="other-sparse"):
            sparse_module.validate_sparse_encoder_identity(client, expected_model="other-sparse")


class TestCrossEncoderRerankerIdentity:
    def test_init_captures_model_from_info_endpoint(self, monkeypatch):
        import rag.reranker as reranker_module

        monkeypatch.setattr(
            reranker_module.httpx,
            "Client",
            lambda **kwargs: _FakeHttpxClient(info={"model": "cross-encoder/x"}, **kwargs),
        )

        client = reranker_module.CrossEncoderReranker(reranker_url="http://reranker")

        assert client.model == "cross-encoder/x"

    def test_validate_reranker_identity_passes_on_match(self, monkeypatch):
        import rag.reranker as reranker_module

        monkeypatch.setattr(
            reranker_module.httpx,
            "Client",
            lambda **kwargs: _FakeHttpxClient(info={"model": "cross-encoder/x"}, **kwargs),
        )
        client = reranker_module.CrossEncoderReranker(reranker_url="http://reranker")

        reranker_module.validate_reranker_identity(client, expected_model="cross-encoder/x")

    def test_validate_reranker_identity_raises_on_mismatch(self, monkeypatch):
        import rag.reranker as reranker_module

        monkeypatch.setattr(
            reranker_module.httpx,
            "Client",
            lambda **kwargs: _FakeHttpxClient(info={"model": "cross-encoder/x"}, **kwargs),
        )
        client = reranker_module.CrossEncoderReranker(reranker_url="http://reranker")

        with pytest.raises(reranker_module.RerankerIdentityMismatch, match="cross-encoder/y"):
            reranker_module.validate_reranker_identity(client, expected_model="cross-encoder/y")

    def test_get_reranker_raises_before_any_rerank_call_on_mismatch(self, monkeypatch):
        import rag.reranker as reranker_module

        monkeypatch.setattr(
            reranker_module.httpx,
            "Client",
            lambda **kwargs: _FakeHttpxClient(info={"model": "cross-encoder/x"}, **kwargs),
        )

        with pytest.raises(reranker_module.RerankerIdentityMismatch):
            reranker_module.get_reranker("catalog-declared-model")

    def test_get_reranker_returns_client_when_identity_matches(self, monkeypatch):
        import rag.reranker as reranker_module

        monkeypatch.setattr(
            reranker_module.httpx,
            "Client",
            lambda **kwargs: _FakeHttpxClient(info={"model": "cross-encoder/x"}, **kwargs),
        )

        client = reranker_module.get_reranker("cross-encoder/x")

        assert isinstance(client, reranker_module.CrossEncoderReranker)
