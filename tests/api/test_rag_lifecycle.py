"""Tests for RAG alias-based lifecycle features.

Tests the knowledge-base config loader, RAGSource schema,
ChatCompletionRequest with rag_sources, metadata exclusion filter,
admin endpoint, and error handling for missing KB/aliases.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

# Ensure settings don't bleed between tests
os.environ.setdefault("GATEWAY_RAG_ENABLED", "false")


@pytest.fixture(autouse=True)
def _reset_kb_registry():
    """Reset the KB registry singleton between tests."""
    import shared.config as cfg

    cfg._KB_REGISTRY = None
    # Also reset the lazy proxy
    cfg.KNOWLEDGE_BASES._loaded = False
    cfg.KNOWLEDGE_BASES.clear()
    yield
    cfg._KB_REGISTRY = None
    cfg.KNOWLEDGE_BASES._loaded = False
    cfg.KNOWLEDGE_BASES.clear()


@pytest.fixture()
def kb_json_file(tmp_path: Path):
    """Create a temporary knowledge_bases.json."""
    data = [
        {
            "knowledge_base": "arxiv",
            "aliases": ["champion", "challenger"],
            "update_strategy": "incremental",
            "label": "ArXiv papers",
            "description": "ML papers",
            "chunking_strategy": "fixed_token",
            "chunk_size": 512,
            "chunk_overlap": 50,
        },
        {
            "knowledge_base": "pytorch_docs",
            "aliases": ["champion"],
            "update_strategy": "replace",
            "label": "PyTorch docs",
            "description": "Coding docs",
            "chunking_strategy": "code",
            "chunk_size": 1000,
            "chunk_overlap": 100,
        },
    ]
    path = tmp_path / "knowledge_bases.json"
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# Config loading tests
# ---------------------------------------------------------------------------


class TestKnowledgeBaseConfig:
    """Tests for the knowledge-base config loader."""

    def test_load_from_json(self, kb_json_file: Path):
        from shared.config import _load_knowledge_bases

        registry = _load_knowledge_bases(kb_json_file)
        assert "arxiv" in registry
        assert "pytorch_docs" in registry
        assert registry["arxiv"].update_strategy == "incremental"
        assert registry["pytorch_docs"].update_strategy == "replace"
        assert "champion" in registry["arxiv"].aliases
        assert "challenger" in registry["arxiv"].aliases
        assert registry["pytorch_docs"].label == "PyTorch docs"

    def test_load_missing_file_returns_empty(self, tmp_path: Path):
        from shared.config import _load_knowledge_bases

        registry = _load_knowledge_bases(tmp_path / "nonexistent.json")
        assert registry == {}

    def test_backward_compat_proxy(self, kb_json_file: Path):
        """KNOWLEDGE_BASES proxy dict returns the same keys."""
        import shared.config as cfg
        from shared.config import KNOWLEDGE_BASES, _load_knowledge_bases

        # Load using explicit path first
        cfg._KB_REGISTRY = _load_knowledge_bases(kb_json_file)
        assert "arxiv" in KNOWLEDGE_BASES
        assert "pytorch_docs" in KNOWLEDGE_BASES
        info = KNOWLEDGE_BASES["arxiv"]
        assert info["label"] == "ArXiv papers"
        assert "aliases" in info

    def test_get_knowledge_bases_caching(self, kb_json_file: Path):
        import shared.config as cfg
        from shared.config import _load_knowledge_bases, get_knowledge_bases

        cfg._KB_REGISTRY = _load_knowledge_bases(kb_json_file)
        reg1 = get_knowledge_bases()
        reg2 = get_knowledge_bases()  # should use cached
        assert reg1 is reg2
        assert reg1 is cfg._KB_REGISTRY


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestRAGSourceSchema:
    """Tests for the RAGSource and ChatCompletionRequest schemas."""

    def test_rag_source_defaults(self):
        from gateway.schemas.openai_chat import RAGSource

        src = RAGSource(knowledge_base="arxiv")
        assert src.alias is None

    def test_rag_source_explicit_alias(self):
        from gateway.schemas.openai_chat import RAGSource

        src = RAGSource(knowledge_base="arxiv", alias="challenger")
        assert src.alias == "challenger"

    def test_chat_request_rag_sources_none(self):
        from gateway.schemas.openai_chat import ChatCompletionRequest

        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
        )
        assert req.rag_sources is None

    def test_chat_request_rag_sources_list(self):
        from gateway.schemas.openai_chat import ChatCompletionRequest

        req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            rag_sources=[
                {"knowledge_base": "arxiv"},
                {"knowledge_base": "pytorch_docs", "alias": "challenger"},
            ],
        )
        assert len(req.rag_sources) == 2
        assert req.rag_sources[0].alias is None
        assert req.rag_sources[1].alias == "challenger"

    def test_chat_request_no_knowledge_base_field(self):
        """The old knowledge_base field should not exist."""
        from gateway.schemas.openai_chat import ChatCompletionRequest

        assert "knowledge_base" not in ChatCompletionRequest.model_fields


# ---------------------------------------------------------------------------
# API endpoint tests (error handling + admin)
# ---------------------------------------------------------------------------


def _make_test_app():
    """Build a minimal FastAPI app with our routes for testing."""
    from gateway.api.v1 import knowledge_bases, openai_compat

    app = FastAPI()
    app.include_router(openai_compat.router, prefix="/v1")
    app.include_router(knowledge_bases.router, prefix="/v1")
    return app


class TestChatCompletionsValidation:
    """Tests for 404 error handling on invalid KB/alias."""

    def test_unknown_kb_returns_404(self, kb_json_file: Path):
        import shared.config as cfg
        from shared.config import _load_knowledge_bases

        cfg._KB_REGISTRY = _load_knowledge_bases(kb_json_file)

        app = _make_test_app()
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "rag_sources": [{"knowledge_base": "unknown_kb"}],
            },
        )
        assert resp.status_code == 404
        assert "unavailable" in resp.json()["detail"].lower()

    def test_invalid_alias_returns_404(self, kb_json_file: Path):
        import shared.config as cfg
        from shared.config import _load_knowledge_bases

        cfg._KB_REGISTRY = _load_knowledge_bases(kb_json_file)

        app = _make_test_app()
        client = TestClient(app, raise_server_exceptions=False)

        # pytorch_docs only has "champion", not "challenger"
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "rag_sources": [{"knowledge_base": "pytorch_docs", "alias": "challenger"}],
            },
        )
        assert resp.status_code == 404
        assert "alias" in resp.json()["detail"].lower()


class TestKnowledgeBasesEndpoint:
    """Tests for GET /v1/knowledge-bases."""

    def test_list_knowledge_bases(self, kb_json_file: Path):
        import shared.config as cfg
        from shared.config import _load_knowledge_bases

        cfg._KB_REGISTRY = _load_knowledge_bases(kb_json_file)

        app = _make_test_app()
        client = TestClient(app)

        resp = client.get("/v1/knowledge-bases")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2

        names = {kb["knowledge_base"] for kb in data}
        assert names == {"arxiv", "pytorch_docs"}

        arxiv_entry = next(kb for kb in data if kb["knowledge_base"] == "arxiv")
        assert arxiv_entry["update_strategy"] == "incremental"
        assert "champion" in arxiv_entry["aliases"]

    def test_list_knowledge_bases_empty(self, tmp_path: Path):
        import shared.config as cfg
        from shared.config import _load_knowledge_bases

        cfg._KB_REGISTRY = _load_knowledge_bases(tmp_path / "nonexistent.json")

        app = _make_test_app()
        client = TestClient(app)

        resp = client.get("/v1/knowledge-bases")
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# Vector store metadata exclusion tests
# ---------------------------------------------------------------------------


class TestMetadataExclusion:
    """Test that search() excludes _meta sentinel points."""

    def test_meta_id_is_valid_uuid(self):
        """Verify _META_ID is a valid UUID string (Qdrant requirement)."""
        import uuid

        from rag.vector_store import QdrantVectorStore

        meta_id = QdrantVectorStore._META_ID
        # Must be a string
        assert isinstance(meta_id, str)
        # Must be parseable as a UUID
        parsed = uuid.UUID(meta_id)
        assert str(parsed) == meta_id

    def test_search_filter_includes_meta_exclusion(self):
        """Verify the filter is built with must_not for collection_meta."""
        from qdrant_client.models import FieldCondition, Filter

        from rag.vector_store import QdrantVectorStore

        # We can't actually connect to Qdrant in unit tests, but we can
        # verify the filter construction logic by inspecting the method.
        # Instead, let's mock the client.
        with patch("rag.vector_store.QdrantClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.query_points.return_value = MagicMock(points=[])

            vs = QdrantVectorStore(
                host="localhost",
                port=6333,
                collection_name="test",
            )
            vs.search(query_embedding=[0.1] * 10, top_k=5)

            # Verify query_points was called with the right filter
            call_kwargs = mock_client.query_points.call_args.kwargs
            qf = call_kwargs["query_filter"]
            assert isinstance(qf, Filter)
            assert qf.must_not is not None
            assert len(qf.must_not) >= 1
            meta_cond = qf.must_not[0]
            assert isinstance(meta_cond, FieldCondition)
            assert meta_cond.key == "type"

    def test_search_with_existing_filter(self):
        """Verify meta exclusion is appended to existing filters."""
        from qdrant_client.models import Filter

        from rag.vector_store import QdrantVectorStore

        with patch("rag.vector_store.QdrantClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.query_points.return_value = MagicMock(points=[])

            vs = QdrantVectorStore(
                host="localhost",
                port=6333,
                collection_name="test",
            )
            vs.search(
                query_embedding=[0.1] * 10,
                top_k=5,
                filter_dict={"must": [{"key": "task", "match": {"value": "chat"}}]},
            )

            call_kwargs = mock_client.query_points.call_args.kwargs
            qf = call_kwargs["query_filter"]
            assert isinstance(qf, Filter)
            # Should have the original must condition AND the meta exclusion
            assert qf.must is not None
            assert qf.must_not is not None


# ---------------------------------------------------------------------------
# RAGService alias resolution tests
# ---------------------------------------------------------------------------


class TestRAGServiceResolution:
    """Test RAGService._qdrant_alias and validation logic."""

    def test_qdrant_alias_construction(self):
        from gateway.services.rag_service import RAGService

        assert RAGService._qdrant_alias("arxiv", "champion") == "arxiv_champion"
        assert RAGService._qdrant_alias("pytorch_docs", "challenger") == "pytorch_docs_challenger"

    def test_available_knowledge_bases(self, kb_json_file: Path):
        import shared.config as cfg
        from shared.config import _load_knowledge_bases

        cfg._KB_REGISTRY = _load_knowledge_bases(kb_json_file)

        from gateway.services.rag_service import RAGService

        result = RAGService.available_knowledge_bases()
        assert "arxiv" in result
        assert "pytorch_docs" in result
        assert result["arxiv"]["update_strategy"] == "incremental"
