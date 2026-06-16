"""Tests for RAG catalog, API validation, and config reload features.

Tests the knowledge-base config loader, RAGSource schema,
ChatCompletionRequest with rag_sources, metadata exclusion filter,
admin endpoint, error handling for missing KB/aliases, and reload-config.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from shared.catalog import (
    catalog_override,
    get_catalog,
    get_kb_config,
    load_catalog,
)
from shared.config import Settings, load_settings
from tests.catalog_samples import (
    write_chat_and_code_catalog,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

_KB_CATALOG_OVERRIDE_STACK: ExitStack | None = None


@pytest.fixture(autouse=True)
def _reset_kb_catalog():
    """Reset the KB catalog singleton between tests."""
    import shared.config as cfg

    global _KB_CATALOG_OVERRIDE_STACK

    cfg.clear_knowledge_base_caches()
    with ExitStack() as stack:
        _KB_CATALOG_OVERRIDE_STACK = stack
        yield
        _KB_CATALOG_OVERRIDE_STACK = None
    cfg.clear_knowledge_base_caches()


@pytest.fixture()
def catalog_file(tmp_path: Path):
    """Create a temporary catalog."""
    return write_chat_and_code_catalog(tmp_path / "catalog.toml")


def _override_loaded_kb_catalog(path: Path) -> None:
    catalog, index = load_catalog(path)
    if _KB_CATALOG_OVERRIDE_STACK is None:
        raise RuntimeError("KB catalog override stack is not initialized")
    _KB_CATALOG_OVERRIDE_STACK.enter_context(catalog_override(catalog, index=index))


def _make_gateway_settings(
    *,
    platform: dict[str, object] | None = None,
    behavior: dict[str, object] | None = None,
    rag: dict[str, object] | None = None,
    auth: dict[str, object] | None = None,
) -> Settings:
    platform_values: dict[str, object] = {
        "vllm_base_url": "http://localhost:8000",
        "embeddings_url": "http://embeddings:8100",
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "redis_url": "redis://localhost:6379/0",
        "celery_broker_url": "amqp://guest:guest@localhost//",
    }
    gateway_values: dict[str, object] = {
        "service_name": "gateway-test",
        "async_enabled": True,
        "cors_allow_origins": [],
        "embeddings_timeout": 30.0,
    }
    rag_values: dict[str, object] = {
        "enabled": True,
        "embedding_model": "test-model",
        "embedding_device": "cpu",
        "build": {"embedding_batch_size": 32, "qdrant_upsert_batch_size": 128},
        "strict_startup": False,
        "sparse_encoder_model": "Qdrant/bm25",
    }
    auth_values: dict[str, object] = {
        "google_client_id": "",
        "agent042_db_url": None,
    }

    if platform is not None:
        platform_values.update(platform)
    if behavior is not None:
        gateway_values.update(behavior)
    if rag is not None:
        rag_values.update(rag)
    if auth is not None:
        auth_values.update(auth)

    return load_settings(
        overrides={
            "vllm": {"model": "test-model"},
            "platform": platform_values,
            "gateway": gateway_values,
            "rag": rag_values,
            "auth": auth_values,
        }
    )


# ---------------------------------------------------------------------------
# Config loading tests
# ---------------------------------------------------------------------------


class TestKnowledgeBaseConfig:
    """Tests for the knowledge-base config loader."""

    def test_load_from_json(self, catalog_file: Path):
        catalog, index = load_catalog(catalog_file)
        assert "chat" in catalog
        assert "code" in catalog
        # Find KB configs within task groups
        ml_papers_cfg = catalog["chat"].knowledge_bases[0]
        pytorch_cfg = catalog["code"].knowledge_bases[0]
        assert ml_papers_cfg.name == "ml_papers_core"
        assert pytorch_cfg.name == "pytorch_reference"
        assert ml_papers_cfg.update_strategy == "replace"
        assert pytorch_cfg.update_strategy == "replace"
        assert "champion" in ml_papers_cfg.aliases
        assert "challenger" in ml_papers_cfg.aliases
        assert pytorch_cfg.label == "PyTorch reference"

    def test_load_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Catalog config file not found"):
            load_catalog(tmp_path / "nonexistent.toml")

    def test_kb_index_lookup(self, catalog_file: Path):
        """get_kb_config returns correct entries from the flat index."""
        catalog, index = load_catalog(catalog_file)

        with catalog_override(catalog, index=index):
            ml_papers_cfg = get_kb_config("ml_papers_core")
            assert ml_papers_cfg is not None
            assert ml_papers_cfg.label == "Core ML papers"
            assert "champion" in ml_papers_cfg.aliases
            assert get_kb_config("nonexistent") is None

    def test_get_catalog_caching(self, catalog_file: Path):
        catalog, index = load_catalog(catalog_file)

        with catalog_override(catalog, index=index):
            reg1 = get_catalog()
            reg2 = get_catalog()  # should use cached
            assert reg1 is reg2
            assert reg1 is catalog


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestRAGSourceSchema:
    """Tests for the RAGSource and ChatCompletionRequest schemas."""

    def test_rag_source_defaults(self):
        from gateway.schemas.openai_chat import RAGSource

        src = RAGSource(knowledge_base="ml_papers_core")
        assert src.alias is None

    def test_rag_source_explicit_alias(self):
        from gateway.schemas.openai_chat import RAGSource

        src = RAGSource(knowledge_base="ml_papers_core", alias="challenger")
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
                {"knowledge_base": "ml_papers_core"},
                {"knowledge_base": "pytorch_reference", "alias": "challenger"},
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

    def test_unknown_kb_returns_404(self, catalog_file: Path):
        _override_loaded_kb_catalog(catalog_file)

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

    def test_invalid_alias_returns_404(self, catalog_file: Path):
        _override_loaded_kb_catalog(catalog_file)

        app = _make_test_app()
        client = TestClient(app, raise_server_exceptions=False)

        # This fixture leaves pytorch_reference without a challenger alias.
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "rag_sources": [{"knowledge_base": "pytorch_reference", "alias": "challenger"}],
            },
        )
        assert resp.status_code == 404
        assert "alias" in resp.json()["detail"].lower()


class TestKnowledgeBasesEndpoint:
    """Tests for GET /v1/knowledge-bases."""

    def test_list_knowledge_bases(self, catalog_file: Path):
        _override_loaded_kb_catalog(catalog_file)

        app = _make_test_app()
        client = TestClient(app)

        resp = client.get("/v1/knowledge-bases")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2

        tasks = {entry["task"] for entry in data}
        assert tasks == {"chat", "code"}

        chat_entry = next(entry for entry in data if entry["task"] == "chat")
        assert chat_entry["label"] == "General knowledge"
        assert len(chat_entry["knowledge_bases"]) == 1

        ml_papers_entry = chat_entry["knowledge_bases"][0]
        assert ml_papers_entry["knowledge_base"] == "ml_papers_core"
        assert ml_papers_entry["update_strategy"] == "replace"
        assert "champion" in ml_papers_entry["aliases"]

    def test_list_knowledge_bases_empty(self):
        if _KB_CATALOG_OVERRIDE_STACK is None:
            raise RuntimeError("KB catalog override stack is not initialized")
        _KB_CATALOG_OVERRIDE_STACK.enter_context(catalog_override({}, index={}))

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
            vs.search(query_embedding=[0.1] * 10, top_k=5, score_threshold=0.0)

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
                score_threshold=0.0,
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
    """Test RAGService catalog discovery helpers."""

    def test_available_knowledge_bases(self, catalog_file: Path):
        _override_loaded_kb_catalog(catalog_file)

        from gateway.services.rag_service import RAGService

        result = RAGService.available_knowledge_bases()
        assert "ml_papers_core" in result
        assert "pytorch_reference" in result
        assert result["ml_papers_core"]["task"] == "chat"
        assert result["pytorch_reference"]["task_label"] == "Coding assistance"
        assert result["ml_papers_core"]["update_strategy"] == "replace"

    def test_available_knowledge_bases_by_task(self, catalog_file: Path):
        _override_loaded_kb_catalog(catalog_file)

        from gateway.services.rag_service import RAGService

        result = RAGService.available_knowledge_bases_by_task()
        assert [entry["task"] for entry in result] == ["chat", "code"]
        assert result[0]["knowledge_bases"][0]["knowledge_base"] == "ml_papers_core"


class TestRequestPathFailureMode:
    """Request handling distinguishes zero-hit retrieval from pipeline failure."""

    def test_chat_request_returns_500_when_rag_pipeline_fails(self, catalog_file: Path):
        _override_loaded_kb_catalog(catalog_file)

        from gateway.api.v1 import openai_compat

        app = _make_test_app()
        client = TestClient(app, raise_server_exceptions=False)

        rag_service = MagicMock()
        rag_service.enabled = True
        rag_service.retrieve_documents.side_effect = RuntimeError("sparse encoder down")

        with patch.object(
            openai_compat.process_chat,
            "ensure_rag_service",
            return_value=rag_service,
        ):
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                    "rag_sources": [{"knowledge_base": "ml_papers_core", "alias": "champion"}],
                },
            )

        assert response.status_code == 500
        rag_service.retrieve_documents.assert_called_once_with(
            query="hi",
            knowledge_base="ml_papers_core",
            alias="champion",
        )

    def test_retrieve_rag_chunks_keeps_zero_hit_results_non_error(self, catalog_file: Path):
        from gateway.schemas.openai_chat import ChatCompletionRequest
        from gateway.services.processing import _ProcessChat

        _override_loaded_kb_catalog(catalog_file)

        processor = _ProcessChat()
        rag_service = MagicMock()
        rag_service.enabled = True
        rag_service.retrieve_documents.return_value = []
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            rag_sources=[{"knowledge_base": "ml_papers_core", "alias": "champion"}],
        )

        with patch.object(processor, "ensure_rag_service", return_value=rag_service):
            rag_chunks = processor._retrieve_rag_chunks(request.rag_sources or [], last_user="hi")

        assert rag_chunks == {}
        rag_service.retrieve_documents.assert_called_once_with(
            query="hi",
            knowledge_base="ml_papers_core",
            alias="champion",
        )


# ---------------------------------------------------------------------------
# POST /v1/admin/reload-config tests
# ---------------------------------------------------------------------------


class TestReloadConfigEndpoint:
    """Tests for the reload-config admin endpoint."""

    def test_reload_requires_auth(self):
        """Endpoint returns 503 when auth is disabled (session_manager is None)."""
        from gateway.api.v1 import knowledge_bases

        app = FastAPI()
        app.include_router(knowledge_bases.router, prefix="/v1")
        app.state.session_manager = None

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/admin/reload-config")
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()

    def test_reload_clears_caches(self, catalog_file: Path):
        """Authenticated reload clears KB caches before reload hook executes."""
        clear_called = False

        from gateway.api.v1 import knowledge_bases

        def _reload_hook(*, settings):
            assert clear_called is True

        def _clear_hook():
            nonlocal clear_called
            clear_called = True

        app = FastAPI()
        app.include_router(knowledge_bases.router, prefix="/v1")
        app.state.session_manager = MagicMock()  # auth enabled

        @app.middleware("http")
        async def fake_auth(request, call_next):
            request.state.user_id = "test-user"
            request.state.session_id = "session-123"
            return await call_next(request)

        with (
            patch.object(knowledge_bases, "clear_knowledge_base_caches", side_effect=_clear_hook),
            patch.object(
                knowledge_bases.process_chat,
                "reload_config_caches",
                side_effect=_reload_hook,
            ) as reload_caches,
            patch("gateway.api.v1.knowledge_bases.get_settings", return_value=MagicMock()),
        ):
            client = TestClient(app)
            resp = client.post("/v1/admin/reload-config")

        assert resp.status_code == 200
        assert resp.json()["status"] == "reloaded"
        reload_caches.assert_called_once()

    def test_reload_rejects_non_session_auth(self):
        """Internal API-key style auth must not be allowed to reload config."""
        from gateway.api.v1 import knowledge_bases

        app = FastAPI()
        app.include_router(knowledge_bases.router, prefix="/v1")
        app.state.session_manager = MagicMock()

        @app.middleware("http")
        async def fake_service_auth(request, call_next):
            request.state.user_id = "__service__"
            request.state.session_id = None
            return await call_next(request)

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/admin/reload-config")
        assert resp.status_code == 403
        assert "user session" in resp.json()["detail"].lower()


class TestGatewayStartupValidation:
    """Gateway lifespan honors strict_startup for RAG validation failures."""

    def test_strict_startup_raises(self):
        import gateway.main as gateway_main

        mock_settings = _make_gateway_settings(
            rag={"embedding_model": "test-embedding", "strict_startup": True}
        )

        with (
            patch("gateway.main.get_settings", return_value=mock_settings),
            patch.object(
                gateway_main.process_chat,
                "ensure_rag_service",
                side_effect=RuntimeError("boom"),
            ),
            patch("gateway.main.RedisStreamService"),
        ):
            app = gateway_main.create_app()

            with pytest.raises(RuntimeError, match="boom"):
                with TestClient(app):
                    pass

    def test_async_disabled_raises(self):
        import gateway.main as gateway_main

        mock_settings = _make_gateway_settings(
            behavior={"async_enabled": False},
            platform={"celery_broker_url": None},
            rag={"enabled": False},
        )

        with (
            patch("gateway.main.get_settings", return_value=mock_settings),
            patch("gateway.main.RedisStreamService"),
        ):
            app = gateway_main.create_app()

            with pytest.raises(RuntimeError, match="GATEWAY__ASYNC_ENABLED=false"):
                with TestClient(app):
                    pass
