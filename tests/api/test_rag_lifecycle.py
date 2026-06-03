"""Tests for RAG alias-based lifecycle features.

Tests the knowledge-base config loader, RAGSource schema,
ChatCompletionRequest with rag_sources, metadata exclusion filter,
admin endpoint, error handling for missing KB/aliases, config
propagation to Retriever, legacy metadata handling, and reload-config.
"""

from __future__ import annotations

import os
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from shared.config import Settings
from shared.catalog import (
    get_kb_config,
    get_catalog,
    load_catalog,
    catalog_override,
)
from tests.catalog_samples import (
    write_chat_and_code_catalog,
    write_chat_only_catalog,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

# Ensure settings don't bleed between tests
os.environ.setdefault("RAG__RAG_ENABLED", "false")

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
        "default_model": "test-model",
        "async_enabled": True,
        "cors_allow_origins": [],
        "embeddings_timeout": 30.0,
    }
    rag_values: dict[str, object] = {
        "rag_enabled": True,
        "embedding_model": "test-model",
        "embedding_device": "cpu",
        "build": {"embedding_batch_size": 32, "qdrant_upsert_batch_size": 128},
        "rag_strict_startup": False,
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

    return Settings(
        platform=platform_values,
        gateway=gateway_values,
        rag=rag_values,
        auth=auth_values,
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

    def test_load_missing_file_returns_empty(self, tmp_path: Path):
        catalog, index = load_catalog(tmp_path / "nonexistent.toml")
        assert catalog == {}
        assert index == {}
        assert index == {}

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

    def test_list_knowledge_bases_empty(self, tmp_path: Path):
        _override_loaded_kb_catalog(tmp_path / "nonexistent.toml")

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
    """Test RAGService._qdrant_alias and validation logic."""

    def test_qdrant_alias_construction(self):
        from gateway.services.rag_service import RAGService

        assert (
            RAGService._qdrant_alias("ml_papers_core", "champion")
            == "ml_papers_core_champion"
        )
        assert (
            RAGService._qdrant_alias("pytorch_reference", "challenger")
            == "pytorch_reference_challenger"
        )

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


# ---------------------------------------------------------------------------
# RAGService.retrieve_documents config propagation tests
# ---------------------------------------------------------------------------


class TestRetrieveDocumentsConfig:
    """RAGService.retrieve_documents passes alias/build config to Retriever."""

    def test_passes_alias_and_build_config(self, catalog_file: Path):
        """retrieve_documents passes top_k, score_threshold, and strategy."""
        _override_loaded_kb_catalog(catalog_file)

        with (
            patch("gateway.services.rag_service.EmbeddingService"),
            patch("gateway.services.rag_service.QdrantVectorStore") as mock_vs_cls,
            patch("gateway.services.rag_service.get_settings") as mock_get_settings,
        ):
            mock_settings = _make_gateway_settings()
            mock_get_settings.return_value = mock_settings

            mock_vs = mock_vs_cls.return_value
            mock_vs.collection_exists.return_value = True

            from rag.ops.meta import BuildConfig

            build_cfg = BuildConfig(
                chunking_strategy="recursive",
                chunk_size=512,
                chunk_overlap=64,
                embedding_model="test-model",
                sparse_encoder=None,
                retrieval_capability="hybrid",
            )

            from gateway.services.rag_service import RAGService

            svc = RAGService(settings=mock_settings)
            # Pre-populate build config cache
            svc._build_configs["ml_papers_core_20260401"] = build_cfg
            svc._resolved_collections["ml_papers_core_champion"] = "ml_papers_core_20260401"

            # Mock the retriever to capture the call
            mock_retriever = MagicMock()
            mock_retriever.retrieve.return_value = []
            svc._retrievers["ml_papers_core_champion"] = mock_retriever
            mock_vs.resolve_alias.return_value = "ml_papers_core_20260401"

            svc.retrieve_documents(
                query="test query",
                knowledge_base="ml_papers_core",
                alias="champion",
            )

            mock_retriever.retrieve.assert_called_once_with(
                query="test query",
                top_k=5,
                score_threshold=0.35,
                strategy="dense",
            )

    def test_lazily_reads_build_config_when_cache_empty(self, catalog_file: Path):
        """Serving-path retrieval re-reads _meta after cache invalidation."""
        _override_loaded_kb_catalog(catalog_file)

        with (
            patch("gateway.services.rag_service.EmbeddingService") as mock_embedding_cls,
            patch("gateway.services.rag_service.QdrantVectorStore") as mock_vs_cls,
            patch("gateway.services.rag_service.read_collection_meta") as mock_read_meta,
            patch("gateway.services.rag_service.Retriever") as mock_retriever_cls,
            patch("gateway.services.rag_service.get_settings") as mock_get_settings,
        ):
            mock_settings = _make_gateway_settings()
            mock_get_settings.return_value = mock_settings

            mock_embedding = mock_embedding_cls.return_value
            mock_embedding.dimension = 384

            mock_vs = mock_vs_cls.return_value
            mock_vs.collection_exists.return_value = True
            mock_vs.resolve_alias.return_value = None
            mock_vs.get_collection_info.return_value = {
                "exists": True,
                "points_count": 0,
                "vector_size": 384,
            }

            from rag.ops.meta import BuildConfig

            build_cfg = BuildConfig(
                chunking_strategy="recursive",
                chunk_size=512,
                chunk_overlap=64,
                embedding_model="test-model",
                sparse_encoder="Qdrant/bm25",
                retrieval_capability="hybrid",
            )
            mock_read_meta.return_value = MagicMock(build_config=build_cfg)

            mock_retriever = mock_retriever_cls.return_value
            mock_retriever.retrieve.return_value = []

            from gateway.services.rag_service import RAGService

            svc = RAGService(settings=mock_settings)
            svc.retrieve_documents(
                query="test query",
                knowledge_base="ml_papers_core",
                alias="champion",
            )

            mock_read_meta.assert_called_once()
            mock_retriever.retrieve.assert_called_once_with(
                query="test query",
                top_k=5,
                score_threshold=0.35,
                strategy="dense",
            )

    def test_reuses_build_config_cache_for_aliases_on_same_collection(self, catalog_file: Path):
        """Different aliases should reuse build metadata for the same physical target."""
        _override_loaded_kb_catalog(catalog_file)

        with (
            patch("gateway.services.rag_service.EmbeddingService") as mock_embedding_cls,
            patch("gateway.services.rag_service.QdrantVectorStore") as mock_vs_cls,
            patch("gateway.services.rag_service.read_collection_meta") as mock_read_meta,
            patch("gateway.services.rag_service.Retriever") as mock_retriever_cls,
            patch("gateway.services.rag_service.get_settings") as mock_get_settings,
        ):
            mock_settings = _make_gateway_settings()
            mock_get_settings.return_value = mock_settings

            mock_embedding = mock_embedding_cls.return_value
            mock_embedding.dimension = 384

            stores: dict[str, MagicMock] = {}

            def make_store(*, host, port, collection_name):
                del host, port
                store = stores.get(collection_name)
                if store is not None:
                    return store

                store = MagicMock()
                store.collection_name = collection_name
                store.collection_exists.return_value = True
                if collection_name in {"ml_papers_core_champion", "ml_papers_core_challenger"}:
                    store.resolve_alias.return_value = "ml_papers_core_20260401"
                else:
                    store.resolve_alias.return_value = None
                    store.get_collection_info.return_value = {
                        "exists": True,
                        "points_count": 0,
                        "vector_size": 384,
                    }
                stores[collection_name] = store
                return store

            mock_vs_cls.side_effect = make_store

            from rag.ops.meta import BuildConfig

            build_cfg = BuildConfig(
                chunking_strategy="recursive",
                chunk_size=512,
                chunk_overlap=64,
                embedding_model="test-model",
                sparse_encoder="Qdrant/bm25",
                retrieval_capability="hybrid",
            )
            mock_read_meta.return_value = MagicMock(build_config=build_cfg)

            champion_retriever = MagicMock()
            champion_retriever.retrieve.return_value = []
            challenger_retriever = MagicMock()
            challenger_retriever.retrieve.return_value = []
            mock_retriever_cls.side_effect = [champion_retriever, challenger_retriever]

            from gateway.services.rag_service import RAGService

            svc = RAGService(settings=mock_settings)
            svc.retrieve_documents(query="q1", knowledge_base="ml_papers_core", alias="champion")
            svc.retrieve_documents(query="q2", knowledge_base="ml_papers_core", alias="challenger")

            mock_read_meta.assert_called_once()
            assert svc._resolved_collections["ml_papers_core_champion"] == "ml_papers_core_20260401"
            assert svc._resolved_collections["ml_papers_core_challenger"] == "ml_papers_core_20260401"
            assert "ml_papers_core_20260401" in svc._build_configs

    def test_alias_rebind_refreshes_retriever_and_build_config(self, catalog_file: Path):
        """The next request after an alias rebind must use the new target metadata."""
        _override_loaded_kb_catalog(catalog_file)

        with (
            patch("gateway.services.rag_service.EmbeddingService") as mock_embedding_cls,
            patch("gateway.services.rag_service.QdrantVectorStore") as mock_vs_cls,
            patch("gateway.services.rag_service.read_collection_meta") as mock_read_meta,
            patch("gateway.services.rag_service.Retriever") as mock_retriever_cls,
            patch("gateway.services.rag_service.get_settings") as mock_get_settings,
        ):
            mock_settings = _make_gateway_settings()
            mock_get_settings.return_value = mock_settings

            mock_embedding = mock_embedding_cls.return_value
            mock_embedding.dimension = 384

            current_target = {"name": "ml_papers_core_20260401"}
            stores: dict[str, MagicMock] = {}

            def make_store(*, host, port, collection_name):
                del host, port
                store = stores.get(collection_name)
                if store is not None:
                    return store

                store = MagicMock()
                store.collection_name = collection_name
                store.collection_exists.return_value = True
                if collection_name == "ml_papers_core_champion":
                    store.resolve_alias.side_effect = lambda alias_name: current_target["name"]
                else:
                    store.resolve_alias.return_value = None
                    store.get_collection_info.return_value = {
                        "exists": True,
                        "points_count": 0,
                        "vector_size": 384,
                    }
                stores[collection_name] = store
                return store

            mock_vs_cls.side_effect = make_store

            from rag.ops.meta import BuildConfig

            old_build_cfg = BuildConfig(
                chunking_strategy="recursive",
                chunk_size=512,
                chunk_overlap=64,
                embedding_model="old-model",
                sparse_encoder=None,
                retrieval_capability="dense",
            )
            new_build_cfg = BuildConfig(
                chunking_strategy="recursive",
                chunk_size=512,
                chunk_overlap=64,
                embedding_model="new-model",
                sparse_encoder=None,
                retrieval_capability="dense",
            )

            def read_meta_side_effect(vector_store, *, context):
                if vector_store.collection_name == "ml_papers_core_20260401":
                    return MagicMock(build_config=old_build_cfg)
                if vector_store.collection_name == "ml_papers_core_20260402":
                    return MagicMock(build_config=new_build_cfg)
                raise AssertionError(f"Unexpected collection: {vector_store.collection_name}")

            mock_read_meta.side_effect = read_meta_side_effect

            first_retriever = MagicMock()
            first_retriever.retrieve.return_value = []
            second_retriever = MagicMock()
            second_retriever.retrieve.return_value = []
            mock_retriever_cls.side_effect = [first_retriever, second_retriever]

            from gateway.services.rag_service import RAGService

            svc = RAGService(settings=mock_settings)

            svc.retrieve_documents(query="first", knowledge_base="ml_papers_core", alias="champion")

            current_target["name"] = "ml_papers_core_20260402"

            svc.retrieve_documents(query="second", knowledge_base="ml_papers_core", alias="champion")

            assert mock_retriever_cls.call_count == 2
            assert svc._resolved_collections["ml_papers_core_champion"] == "ml_papers_core_20260402"
            assert "ml_papers_core_20260401" in svc._build_configs
            assert "ml_papers_core_20260402" in svc._build_configs
            assert [call.kwargs["context"] for call in mock_read_meta.call_args_list] == [
                "ml_papers_core_20260401",
                "ml_papers_core_20260402",
            ]
            first_retriever.retrieve.assert_called_once()
            second_retriever.retrieve.assert_called_once()

    def test_retriever_failure_raises_instead_of_returning_empty(self, catalog_file: Path):
        """Pipeline failures must propagate so requests fail closed."""
        _override_loaded_kb_catalog(catalog_file)

        with (
            patch("gateway.services.rag_service.EmbeddingService"),
            patch("gateway.services.rag_service.QdrantVectorStore") as mock_vs_cls,
            patch("gateway.services.rag_service.get_settings") as mock_get_settings,
        ):
            mock_settings = _make_gateway_settings()
            mock_get_settings.return_value = mock_settings
            mock_vs = mock_vs_cls.return_value
            mock_vs.collection_exists.return_value = True

            from rag.ops.meta import BuildConfig

            build_cfg = BuildConfig(
                chunking_strategy="recursive",
                chunk_size=512,
                chunk_overlap=64,
                embedding_model="test-model",
                sparse_encoder=None,
                retrieval_capability="dense",
            )

            from gateway.services.rag_service import RAGService

            svc = RAGService(settings=mock_settings)
            svc._build_configs["ml_papers_core_20260401"] = build_cfg
            svc._resolved_collections["ml_papers_core_champion"] = "ml_papers_core_20260401"

            mock_retriever = MagicMock()
            mock_retriever.retrieve.side_effect = RuntimeError("reranker down")
            svc._retrievers["ml_papers_core_champion"] = mock_retriever
            mock_vs.resolve_alias.return_value = "ml_papers_core_20260401"

            with pytest.raises(RuntimeError, match="Failed to retrieve RAG documents"):
                svc.retrieve_documents(
                    query="test query",
                    knowledge_base="ml_papers_core",
                    alias="champion",
                )


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
# Legacy metadata handling tests
# ---------------------------------------------------------------------------


class TestLegacyMetadataHandling:
    """Legacy collections missing sparse_encoder/retrieval_strategy are rejected."""

    def _make_catalog(self, tmp_path: Path, *, retrieval_strategy: str = "dense") -> Path:
        return write_chat_only_catalog(
            tmp_path / "kb.toml",
            retrieval_strategy=retrieval_strategy,
        )

    def test_legacy_meta_non_strict_marks_unavailable(self, tmp_path: Path):
        """In non-strict mode, legacy _meta marks alias unavailable."""
        kb_path = self._make_catalog(tmp_path)
        _override_loaded_kb_catalog(kb_path)

        with (
            patch("gateway.services.rag_service.EmbeddingService"),
            patch("gateway.services.rag_service.QdrantVectorStore") as mock_vs_cls,
            patch("gateway.services.rag_service.read_collection_meta") as mock_read_meta,
            patch("gateway.services.rag_service.get_settings") as mock_get_settings,
        ):
            mock_settings = _make_gateway_settings()
            mock_get_settings.return_value = mock_settings

            mock_vs = mock_vs_cls.return_value
            mock_vs.collection_exists.return_value = True

            # Simulate legacy _meta missing retrieval_strategy
            mock_read_meta.side_effect = ValueError(
                "build_config: 'retrieval_strategy' must be one of "
                "'dense', 'hybrid', 'sparse' (got None)"
            )

            from gateway.services.rag_service import RAGService

            svc = RAGService(settings=mock_settings)
            svc.validate_knowledge_bases()

            assert "ml_papers_core_champion" in svc._unavailable

    def test_legacy_meta_strict_raises(self, tmp_path: Path):
        """With rag_strict_startup=True, legacy _meta raises RuntimeError."""
        kb_path = self._make_catalog(tmp_path)
        _override_loaded_kb_catalog(kb_path)

        with (
            patch("gateway.services.rag_service.EmbeddingService"),
            patch("gateway.services.rag_service.QdrantVectorStore") as mock_vs_cls,
            patch("gateway.services.rag_service.read_collection_meta") as mock_read_meta,
            patch("gateway.services.rag_service.get_settings") as mock_get_settings,
        ):
            mock_settings = _make_gateway_settings(rag={"rag_strict_startup": True})
            mock_get_settings.return_value = mock_settings

            mock_vs = mock_vs_cls.return_value
            mock_vs.collection_exists.return_value = True

            mock_read_meta.side_effect = ValueError("retrieval_strategy")

            from gateway.services.rag_service import RAGService

            svc = RAGService(settings=mock_settings)

            with pytest.raises(RuntimeError, match="Failed to read _meta"):
                svc.validate_knowledge_bases()

    def test_build_config_from_payload_rejects_missing_retrieval_strategy(self):
        """BuildConfig.from_payload rejects payload without retrieval_strategy."""
        from rag.ops.meta import BuildConfig

        legacy_payload = {
            "chunking_strategy": "recursive",
            "chunk_size": 512,
            "chunk_overlap": 64,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        }

        with pytest.raises(ValueError, match="retrieval_capability"):
            BuildConfig.from_payload(legacy_payload)

    def test_dimension_mismatch_non_strict_marks_unavailable(self, tmp_path: Path):
        """In non-strict mode, embedding dimension mismatch marks alias unavailable."""
        kb_path = self._make_catalog(tmp_path)
        _override_loaded_kb_catalog(kb_path)

        with (
            patch("gateway.services.rag_service.EmbeddingService") as mock_embedding_cls,
            patch("gateway.services.rag_service.QdrantVectorStore") as mock_vs_cls,
            patch("gateway.services.rag_service.read_collection_meta") as mock_read_meta,
            patch("gateway.services.rag_service.get_settings") as mock_get_settings,
        ):
            mock_settings = _make_gateway_settings()
            mock_get_settings.return_value = mock_settings

            mock_embedding = mock_embedding_cls.return_value
            mock_embedding.dimension = 384

            mock_vs = mock_vs_cls.return_value
            mock_vs.collection_exists.return_value = True
            mock_vs.get_collection_info.return_value = {
                "exists": True,
                "points_count": 0,
                "vector_size": 1536,
            }

            from rag.ops.meta import BuildConfig

            mock_read_meta.return_value = MagicMock(
                build_config=BuildConfig(
                    chunking_strategy="recursive",
                    chunk_size=512,
                    chunk_overlap=64,
                    embedding_model="other-model",
                    sparse_encoder=None,
                    retrieval_capability="dense",
                )
            )

            from gateway.services.rag_service import RAGService

            svc = RAGService(settings=mock_settings)
            svc.validate_knowledge_bases()

            assert "ml_papers_core_champion" in svc._unavailable

    def test_dimension_mismatch_strict_raises(self, tmp_path: Path):
        """With rag_strict_startup=True, dimension mismatch raises RuntimeError."""
        kb_path = self._make_catalog(tmp_path)
        _override_loaded_kb_catalog(kb_path)

        with (
            patch("gateway.services.rag_service.EmbeddingService") as mock_embedding_cls,
            patch("gateway.services.rag_service.QdrantVectorStore") as mock_vs_cls,
            patch("gateway.services.rag_service.read_collection_meta") as mock_read_meta,
            patch("gateway.services.rag_service.get_settings") as mock_get_settings,
        ):
            mock_settings = _make_gateway_settings(rag={"rag_strict_startup": True})
            mock_get_settings.return_value = mock_settings

            mock_embedding = mock_embedding_cls.return_value
            mock_embedding.dimension = 384

            mock_vs = mock_vs_cls.return_value
            mock_vs.collection_exists.return_value = True
            mock_vs.get_collection_info.return_value = {
                "exists": True,
                "points_count": 0,
                "vector_size": 1536,
            }

            from rag.ops.meta import BuildConfig

            mock_read_meta.return_value = MagicMock(
                build_config=BuildConfig(
                    chunking_strategy="recursive",
                    chunk_size=512,
                    chunk_overlap=64,
                    embedding_model="other-model",
                    sparse_encoder=None,
                    retrieval_capability="dense",
                )
            )

            from gateway.services.rag_service import RAGService

            svc = RAGService(settings=mock_settings)

            with pytest.raises(RuntimeError, match="Embedding dimension mismatch"):
                svc.validate_knowledge_bases()

    def test_hybrid_query_dense_build_non_strict_marks_unavailable(self, tmp_path: Path):
        """Incompatible alias/build capability marks the alias unavailable."""
        kb_path = self._make_catalog(tmp_path, retrieval_strategy="hybrid")
        _override_loaded_kb_catalog(kb_path)

        with (
            patch("gateway.services.rag_service.EmbeddingService") as mock_embedding_cls,
            patch("gateway.services.rag_service.QdrantVectorStore") as mock_vs_cls,
            patch("gateway.services.rag_service.read_collection_meta") as mock_read_meta,
            patch("gateway.services.rag_service.get_settings") as mock_get_settings,
        ):
            mock_settings = _make_gateway_settings()
            mock_get_settings.return_value = mock_settings

            mock_embedding = mock_embedding_cls.return_value
            mock_embedding.dimension = 384

            mock_vs = mock_vs_cls.return_value
            mock_vs.collection_exists.return_value = True
            mock_vs.get_collection_info.return_value = {
                "exists": True,
                "points_count": 0,
                "vector_size": 384,
            }

            from rag.ops.meta import BuildConfig

            mock_read_meta.return_value = MagicMock(
                build_config=BuildConfig(
                    chunking_strategy="recursive",
                    chunk_size=512,
                    chunk_overlap=64,
                    embedding_model="test-model",
                    sparse_encoder=None,
                    retrieval_capability="dense",
                )
            )

            from gateway.services.rag_service import RAGService

            svc = RAGService(settings=mock_settings)
            svc.validate_knowledge_bases()

            assert "ml_papers_core_champion" in svc._unavailable

    def test_hybrid_query_dense_build_strict_raises(self, tmp_path: Path):
        """Strict startup raises on query/build capability mismatches."""
        kb_path = self._make_catalog(tmp_path, retrieval_strategy="hybrid")
        _override_loaded_kb_catalog(kb_path)

        with (
            patch("gateway.services.rag_service.EmbeddingService") as mock_embedding_cls,
            patch("gateway.services.rag_service.QdrantVectorStore") as mock_vs_cls,
            patch("gateway.services.rag_service.read_collection_meta") as mock_read_meta,
            patch("gateway.services.rag_service.get_settings") as mock_get_settings,
        ):
            mock_settings = _make_gateway_settings(rag={"rag_strict_startup": True})
            mock_get_settings.return_value = mock_settings

            mock_embedding = mock_embedding_cls.return_value
            mock_embedding.dimension = 384

            mock_vs = mock_vs_cls.return_value
            mock_vs.collection_exists.return_value = True
            mock_vs.get_collection_info.return_value = {
                "exists": True,
                "points_count": 0,
                "vector_size": 384,
            }

            from rag.ops.meta import BuildConfig

            mock_read_meta.return_value = MagicMock(
                build_config=BuildConfig(
                    chunking_strategy="recursive",
                    chunk_size=512,
                    chunk_overlap=64,
                    embedding_model="test-model",
                    sparse_encoder=None,
                    retrieval_capability="dense",
                )
            )

            from gateway.services.rag_service import RAGService

            svc = RAGService(settings=mock_settings)

            with pytest.raises(RuntimeError, match="requires build capability 'hybrid'"):
                svc.validate_knowledge_bases()

    def test_sparse_encoder_mismatch_rejects_lazy_retriever_creation(self, tmp_path: Path):
        """Serving path rejects aliases whose sparse encoder config no longer matches."""
        kb_path = self._make_catalog(tmp_path, retrieval_strategy="hybrid")
        _override_loaded_kb_catalog(kb_path)

        with (
            patch("gateway.services.rag_service.EmbeddingService") as mock_embedding_cls,
            patch("gateway.services.rag_service.QdrantVectorStore") as mock_vs_cls,
            patch("gateway.services.rag_service.read_collection_meta") as mock_read_meta,
            patch("gateway.services.rag_service.get_settings") as mock_get_settings,
        ):
            mock_settings = _make_gateway_settings(rag={"sparse_encoder_model": "other/model"})
            mock_get_settings.return_value = mock_settings

            mock_embedding = mock_embedding_cls.return_value
            mock_embedding.dimension = 384

            mock_vs = mock_vs_cls.return_value
            mock_vs.collection_exists.return_value = True
            mock_vs.get_collection_info.return_value = {
                "exists": True,
                "points_count": 0,
                "vector_size": 384,
            }

            from rag.ops.meta import BuildConfig

            mock_read_meta.return_value = MagicMock(
                build_config=BuildConfig(
                    chunking_strategy="recursive",
                    chunk_size=512,
                    chunk_overlap=64,
                    embedding_model="test-model",
                    sparse_encoder="Qdrant/bm25",
                    retrieval_capability="hybrid",
                )
            )

            from gateway.services.rag_service import RAGService

            svc = RAGService(settings=mock_settings)

            with pytest.raises(RuntimeError, match="RAG retriever unavailable"):
                svc.retrieve_documents(
                    query="test query",
                    knowledge_base="ml_papers_core",
                    alias="champion",
                )

            assert "ml_papers_core_champion" in svc._unavailable


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
    """Gateway lifespan honors rag_strict_startup for RAG validation failures."""

    def test_strict_startup_raises(self):
        import gateway.main as gateway_main

        mock_settings = _make_gateway_settings(
            rag={"embedding_model": "test-embedding", "rag_strict_startup": True}
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
            rag={"rag_enabled": False},
        )

        with (
            patch("gateway.main.get_settings", return_value=mock_settings),
            patch("gateway.main.RedisStreamService"),
        ):
            app = gateway_main.create_app()

            with pytest.raises(RuntimeError, match="GATEWAY__ASYNC_ENABLED=false"):
                with TestClient(app):
                    pass
