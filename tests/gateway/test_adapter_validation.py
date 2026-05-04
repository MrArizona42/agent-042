from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import shared.config as cfg
from gateway.services.rag_service import RAGService
from shared.config import AdapterConfig, KBConfig, TaskConfig


def _alias_config() -> dict[str, object]:
    return {
        "top_k": 5,
        "score_threshold": 0.35,
        "reranker": None,
        "retrieval_strategy": "dense",
        "reranker_multiplier": 4,
    }


def _settings(**overrides: object) -> SimpleNamespace:
    data = {
        "rag_enabled": True,
        "embedding_model": "test-embedding",
        "embedding_device": "cpu",
        "embedding_batch_size": 32,
        "embeddings_url": "http://embeddings:8100",
        "embeddings_timeout": 10.0,
        "kb_selection_threshold": 0.3,
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "rag_strict_startup": False,
        "sparse_encoder_model": "Qdrant/bm25",
        "vllm_base_url": "http://localhost:8000",
        "vllm_timeout": 30.0,
        "api_key": None,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _load_registry(*, summarize_adapter_enabled: bool) -> None:
    arxiv = KBConfig(
        name="arxiv",
        default_alias="champion",
        aliases={"champion": _alias_config()},
        label="ArXiv",
        description="Research papers",
        selection_description="Research papers and literature-grounded answers.",
    )

    cfg._KB_REGISTRY = {
        "chat": TaskConfig(
            task="chat",
            label="General knowledge",
            routing_description="General ML research discussion.",
            adapter=AdapterConfig(name="", alias="", enabled=False),
            knowledge_bases=[arxiv],
        ),
        "summarize": TaskConfig(
            task="summarize",
            label="Summarization",
            routing_description="Summarize user-provided content.",
            adapter=AdapterConfig(
                name="lora-summarize",
                alias="champion",
                enabled=summarize_adapter_enabled,
            ),
            knowledge_bases=[],
        ),
    }
    cfg._KB_INDEX = {"arxiv": arxiv}


def test_validate_knowledge_bases_warns_for_missing_enabled_adapter(caplog) -> None:
    _load_registry(summarize_adapter_enabled=True)
    try:
        with (
            patch("gateway.services.rag_service.EmbeddingService") as mock_embedding_cls,
            patch("gateway.services.rag_service.QdrantVectorStore") as mock_vs_cls,
            patch("gateway.services.rag_service.read_collection_meta") as mock_read_meta,
        ):
            mock_embedding = mock_embedding_cls.return_value
            mock_embedding.dimension = 384

            mock_vs = mock_vs_cls.return_value
            mock_vs.collection_exists.return_value = True
            mock_vs.resolve_alias.return_value = None
            mock_vs.get_collection_info.return_value = {"vector_size": 384}

            from rag.ops.meta import BuildConfig

            mock_read_meta.return_value = MagicMock(
                build_config=BuildConfig(
                    chunking_strategy="recursive",
                    chunk_size=512,
                    chunk_overlap=64,
                    embedding_model="test-embedding",
                    sparse_encoder=None,
                    retrieval_capability="dense",
                )
            )

            service = RAGService(settings=_settings())

            with patch.object(service, "_load_available_vllm_models", return_value={"base-model"}):
                with caplog.at_level(logging.WARNING):
                    service.validate_knowledge_bases()

            assert "lora-summarize-champion" in caplog.text
            assert "fall back to default_model" in caplog.text
    finally:
        cfg._KB_REGISTRY = None
        cfg._KB_INDEX = None


def test_validate_knowledge_bases_accepts_present_enabled_adapter() -> None:
    _load_registry(summarize_adapter_enabled=True)
    try:
        with (
            patch("gateway.services.rag_service.EmbeddingService") as mock_embedding_cls,
            patch("gateway.services.rag_service.QdrantVectorStore") as mock_vs_cls,
            patch("gateway.services.rag_service.read_collection_meta") as mock_read_meta,
        ):
            mock_embedding = mock_embedding_cls.return_value
            mock_embedding.dimension = 384

            mock_vs = mock_vs_cls.return_value
            mock_vs.collection_exists.return_value = True
            mock_vs.resolve_alias.return_value = None
            mock_vs.get_collection_info.return_value = {"vector_size": 384}

            from rag.ops.meta import BuildConfig

            mock_read_meta.return_value = MagicMock(
                build_config=BuildConfig(
                    chunking_strategy="recursive",
                    chunk_size=512,
                    chunk_overlap=64,
                    embedding_model="test-embedding",
                    sparse_encoder=None,
                    retrieval_capability="dense",
                )
            )

            service = RAGService(settings=_settings())

            with patch.object(
                service,
                "_load_available_vllm_models",
                return_value={"base-model", "lora-summarize-champion"},
            ):
                service.validate_knowledge_bases()
    finally:
        cfg._KB_REGISTRY = None
        cfg._KB_INDEX = None


def test_invalidate_caches_clears_available_vllm_model_snapshot() -> None:
    _load_registry(summarize_adapter_enabled=False)
    try:
        with patch("gateway.services.rag_service.EmbeddingService"):
            service = RAGService(settings=_settings())

        service._available_vllm_models = {"base-model", "lora-chat-champion"}

        service.invalidate_caches()

        assert service._available_vllm_models is None
    finally:
        cfg._KB_REGISTRY = None
        cfg._KB_INDEX = None
