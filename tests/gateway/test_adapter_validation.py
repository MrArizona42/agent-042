from __future__ import annotations

import logging
from unittest.mock import patch

from gateway.services.rag_service import RAGService
from shared.catalog import AdapterConfig, KBConfig, TaskConfig, catalog_override
from shared.config import Settings


def _alias_config() -> dict[str, object]:
    return {
        "top_k": 5,
        "score_threshold": 0.35,
        "reranker": None,
        "retrieval_strategy": "dense",
        "reranker_multiplier": 4,
    }


def _settings(
    *,
    platform: dict[str, object] | None = None,
    behavior: dict[str, object] | None = None,
    rag: dict[str, object] | None = None,
) -> Settings:
    platform_values = {
        "embeddings_url": "http://embeddings:8100",
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "vllm_base_url": "http://localhost:8000",
    }
    gateway_values = {
        "embeddings_timeout": 10.0,
        "vllm_timeout": 30.0,
        "api_key": None,
        "default_model": "base-model",
    }
    rag_values = {
        "rag_enabled": True,
        "embedding_model": "test-embedding",
        "embedding_device": "cpu",
        "build": {"embedding_batch_size": 32, "qdrant_upsert_batch_size": 128},
        "kb_selection_threshold": 0.3,
        "rag_strict_startup": False,
        "sparse_encoder_model": "Qdrant/bm25",
    }
    if platform is not None:
        platform_values.update(platform)
    if behavior is not None:
        gateway_values.update(behavior)
    if rag is not None:
        rag_values.update(rag)
    return Settings(
        platform=platform_values,
        gateway=gateway_values,
        rag=rag_values,
    )


def _build_registry(*, summarize_adapter_enabled: bool) -> dict[str, TaskConfig]:
    arxiv = KBConfig(
        name="arxiv",
        default_alias="champion",
        aliases={"champion": _alias_config()},
        label="ArXiv",
        description="Research papers",
        selection_description="Research papers and literature-grounded answers.",
    )

    return {
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


def test_validate_knowledge_bases_warns_for_missing_enabled_adapter(caplog) -> None:
    with catalog_override(_build_registry(summarize_adapter_enabled=True)):
        with patch("gateway.services.rag_service.EmbeddingService") as mock_embedding_cls:
            mock_embedding = mock_embedding_cls.return_value
            mock_embedding.dimension = 384

            service = RAGService(settings=_settings())

            with (
                patch.object(service.runtime, "validate_aliases"),
                patch.object(service, "_load_available_vllm_models", return_value={"base-model"}),
            ):
                with caplog.at_level(logging.WARNING):
                    service.validate_knowledge_bases()

            assert "lora-summarize-champion" in caplog.text
            assert "fall back to default_model" in caplog.text


def test_validate_knowledge_bases_accepts_present_enabled_adapter() -> None:
    with catalog_override(_build_registry(summarize_adapter_enabled=True)):
        with patch("gateway.services.rag_service.EmbeddingService") as mock_embedding_cls:
            mock_embedding = mock_embedding_cls.return_value
            mock_embedding.dimension = 384

            service = RAGService(settings=_settings())

            with (
                patch.object(service.runtime, "validate_aliases"),
                patch.object(
                    service,
                    "_load_available_vllm_models",
                    return_value={"base-model", "lora-summarize-champion"},
                ),
            ):
                service.validate_knowledge_bases()


def test_invalidate_caches_clears_available_vllm_model_snapshot() -> None:
    with catalog_override(_build_registry(summarize_adapter_enabled=False)):
        with patch("gateway.services.rag_service.EmbeddingService"):
            service = RAGService(settings=_settings())

        service._available_vllm_models = {"base-model", "lora-chat-champion"}

        service.invalidate_caches()

        assert service._available_vllm_models is None
