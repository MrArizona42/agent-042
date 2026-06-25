from __future__ import annotations

from unittest.mock import patch

from app_config.catalog import AdapterConfig, KBConfig, TaskConfig, catalog_override
from app_config.runtime import Settings, load_settings
from gateway.domain.rag_service import RAGService


def _alias_config() -> dict[str, object]:
    return {
        "top_k": 5,
        "score_threshold": 0.35,
        "reranker": None,
        "retrieval_strategy": "dense",
        "reranker_multiplier": 4,
    }


class _FakeEmbeddingService:
    def __init__(self) -> None:
        self.embed_documents_calls = 0
        self._document_vectors = {
            "Research papers and literature-grounded answers.": [1.0, 0.0],
            "PyTorch API reference and implementation guidance.": [0.0, 1.0],
        }
        self._query_vectors = {
            "Explain the latest transformer paper": [0.95, 0.05],
            "What tensor shape does Conv2d expect?": [0.05, 0.95],
            "Unrelated request": [0.2, 0.2],
        }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_documents_calls += 1
        return [self._document_vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._query_vectors[text]


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
    }
    gateway_values = {
        "embeddings_timeout": 10.0,
    }
    rag_values = {
        "enabled": True,
        "embedding_model": "test-embedding",
        "embedding_device": "cpu",
        "build": {"embedding_batch_size": 32, "qdrant_upsert_batch_size": 128},
        "kb_selection_threshold": 0.3,
        "strict_startup": False,
        "sparse_encoder_model": "Qdrant/bm25",
    }
    if platform is not None:
        platform_values.update(platform)
    if behavior is not None:
        gateway_values.update(behavior)
    if rag is not None:
        rag_values.update(rag)
    return load_settings(
        overrides={
            "platform": platform_values,
            "gateway": gateway_values,
            "rag": rag_values,
        }
    )


def _build_registry() -> dict[str, TaskConfig]:
    arxiv = KBConfig(
        name="arxiv",
        default_alias="champion",
        aliases={"champion": _alias_config()},
        description="Research papers and literature-grounded answers.",
    )
    pytorch_docs = KBConfig(
        name="pytorch_docs",
        default_alias="champion",
        aliases={"champion": _alias_config()},
        description="PyTorch API reference and implementation guidance.",
    )

    return {
        "chat": TaskConfig(
            task="chat",
            description="General ML research discussion.",
            adapter=AdapterConfig(name="", alias="", enabled=False),
            knowledge_bases=[arxiv],
        ),
        "code": TaskConfig(
            task="code",
            description="Programming help for ML systems.",
            adapter=AdapterConfig(name="", alias="", enabled=False),
            knowledge_bases=[pytorch_docs],
        ),
        "summarize": TaskConfig(
            task="summarize",
            description="Summarize user-provided content.",
            adapter=AdapterConfig(name="", alias="", enabled=False),
            knowledge_bases=[],
        ),
    }


def test_select_knowledge_bases_returns_task_scoped_match() -> None:
    with catalog_override(_build_registry()):
        embedding_service = _FakeEmbeddingService()
        with patch("gateway.domain.rag_service.EmbeddingService", return_value=embedding_service):
            service = RAGService(settings=_settings())

        sources = service.select_knowledge_bases(
            "Explain the latest transformer paper",
            task="chat",
        )

        assert [source.knowledge_base for source in sources] == ["arxiv"]


def test_select_knowledge_bases_returns_empty_below_threshold() -> None:
    with catalog_override(_build_registry()):
        with patch(
            "gateway.domain.rag_service.EmbeddingService",
            return_value=_FakeEmbeddingService(),
        ):
            service = RAGService(settings=_settings(rag={"kb_selection_threshold": 0.8}))

        sources = service.select_knowledge_bases("Unrelated request", task="chat")

        assert sources == []


def test_select_knowledge_bases_skips_tasks_without_kbs() -> None:
    with catalog_override(_build_registry()):
        with patch(
            "gateway.domain.rag_service.EmbeddingService",
            return_value=_FakeEmbeddingService(),
        ):
            service = RAGService(settings=_settings())

        sources = service.select_knowledge_bases(
            "Explain the latest transformer paper", task="summarize"
        )

        assert sources == []


def test_select_knowledge_bases_caches_kb_prototypes_until_invalidated() -> None:
    with catalog_override(_build_registry()):
        embedding_service = _FakeEmbeddingService()
        with patch("gateway.domain.rag_service.EmbeddingService", return_value=embedding_service):
            service = RAGService(settings=_settings())

        service.select_knowledge_bases("Explain the latest transformer paper", task="chat")
        service.select_knowledge_bases("What tensor shape does Conv2d expect?", task="code")

        assert embedding_service.embed_documents_calls == 1

        service.invalidate_caches()
        service.select_knowledge_bases("Explain the latest transformer paper", task="chat")

        assert embedding_service.embed_documents_calls == 2
