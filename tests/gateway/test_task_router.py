from __future__ import annotations

from gateway.services.task_router import EmbeddingTaskRouter
from shared.operator_registry import TaskConfig


def _registry() -> dict[str, TaskConfig]:
    return {
        "chat": TaskConfig(
            task="chat",
            label="General knowledge",
            routing_description="General ML research discussion.",
        ),
        "code": TaskConfig(
            task="code",
            label="Coding assistance",
            routing_description="Programming help for ML systems.",
        ),
        "summarize": TaskConfig(
            task="summarize",
            label="Summarization",
            routing_description="Summarize user-provided content.",
        ),
    }


class _FakeEmbeddingService:
    def __init__(self) -> None:
        self.embed_documents_calls = 0
        self._document_vectors = {
            "General ML research discussion.": [1.0, 0.0, 0.0],
            "Programming help for ML systems.": [0.0, 1.0, 0.0],
            "Summarize user-provided content.": [0.0, 0.0, 1.0],
        }
        self._query_vectors = {
            "How do I fix this traceback?": [0.0, 0.9, 0.1],
            "Please summarize this paper": [0.0, 0.1, 0.95],
            "Ambiguous request": [0.6, 0.6, 0.0],
        }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_documents_calls += 1
        return [self._document_vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._query_vectors[text]


def test_embedding_task_router_selects_code_for_debug_queries() -> None:
    router = EmbeddingTaskRouter(
        embedding_service=_FakeEmbeddingService(),
        registry_loader=_registry,
    )

    decision = router.decide("How do I fix this traceback?")

    assert decision.task == "code"


def test_embedding_task_router_selects_summarize_for_summary_queries() -> None:
    router = EmbeddingTaskRouter(
        embedding_service=_FakeEmbeddingService(),
        registry_loader=_registry,
    )

    decision = router.decide("Please summarize this paper")

    assert decision.task == "summarize"


def test_embedding_task_router_caches_task_embeddings_until_invalidated() -> None:
    embedding_service = _FakeEmbeddingService()
    router = EmbeddingTaskRouter(
        embedding_service=embedding_service,
        registry_loader=_registry,
    )

    router.decide("How do I fix this traceback?")
    router.decide("Please summarize this paper")

    assert embedding_service.embed_documents_calls == 1

    router.invalidate_cache()
    router.decide("How do I fix this traceback?")

    assert embedding_service.embed_documents_calls == 2


def test_embedding_task_router_falls_back_to_chat_below_threshold() -> None:
    router = EmbeddingTaskRouter(
        embedding_service=_FakeEmbeddingService(),
        registry_loader=_registry,
        task_classification_threshold=0.8,
    )

    decision = router.decide("Ambiguous request")

    assert decision.task == "chat"


def test_embedding_task_router_falls_back_to_chat_when_embeddings_unavailable() -> None:
    def _raise_embedding_service() -> _FakeEmbeddingService:
        raise RuntimeError("embeddings unavailable")

    router = EmbeddingTaskRouter(
        embedding_service_factory=_raise_embedding_service,
        registry_loader=_registry,
    )

    decision = router.decide("How do I fix this traceback?")

    assert decision.task == "chat"
