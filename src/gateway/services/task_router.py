from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

from rag.embeddings import EmbeddingService
from shared.catalog import TaskConfig, get_catalog
from shared.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteDecision:
    task: str  # chat | summarize | code


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0

    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0
    return dot_product / (left_norm * right_norm)


class EmbeddingTaskRouter:
    """Task router that classifies the latest user message via embedding similarity.

    Task prototype embeddings are derived from ``TaskConfig.routing_description``
    and cached after the first successful build. If the embeddings service is
    unavailable, routing falls back to ``chat`` so the gateway can continue to
    serve requests.
    """

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService | None = None,
        embedding_service_factory: Callable[[], EmbeddingService] = EmbeddingService,
        catalog_loader: Callable[[], dict[str, TaskConfig]] = get_catalog,
        task_classification_threshold: float | None = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._embedding_service_factory = embedding_service_factory
        self._catalog_loader = catalog_loader
        self._task_classification_threshold = task_classification_threshold
        self._task_embeddings: dict[str, list[float]] = {}

    def invalidate_cache(self) -> None:
        self._task_embeddings.clear()

    def warm_cache(self) -> None:
        self._build_task_embeddings()

    def _ensure_embedding_service(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = self._embedding_service_factory()
        return self._embedding_service

    def _threshold(self) -> float:
        if self._task_classification_threshold is not None:
            return self._task_classification_threshold
        return float(get_settings().rag.task_classification_threshold)

    def _build_task_embeddings(self) -> dict[str, list[float]]:
        if self._task_embeddings:
            return self._task_embeddings

        catalog = self._catalog_loader()
        task_items = [
            (task_cfg.task, task_cfg.routing_description)
            for task_cfg in catalog.values()
            if task_cfg.routing_description.strip()
        ]
        if not task_items:
            raise RuntimeError("No task routing descriptions are configured")

        tasks = [task for task, _ in task_items]
        descriptions = [description for _, description in task_items]
        embeddings = self._ensure_embedding_service().embed_documents(descriptions)
        if len(embeddings) != len(tasks):
            raise RuntimeError("Task embedding count does not match configured tasks")

        self._task_embeddings = {
            task: embedding for task, embedding in zip(tasks, embeddings, strict=True)
        }
        return self._task_embeddings

    def decide(self, user_text: str) -> RouteDecision:
        if not user_text.strip():
            return RouteDecision(task="chat")

        try:
            task_embeddings = self._build_task_embeddings()
            query_embedding = self._ensure_embedding_service().embed_query(user_text)
        except Exception:
            logger.warning(
                "Task routing fallback to chat because embeddings are unavailable", exc_info=True
            )
            return RouteDecision(task="chat")

        best_task = "chat"
        best_score = float("-inf")
        for task, task_embedding in task_embeddings.items():
            score = _cosine_similarity(query_embedding, task_embedding)
            if score > best_score:
                best_task = task
                best_score = score

        threshold = self._threshold()
        if threshold > 0.0 and best_score < threshold:
            return RouteDecision(task="chat")
        return RouteDecision(task=best_task)


class RuleBasedTaskRouter(EmbeddingTaskRouter):
    """Backward-compatible alias for older imports."""
