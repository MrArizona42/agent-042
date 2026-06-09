"""RAG service for the gateway."""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import httpx

from gateway.schemas.openai_chat import RAGSource
from rag.embeddings import EmbeddingService
from rag.runtime import RagRuntime, RagRuntimeSource
from rag.vector_store import Document
from shared.catalog import get_catalog, get_kb_config
from shared.config import get_settings, secret_value

logger = logging.getLogger(__name__)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0

    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return -1.0
    return dot_product / (left_norm * right_norm)


class RAGService:
    """Service for retrieving relevant context using RAG.

    Each Qdrant collection is exposed via alias-based resolution:
    ``(knowledge_base, alias)`` → ``rag__<kb>__<alias>`` → Qdrant alias.
    """

    def __init__(self, settings: Any | None = None):
        """Initialize RAG service.

        Args:
            settings: Nested gateway settings object (uses cached settings if None)
        """
        if settings is None:
            settings = get_settings()

        self.settings = settings
        self.platform_settings = settings.platform
        self.gateway_settings = settings.gateway
        self.rag_settings = settings.rag

        self.enabled = self.rag_settings.rag_enabled

        # Always initialise caches so invalidate_caches() is safe even when RAG is disabled.
        self._kb_embeddings: dict[str, list[float]] = {}
        self._available_vllm_models: set[str] | None = None

        if not self.enabled:
            logger.info("RAG is disabled")
            return

        logger.info("Initializing RAG service...")

        # Initialize embedding service using config device
        self.embedding_service = EmbeddingService(
            model_name=self.rag_settings.embedding_model,
            device=self.rag_settings.embedding_device,
            batch_size=self.rag_settings.build.embedding_batch_size,
        )
        self.runtime = RagRuntime(
            settings=settings,
            embedding_service=self.embedding_service,
        )

        logger.info("RAG service initialized (retrievers will be created lazily)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def invalidate_caches(self) -> None:
        """Clear cached runtime and config-derived state.

        Called by the ``/v1/admin/reload-config`` endpoint so the next
        request re-reads catalog-derived data and runtime Qdrant attestations.
        """
        self._kb_embeddings.clear()
        self._available_vllm_models = None
        if hasattr(self, "runtime"):
            self.runtime.invalidate_caches()

    def warm_caches(self, *, validate: bool = False) -> None:
        """Best-effort eager rebuild of config-derived caches.

        Used by config reload and startup flows that want to surface cache/
        config issues earlier instead of waiting for the next request.
        """
        if not self.enabled:
            return

        if validate:
            self.validate_knowledge_bases()
        self._build_kb_embeddings()

    def _build_kb_embeddings(self) -> dict[str, list[float]]:
        if self._kb_embeddings:
            return self._kb_embeddings

        kb_items: list[tuple[str, str]] = []
        for task_cfg in get_catalog().values():
            for kb_cfg in task_cfg.knowledge_bases:
                if kb_cfg.selection_description.strip():
                    kb_items.append((kb_cfg.name, kb_cfg.selection_description))

        if not kb_items:
            self._kb_embeddings = {}
            return self._kb_embeddings

        kb_names = [kb_name for kb_name, _ in kb_items]
        descriptions = [description for _, description in kb_items]
        embeddings = self.embedding_service.embed_documents(descriptions)
        if len(embeddings) != len(kb_names):
            raise RuntimeError("KB embedding count does not match configured knowledge bases")

        self._kb_embeddings = {
            kb_name: embedding for kb_name, embedding in zip(kb_names, embeddings, strict=True)
        }
        return self._kb_embeddings

    def _vllm_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = secret_value(getattr(self.gateway_settings, "api_key", None))
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _load_available_vllm_models(self) -> set[str] | None:
        if self._available_vllm_models is not None:
            return self._available_vllm_models

        try:
            base_url = str(self.platform_settings.vllm_base_url).rstrip("/")
            timeout_s = float(getattr(self.gateway_settings, "vllm_timeout", 60.0))
            with httpx.Client(timeout=timeout_s) as client:
                response = client.get(
                    f"{base_url}/v1/models",
                    headers=self._vllm_headers(),
                )
                response.raise_for_status()
                payload = response.json()
        except Exception:
            logger.warning(
                "Failed to load vLLM models for adapter validation",
                exc_info=True,
            )
            return None

        self._available_vllm_models = {
            str(model_info["id"])
            for model_info in payload.get("data", [])
            if isinstance(model_info, dict) and model_info.get("id")
        }
        return self._available_vllm_models

    def _validate_task_adapters(self) -> None:
        available_models = self._load_available_vllm_models()
        if available_models is None:
            return

        for task_cfg in get_catalog().values():
            adapter_cfg = task_cfg.adapter
            if not adapter_cfg.enabled:
                continue

            model_name = f"{adapter_cfg.name}-{adapter_cfg.alias}"
            if model_name in available_models:
                continue

            logger.warning(
                "Enabled adapter not found in vLLM at validation time: task=%s adapter=%s; "
                "gateway will fall back to default_model until it is loaded",
                task_cfg.task,
                model_name,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def available_knowledge_bases() -> dict[str, dict[str, Any]]:
        """Return the catalog of available knowledge bases.

        Returns a dict keyed by KB name with task metadata plus ``label``,
        ``description``, ``aliases`` (full config), ``default_alias``,
        and ``update_strategy``.
        """
        result: dict[str, dict[str, Any]] = {}
        for task_cfg in get_catalog().values():
            for kb_cfg in task_cfg.knowledge_bases:
                result[kb_cfg.name] = {
                    "task": task_cfg.task,
                    "task_label": task_cfg.label,
                    "label": kb_cfg.label,
                    "description": kb_cfg.description,
                    "aliases": {
                        name: alias_cfg.model_dump() for name, alias_cfg in kb_cfg.aliases.items()
                    },
                    "default_alias": kb_cfg.default_alias,
                    "update_strategy": kb_cfg.update_strategy,
                }
        return result

    @staticmethod
    def available_knowledge_bases_by_task() -> list[dict[str, Any]]:
        """Return the catalog grouped by task for discovery endpoints."""
        result: list[dict[str, Any]] = []
        for task_cfg in get_catalog().values():
            result.append(
                {
                    "task": task_cfg.task,
                    "label": task_cfg.label,
                    "knowledge_bases": [
                        {
                            "knowledge_base": kb_cfg.name,
                            "label": kb_cfg.label,
                            "description": kb_cfg.description,
                            "aliases": {
                                name: alias_cfg.model_dump()
                                for name, alias_cfg in kb_cfg.aliases.items()
                            },
                            "default_alias": kb_cfg.default_alias,
                            "update_strategy": kb_cfg.update_strategy,
                        }
                        for kb_cfg in task_cfg.knowledge_bases
                    ],
                }
            )
        return result

    def select_knowledge_bases(self, query: str, task: str) -> list[RAGSource]:
        """Select task-scoped knowledge bases by embedding similarity."""
        if not self.enabled or not query.strip():
            return []

        task_cfg = get_catalog().get(task)
        if task_cfg is None or not task_cfg.knowledge_bases:
            return []

        kb_embeddings = self._build_kb_embeddings()
        query_embedding = self.embedding_service.embed_query(query)
        threshold = float(self.rag_settings.kb_selection_threshold)

        scored_candidates: list[tuple[float, str]] = []
        for kb_cfg in task_cfg.knowledge_bases:
            kb_embedding = kb_embeddings.get(kb_cfg.name)
            if kb_embedding is None:
                continue
            score = _cosine_similarity(query_embedding, kb_embedding)
            if score >= threshold:
                scored_candidates.append((score, kb_cfg.name))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        return [RAGSource(knowledge_base=kb_name) for _, kb_name in scored_candidates]

    def validate_knowledge_bases(self) -> None:
        """Check every alias in config resolves to a compatible Qdrant collection."""
        if not self.enabled:
            return

        self.runtime.validate_aliases(strict=self.rag_settings.rag_strict_startup)
        self._validate_task_adapters()

    def retrieve_documents(
        self,
        query: str,
        knowledge_base: Optional[str] = None,
        alias: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list:
        """Retrieve relevant documents as a list of Document objects.

        Args:
            query: User query
            knowledge_base: Knowledge base key (e.g. "arxiv", "pytorch_docs").
            alias: Alias role (uses KB's default_alias if None).
            top_k: Number of documents to retrieve (uses alias config if None)

        Returns:
            List of Document objects. Empty results only mean "no matches".
        """
        if not self.enabled:
            return []

        if not knowledge_base:
            logger.info("No knowledge base selected — skipping RAG retrieval")
            return []

        kb_cfg = get_kb_config(knowledge_base)
        if kb_cfg is None:
            logger.warning("Unknown knowledge base: %s", knowledge_base)
            return []

        if alias is None:
            alias = kb_cfg.default_alias
        alias_cfg = kb_cfg.aliases.get(alias)
        if alias_cfg is None:
            logger.warning("Invalid alias '%s' for knowledge base '%s'", alias, knowledge_base)
            return []

        try:
            runtime_result = self.runtime.retrieve(
                query=query,
                sources=[
                    RagRuntimeSource(
                        knowledge_base=knowledge_base,
                        alias=alias,
                    )
                ],
            )
            documents = [
                Document(
                    content=hit.text,
                    metadata={
                        **hit.metadata,
                        "chunk_id": hit.chunk_id,
                        "document_id": hit.document_id,
                        "source_type": hit.source_type,
                        "title": hit.title,
                        "source_uri": hit.uri,
                        "section_title": hit.section_title,
                    },
                    score=hit.score,
                )
                for hit in runtime_result.hits
            ]
            if top_k is not None:
                documents = documents[:top_k]
            logger.info(
                "Retrieved %s documents (kb=%s, alias=%s, timings_ms=%s, diagnostics=%s)",
                len(documents),
                knowledge_base,
                alias,
                runtime_result.timings_ms,
                runtime_result.diagnostics,
            )
            return documents
        except Exception as exc:
            logger.error("Error retrieving documents", exc_info=True)
            raise RuntimeError(
                "Failed to retrieve RAG documents for "
                f"knowledge base '{knowledge_base}' alias '{alias}'"
            ) from exc
