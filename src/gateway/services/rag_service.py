"""RAG service for the gateway."""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

from gateway.config import get_settings
from gateway.schemas.openai_chat import RAGSource
from rag.embeddings import EmbeddingService
from rag.ops.meta import BuildConfig, read_collection_meta, validate_query_compatibility
from rag.reranker import Reranker, get_reranker
from rag.retriever import Retriever
from rag.sparse_encoder import SparseEncoderService
from rag.vector_store import QdrantVectorStore
from shared.config import Settings, get_kb_config, get_knowledge_bases

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
    ``(knowledge_base, alias)`` → ``{kb}_{alias}`` → Qdrant alias.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize RAG service.

        Args:
            settings: Settings instance (uses cached settings if None)
        """
        if settings is None:
            settings = get_settings()

        self.settings = settings
        self.enabled = settings.rag_enabled

        # Always initialise caches so invalidate_caches() is safe even
        # when RAG is disabled.
        self._retrievers: dict[str, Retriever] = {}
        self._build_configs: dict[str, BuildConfig] = {}
        self._resolved_collections: dict[str, str] = {}
        self._unavailable: set[str] = set()
        self._kb_embeddings: dict[str, list[float]] = {}

        if not self.enabled:
            logger.info("RAG is disabled")
            return

        logger.info("Initializing RAG service...")

        # Initialize embedding service using config device
        self.embedding_service = EmbeddingService(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        )

        logger.info("RAG service initialized (retrievers will be created lazily)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def invalidate_caches(self) -> None:
        """Clear cached retrievers and build configs.

        Called by the ``/v1/admin/reload-config`` endpoint so the next
        request per alias re-reads ``BuildConfig`` from Qdrant ``_meta``
        and re-creates the retriever.
        """
        self._retrievers.clear()
        self._build_configs.clear()
        self._resolved_collections.clear()
        self._unavailable.clear()
        self._kb_embeddings.clear()

    def _build_kb_embeddings(self) -> dict[str, list[float]]:
        if self._kb_embeddings:
            return self._kb_embeddings

        kb_items: list[tuple[str, str]] = []
        for task_cfg in get_knowledge_bases().values():
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

    @staticmethod
    def _qdrant_alias(kb_name: str, alias: str) -> str:
        """Construct the Qdrant alias name: ``{kb}_{alias}``."""
        return f"{kb_name}_{alias}"

    def _mark_alias_unavailable(
        self,
        cache_key: str,
        message: str,
        *,
        strict: bool,
        exc: Exception | None = None,
    ) -> None:
        """Invalidate cached state for an alias and optionally raise."""
        self._retrievers.pop(cache_key, None)
        self._resolved_collections.pop(cache_key, None)
        self._unavailable.add(cache_key)
        if strict:
            raise RuntimeError(message) from exc
        logger.warning("%s — marking unavailable", message)

    def _set_resolved_collection(self, cache_key: str, resolved_collection: str) -> None:
        """Track which physical collection an alias currently resolves to."""
        previous = self._resolved_collections.get(cache_key)
        if previous is not None and previous != resolved_collection:
            logger.info(
                "Alias '%s' changed target from '%s' to '%s'; refreshing cached retriever",
                cache_key,
                previous,
                resolved_collection,
            )
            self._retrievers.pop(cache_key, None)
        self._resolved_collections[cache_key] = resolved_collection

    def _ensure_build_config(
        self,
        cache_key: str,
        collection_cache_key: str,
        vector_store: QdrantVectorStore,
        *,
        query_strategy: str,
        strict: bool,
    ) -> BuildConfig | None:
        """Read and cache BuildConfig for an alias, enforcing runtime compatibility."""
        build_config = self._build_configs.get(collection_cache_key)
        if build_config is None:
            try:
                meta = read_collection_meta(vector_store, context=collection_cache_key)
            except Exception as exc:
                self._mark_alias_unavailable(
                    cache_key,
                    (
                        f"Failed to read _meta for collection '{collection_cache_key}' "
                        f"behind alias '{cache_key}': {exc}"
                    ),
                    strict=strict,
                    exc=exc,
                )
                return None

            collection_info = vector_store.get_collection_info()
            vector_size = (
                collection_info.get("vector_size") if isinstance(collection_info, dict) else None
            )
            runtime_dimension = getattr(self.embedding_service, "dimension", None)
            if (
                isinstance(vector_size, int)
                and isinstance(runtime_dimension, int)
                and vector_size != runtime_dimension
            ):
                self._mark_alias_unavailable(
                    cache_key,
                    (
                        f"Embedding dimension mismatch for collection '{collection_cache_key}' "
                        f"(alias '{cache_key}'): collection={vector_size}, "
                        f"runtime={runtime_dimension}, "
                        f"build_embedding_model={meta.build_config.embedding_model}"
                    ),
                    strict=strict,
                )
                return None

            build_config = meta.build_config
            self._build_configs[collection_cache_key] = build_config
            logger.info(
                "Cached build config for physical collection %s (embedding_model=%s)",
                collection_cache_key,
                build_config.embedding_model,
            )

        try:
            validate_query_compatibility(
                query_strategy=query_strategy,
                build_config=build_config,
                runtime_sparse_encoder=getattr(self.settings, "sparse_encoder_model", None),
                context=f"{cache_key} -> {collection_cache_key}",
            )
        except ValueError as exc:
            self._mark_alias_unavailable(
                cache_key,
                str(exc),
                strict=strict,
                exc=exc,
            )
            return None

        self._unavailable.discard(cache_key)
        return build_config

    def _get_retriever(self, kb_name: str, alias: str) -> Optional[Retriever]:
        """Return (and lazily create) a retriever for *(kb_name, alias)*.

        Validates that the KB exists in the registry and that the alias
        is in the allowed list before attempting to connect.
        """
        cache_key = self._qdrant_alias(kb_name, alias)

        kb_cfg = get_kb_config(kb_name)
        if kb_cfg is None:
            return None
        if alias not in kb_cfg.aliases:
            return None

        qdrant_alias_name = cache_key
        alias_cfg = kb_cfg.aliases.get(alias)
        if alias_cfg is None:
            return None

        alias_store = QdrantVectorStore(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
            collection_name=qdrant_alias_name,
        )

        if not alias_store.collection_exists():
            if cache_key not in self._unavailable:
                logger.warning(
                    f"Qdrant alias '{qdrant_alias_name}' does not resolve. "
                    f"Knowledge base '{kb_name}' alias '{alias}' is not available."
                )
                self._unavailable.add(cache_key)
            return None

        resolved_collection = alias_store.resolve_alias(qdrant_alias_name)
        if not isinstance(resolved_collection, str) or not resolved_collection:
            resolved_collection = qdrant_alias_name
        self._set_resolved_collection(cache_key, resolved_collection)

        if cache_key in self._retrievers:
            return self._retrievers[cache_key]

        vector_store = alias_store
        if resolved_collection != qdrant_alias_name:
            vector_store = QdrantVectorStore(
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
                collection_name=resolved_collection,
            )

        build_cfg = self._ensure_build_config(
            cache_key,
            resolved_collection,
            vector_store,
            query_strategy=alias_cfg.retrieval_strategy,
            strict=False,
        )
        if build_cfg is None:
            return None

        reranker: Reranker | None = None
        if alias_cfg and alias_cfg.reranker:
            reranker = get_reranker(alias_cfg.reranker)

        sparse_encoder_service: SparseEncoderService | None = None
        if alias_cfg and alias_cfg.retrieval_strategy in {"hybrid", "sparse"}:
            sparse_encoder_service = SparseEncoderService(
                embeddings_url=self.settings.embeddings_url
            )

        retriever = Retriever(
            embedding_service=self.embedding_service,
            vector_store=vector_store,
            reranker=reranker,
            sparse_encoder_service=sparse_encoder_service,
            reranker_multiplier=alias_cfg.reranker_multiplier if alias_cfg else 1,
        )
        self._retrievers[cache_key] = retriever
        logger.info(
            f"Retriever for '{kb_name}' alias '{alias}' is now available "
            f"(Qdrant alias: {qdrant_alias_name}, target: {resolved_collection})"
        )
        return retriever

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def available_knowledge_bases() -> dict[str, dict[str, Any]]:
        """Return the registry of available knowledge bases.

        Returns a dict keyed by KB name with task metadata plus ``label``,
        ``description``, ``aliases`` (full config), ``default_alias``,
        and ``update_strategy``.
        """
        result: dict[str, dict[str, Any]] = {}
        for task_cfg in get_knowledge_bases().values():
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
        """Return the registry grouped by task for discovery endpoints."""
        result: list[dict[str, Any]] = []
        for task_cfg in get_knowledge_bases().values():
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

        task_cfg = get_knowledge_bases().get(task)
        if task_cfg is None or not task_cfg.knowledge_bases:
            return []

        kb_embeddings = self._build_kb_embeddings()
        query_embedding = self.embedding_service.embed_query(query)
        threshold = float(self.settings.kb_selection_threshold)

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
        """Check every alias in config resolves to an existing Qdrant collection.

        For each resolvable alias, reads ``BuildConfig`` from the ``_meta``
        sentinel point and caches it in ``self._build_configs``.  When
        ``rag_strict_startup`` is ``True``, missing collections, invalid
        ``_meta``, or embedding dimension mismatches cause startup to raise
        instead of just logging.
        """
        if not self.enabled:
            return

        strict = self.settings.rag_strict_startup

        for task_cfg in get_knowledge_bases().values():
            for kb_cfg in task_cfg.knowledge_bases:
                for alias, alias_cfg in kb_cfg.aliases.items():
                    cache_key = self._qdrant_alias(kb_cfg.name, alias)
                    alias_store = QdrantVectorStore(
                        host=self.settings.qdrant_host,
                        port=self.settings.qdrant_port,
                        collection_name=cache_key,
                    )

                    if not alias_store.collection_exists():
                        msg = (
                            f"KB alias not found in Qdrant at startup: "
                            f"task={task_cfg.task} kb={kb_cfg.name} "
                            f"alias={alias} collection={cache_key}"
                        )
                        self._mark_alias_unavailable(cache_key, msg, strict=strict)
                        continue

                    resolved_collection = alias_store.resolve_alias(cache_key)
                    if not isinstance(resolved_collection, str) or not resolved_collection:
                        resolved_collection = cache_key
                    self._set_resolved_collection(cache_key, resolved_collection)
                    vector_store = alias_store
                    if resolved_collection != cache_key:
                        vector_store = QdrantVectorStore(
                            host=self.settings.qdrant_host,
                            port=self.settings.qdrant_port,
                            collection_name=resolved_collection,
                        )

                    self._ensure_build_config(
                        cache_key,
                        resolved_collection,
                        vector_store,
                        query_strategy=alias_cfg.retrieval_strategy,
                        strict=strict,
                    )

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
        if top_k is None:
            top_k = alias_cfg.top_k
        score_threshold = alias_cfg.score_threshold
        cache_key = self._qdrant_alias(knowledge_base, alias)

        retriever = self._get_retriever(knowledge_base, alias)
        if retriever is None:
            raise RuntimeError(
                f"RAG retriever unavailable for knowledge base '{knowledge_base}' alias '{alias}'"
            )

        try:
            resolved_collection = self._resolved_collections.get(cache_key)
            if resolved_collection is None or resolved_collection not in self._build_configs:
                raise RuntimeError(
                    "RAG build config unavailable for "
                    f"knowledge base '{knowledge_base}' alias '{alias}'"
                )

            documents = retriever.retrieve(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
                strategy=alias_cfg.retrieval_strategy,
            )
            logger.info(
                f"Retrieved {len(documents)} documents (kb={knowledge_base}, alias={alias})"
            )
            return documents
        except Exception as exc:
            logger.error("Error retrieving documents", exc_info=True)
            raise RuntimeError(
                "Failed to retrieve RAG documents for "
                f"knowledge base '{knowledge_base}' alias '{alias}'"
            ) from exc
