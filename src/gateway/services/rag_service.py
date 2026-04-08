"""RAG service for the gateway."""

from __future__ import annotations

import logging
from typing import Optional

from gateway.config import get_settings
from rag.embeddings import EmbeddingService
from rag.ops.meta import BuildConfig, read_collection_meta
from rag.reranker import Reranker, get_reranker
from rag.retriever import Retriever
from rag.vector_store import QdrantVectorStore
from shared.config import Settings, get_kb_config, get_knowledge_bases

logger = logging.getLogger(__name__)


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
        self._unavailable: set[str] = set()

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
        self._unavailable.clear()

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
        self._build_configs.pop(cache_key, None)
        self._unavailable.add(cache_key)
        if strict:
            raise RuntimeError(message) from exc
        logger.warning("%s — marking unavailable", message)

    def _ensure_build_config(
        self,
        cache_key: str,
        vector_store: QdrantVectorStore,
        *,
        strict: bool,
    ) -> BuildConfig | None:
        """Read and cache BuildConfig for an alias, enforcing runtime compatibility."""
        cached = self._build_configs.get(cache_key)
        if cached is not None:
            return cached

        try:
            meta = read_collection_meta(vector_store, context=cache_key)
        except Exception as exc:
            self._mark_alias_unavailable(
                cache_key,
                f"Failed to read _meta for collection '{cache_key}': {exc}",
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
                    f"Embedding dimension mismatch for collection '{cache_key}': "
                    f"collection={vector_size}, runtime={runtime_dimension}, "
                    f"build_embedding_model={meta.build_config.embedding_model}"
                ),
                strict=strict,
            )
            return None

        self._unavailable.discard(cache_key)
        self._build_configs[cache_key] = meta.build_config
        logger.info(
            "Cached build config for %s (embedding_model=%s)",
            cache_key,
            meta.build_config.embedding_model,
        )
        return meta.build_config

    def _get_retriever(self, kb_name: str, alias: str) -> Optional[Retriever]:
        """Return (and lazily create) a retriever for *(kb_name, alias)*.

        Validates that the KB exists in the registry and that the alias
        is in the allowed list before attempting to connect.
        """
        cache_key = self._qdrant_alias(kb_name, alias)

        if cache_key in self._retrievers:
            return self._retrievers[cache_key]

        kb_cfg = get_kb_config(kb_name)
        if kb_cfg is None:
            return None
        if alias not in kb_cfg.aliases:
            return None

        qdrant_alias_name = cache_key

        vector_store = QdrantVectorStore(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
            collection_name=qdrant_alias_name,
        )

        if not vector_store.collection_exists():
            if cache_key not in self._unavailable:
                logger.warning(
                    f"Qdrant alias '{qdrant_alias_name}' does not resolve. "
                    f"Knowledge base '{kb_name}' alias '{alias}' is not available."
                )
                self._unavailable.add(cache_key)
            return None

        build_cfg = self._ensure_build_config(cache_key, vector_store, strict=False)
        if build_cfg is None:
            return None

        alias_cfg = kb_cfg.aliases.get(alias)
        reranker: Reranker | None = None
        if alias_cfg and alias_cfg.reranker:
            reranker = get_reranker(alias_cfg.reranker)

        retriever = Retriever(
            embedding_service=self.embedding_service,
            vector_store=vector_store,
            reranker=reranker,
        )
        self._retrievers[cache_key] = retriever
        logger.info(
            f"Retriever for '{kb_name}' alias '{alias}' is now available "
            f"(Qdrant alias: {qdrant_alias_name})"
        )
        return retriever

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def available_knowledge_bases() -> dict[str, dict]:
        """Return the registry of available knowledge bases.

        Returns a dict keyed by KB name with ``label``, ``description``,
        ``aliases`` (full config), ``default_alias``, and ``update_strategy``.
        """
        result: dict[str, dict] = {}
        for task_cfg in get_knowledge_bases().values():
            for kb_cfg in task_cfg.knowledge_bases:
                result[kb_cfg.name] = {
                    "label": kb_cfg.label,
                    "description": kb_cfg.description,
                    "aliases": {
                        name: alias_cfg.model_dump() for name, alias_cfg in kb_cfg.aliases.items()
                    },
                    "default_alias": kb_cfg.default_alias,
                    "update_strategy": kb_cfg.update_strategy,
                }
        return result

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
                for alias in kb_cfg.aliases:
                    cache_key = self._qdrant_alias(kb_cfg.name, alias)
                    vs = QdrantVectorStore(
                        host=self.settings.qdrant_host,
                        port=self.settings.qdrant_port,
                        collection_name=cache_key,
                    )

                    if not vs.collection_exists():
                        msg = (
                            f"KB alias not found in Qdrant at startup: "
                            f"task={task_cfg.task} kb={kb_cfg.name} "
                            f"alias={alias} collection={cache_key}"
                        )
                        self._mark_alias_unavailable(cache_key, msg, strict=strict)
                        continue

                    self._ensure_build_config(cache_key, vs, strict=strict)

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
            List of Document objects, or empty list if unavailable.
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

        retriever = self._get_retriever(knowledge_base, alias)
        if retriever is None:
            logger.warning(
                f"No retriever available for knowledge base: {knowledge_base} alias: {alias}"
            )
            return []

        try:
            cache_key = self._qdrant_alias(knowledge_base, alias)
            build_cfg = self._build_configs.get(cache_key)
            if build_cfg is None:
                logger.warning(
                    "No build config cached for knowledge base: %s alias: %s",
                    knowledge_base,
                    alias,
                )
                return []

            documents = retriever.retrieve(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold,
                strategy=build_cfg.retrieval_strategy,
            )
            logger.info(
                f"Retrieved {len(documents)} documents (kb={knowledge_base}, alias={alias})"
            )
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}", exc_info=True)
            return []
