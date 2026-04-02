"""RAG service for the gateway."""

from __future__ import annotations

import logging
from typing import Optional

from gateway.config import get_settings
from rag.embeddings import EmbeddingService
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

        # Retrievers are created lazily on first request for each (kb, alias).
        # This avoids a startup race when Qdrant is not yet ready.
        self._retrievers: dict[str, Retriever] = {}
        self._unavailable: set[str] = set()

        logger.info("RAG service initialized (retrievers will be created lazily)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _qdrant_alias(kb_name: str, alias: str) -> str:
        """Construct the Qdrant alias name: ``{kb}_{alias}``."""
        return f"{kb_name}_{alias}"

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

        # Alias resolved — create the retriever and cache it
        self._unavailable.discard(cache_key)
        retriever = Retriever(
            embedding_service=self.embedding_service,
            vector_store=vector_store,
            settings=self.settings,
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
        ``aliases``, and ``update_strategy``.
        """
        result: dict[str, dict] = {}
        for task_cfg in get_knowledge_bases().values():
            for kb_cfg in task_cfg.knowledge_bases:
                result[kb_cfg.name] = {
                    "label": kb_cfg.label,
                    "description": kb_cfg.description,
                    "aliases": kb_cfg.aliases,
                    "update_strategy": kb_cfg.update_strategy,
                }
        return result

    def validate_knowledge_bases(self) -> None:
        """Check every alias in config resolves to an existing Qdrant collection.

        Logs warnings for missing collections; does not raise.
        """
        if not self.enabled:
            return

        for task_cfg in get_knowledge_bases().values():
            for kb_cfg in task_cfg.knowledge_bases:
                for alias in kb_cfg.aliases:
                    collection = self._qdrant_alias(kb_cfg.name, alias)
                    vs = QdrantVectorStore(
                        host=self.settings.qdrant_host,
                        port=self.settings.qdrant_port,
                        collection_name=collection,
                    )
                    if not vs.collection_exists():
                        logger.warning(
                            "KB alias not found in Qdrant at startup: "
                            "task=%s kb=%s alias=%s collection=%s — marking unavailable",
                            task_cfg.task, kb_cfg.name, alias, collection,
                        )
                        self._unavailable.add(collection)

    def retrieve_context(
        self,
        query: str,
        knowledge_base: Optional[str] = None,
        alias: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Optional[str]:
        """Retrieve relevant context for a query.

        Args:
            query: User query
            knowledge_base: Knowledge base key (e.g. "arxiv", "pytorch_docs").
                If None the retrieval is skipped.
            alias: Alias role (uses settings.default_alias if None).
            top_k: Number of documents to retrieve (uses config default if None)

        Returns:
            Formatted context string or None if RAG is disabled/unavailable
        """
        if alias is None:
            alias = self.settings.default_alias
        if top_k is None:
            top_k = self.settings.top_k
        docs = self.retrieve_documents(
            query=query,
            knowledge_base=knowledge_base,
            alias=alias,
            top_k=top_k,
        )
        if not docs:
            return None
        return self.format_documents(docs)

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
            alias: Alias role (uses settings.default_alias if None).
            top_k: Number of documents to retrieve (uses config default if None)

        Returns:
            List of Document objects, or empty list if unavailable.
        """
        if not self.enabled:
            return []

        if alias is None:
            alias = self.settings.default_alias
        if top_k is None:
            top_k = self.settings.top_k

        if not knowledge_base:
            logger.info("No knowledge base selected — skipping RAG retrieval")
            return []

        retriever = self._get_retriever(knowledge_base, alias)
        if retriever is None:
            logger.warning(
                f"No retriever available for knowledge base: {knowledge_base} alias: {alias}"
            )
            return []

        try:
            documents = retriever.retrieve(query=query, top_k=top_k)
            logger.info(
                f"Retrieved {len(documents)} documents (kb={knowledge_base}, alias={alias})"
            )
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}", exc_info=True)
            return []

    def format_documents(self, documents: list) -> Optional[str]:
        """Format retrieved documents into a context string.

        Args:
            documents: List of Document objects.

        Returns:
            Formatted context string or None if no documents.
        """
        if not documents:
            return None

        parts: list[str] = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "unknown")
            score = doc.score if doc.score is not None else 0.0
            parts.append(f"[Document {i}] (Source: {source}, Score: {score:.3f})\n{doc.content}")

        context = "\n\n".join(parts)
        max_len = self.settings.context_max_length
        if len(context) > max_len:
            context = context[:max_len]
        return context
