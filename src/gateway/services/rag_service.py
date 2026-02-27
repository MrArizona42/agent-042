"""RAG service for the gateway."""

from __future__ import annotations

import logging
from typing import Optional

from gateway.config import get_settings
from rag.embeddings import EmbeddingService
from rag.retriever import Retriever
from rag.vector_store import QdrantVectorStore
from shared.config import KNOWLEDGE_BASES, Settings

logger = logging.getLogger(__name__)


class RAGService:
    """Service for retrieving relevant context using RAG.

    Each Qdrant collection is exposed as a named knowledge base that the
    user can select explicitly from the UI.
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

        # Retrievers are created lazily on first request for each KB.
        # This avoids a startup race when Qdrant is not yet ready.
        self.retrievers: dict[str, Retriever] = {}
        self._unavailable_kbs: set[str] = set()

        logger.info("RAG service initialized (retrievers will be created lazily)")

    # ------------------------------------------------------------------
    def _get_retriever(self, kb_name: str) -> Optional[Retriever]:
        """Return (and lazily create) a retriever for *kb_name*."""
        if kb_name in self.retrievers:
            return self.retrievers[kb_name]

        if kb_name not in KNOWLEDGE_BASES:
            return None

        # If we already tried and the collection was missing, retry —
        # it may have been created since last attempt.
        kb_info = KNOWLEDGE_BASES[kb_name]
        collection_name = kb_info["collection"]

        vector_store = QdrantVectorStore(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
            collection_name=collection_name,
        )

        if not vector_store.collection_exists():
            if kb_name not in self._unavailable_kbs:
                logger.warning(
                    f"Collection '{collection_name}' does not exist. "
                    f"Knowledge base '{kb_name}' is not available yet."
                )
                self._unavailable_kbs.add(kb_name)
            return None

        # Collection appeared — create the retriever and cache it
        self._unavailable_kbs.discard(kb_name)
        retriever = Retriever(
            embedding_service=self.embedding_service,
            vector_store=vector_store,
            settings=self.settings,
        )
        self.retrievers[kb_name] = retriever
        logger.info(f"Retriever for knowledge base '{kb_name}' is now available")
        return retriever

    @staticmethod
    def available_knowledge_bases() -> dict[str, dict[str, str]]:
        """Return the registry of available knowledge bases."""
        return KNOWLEDGE_BASES

    def retrieve_context(
        self,
        query: str,
        knowledge_base: Optional[str] = None,
        top_k: int = 5,
    ) -> Optional[str]:
        """Retrieve relevant context for a query.

        Args:
            query: User query
            knowledge_base: Knowledge base key (e.g. "arxiv", "pytorch_docs").
                If None the retrieval is skipped.
            top_k: Number of documents to retrieve

        Returns:
            Formatted context string or None if RAG is disabled/unavailable
        """
        if not self.enabled:
            return None

        if not knowledge_base:
            logger.info("No knowledge base selected — skipping RAG retrieval")
            return None

        # Check if retriever exists for this knowledge base
        retriever = self._get_retriever(knowledge_base)
        if retriever is None:
            logger.warning(f"No retriever available for knowledge base: {knowledge_base}")
            return None

        try:
            # Retrieve documents
            documents = retriever.retrieve(
                query=query,
                top_k=top_k,
            )

            if not documents:
                logger.info("No relevant documents found")
                return None

            # Format context using configured max length
            context = retriever.format_context(
                documents,
                max_length=self.settings.context_max_length,
            )
            logger.info(
                f"Retrieved context of {len(context)} characters "
                f"from {len(documents)} documents (kb={knowledge_base})"
            )

            return context

        except Exception as e:
            logger.error(f"Error retrieving context: {e}", exc_info=True)
            return None
