"""RAG service for the gateway."""
from __future__ import annotations

import logging
from typing import Optional

from gateway.config import GatewaySettings
from rag.config import RAGSettings
from rag.embeddings import EmbeddingService
from rag.retriever import Retriever
from rag.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class RAGService:
    """Service for retrieving relevant context using RAG."""

    def __init__(self, settings: GatewaySettings):
        """Initialize RAG service.

        Args:
            settings: Gateway settings containing RAG configuration
        """
        self.enabled = settings.rag_enabled

        if not self.enabled:
            logger.info("RAG is disabled")
            return

        logger.info("Initializing RAG service...")

        # Initialize embedding service
        self.embedding_service = EmbeddingService(
            model_name=settings.embedding_model,
            device="cpu",  # Run on CPU to save GPU for vLLM
        )

        # Initialize retrievers for each task
        self.retrievers = {}
        for task in ["chat", "code"]:
            collection_name = f"{task}_documents"
            vector_store = QdrantVectorStore(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                collection_name=collection_name,
            )

            # Check if collection exists
            if not vector_store.collection_exists():
                logger.warning(f"Collection '{collection_name}' does not exist. Retrieval for task '{task}' will be disabled.")
                continue

            rag_settings = RAGSettings(
                qdrant_host=settings.qdrant_host,
                qdrant_port=settings.qdrant_port,
                embedding_model=settings.embedding_model,
            )

            self.retrievers[task] = Retriever(
                embedding_service=self.embedding_service,
                vector_store=vector_store,
                settings=rag_settings,
            )

        logger.info(f"RAG service initialized. Available tasks: {list(self.retrievers.keys())}")

    def retrieve_context(
        self,
        query: str,
        task: str = "chat",
        top_k: int = 5,
    ) -> Optional[str]:
        """Retrieve relevant context for a query.

        Args:
            query: User query
            task: Task type (chat, code, summarize)
            top_k: Number of documents to retrieve

        Returns:
            Formatted context string or None if RAG is disabled/unavailable
        """
        if not self.enabled:
            return None

        # Map summarize task to chat collection
        if task == "summarize":
            task = "chat"

        # Check if retriever exists for this task
        if task not in self.retrievers:
            logger.warning(f"No retriever available for task: {task}")
            return None

        try:
            # Retrieve documents
            documents = self.retrievers[task].retrieve(
                query=query,
                top_k=top_k,
                task=task,
            )

            if not documents:
                logger.info("No relevant documents found")
                return None

            # Format context
            context = self.retrievers[task].format_context(documents, max_length=3000)
            logger.info(f"Retrieved context of {len(context)} characters from {len(documents)} documents")

            return context

        except Exception as e:
            logger.error(f"Error retrieving context: {e}", exc_info=True)
            return None
