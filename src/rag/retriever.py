"""Retrieval service that orchestrates vector search."""

from __future__ import annotations

import logging
from typing import List, Optional

from rag.embeddings import EmbeddingService
from rag.vector_store import Document, QdrantVectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """High-level retrieval service for RAG pipeline."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
    ):
        """Initialize retriever.

        Args:
            embedding_service: Service for generating embeddings
            vector_store: Vector database for similarity search
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        task: Optional[str] = None,
    ) -> List[Document]:
        """Retrieve relevant documents for a query.

        Args:
            query: User query text
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score
            task: Task type for filtering (chat, code, summarize)

        Returns:
            List of relevant documents with scores
        """
        if not query.strip():
            logger.warning("Empty query provided to retriever")
            return []

        # Embed query
        logger.info(f"Embedding query: {query[:100]}...")
        query_embedding = self.embedding_service.embed_query(query)

        # Build filter if task specified
        filter_dict = None
        if task:
            filter_dict = {"must": [{"key": "task", "match": {"value": task}}]}

        # Search vector store
        logger.info(f"Searching for top {top_k} documents (threshold={score_threshold})")
        documents = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_dict=filter_dict,
        )

        logger.info(f"Retrieved {len(documents)} documents")
        return documents

    def format_context(self, documents: List[Document], max_length: int) -> str:
        """Format retrieved documents into context string.

        Args:
            documents: Retrieved documents
            max_length: Maximum character length of context

        Returns:
            Formatted context string
        """
        if not documents:
            return ""

        context_parts = []
        current_length = 0

        for i, doc in enumerate(documents, 1):
            # Format document with metadata
            source = doc.metadata.get("source", "unknown")
            doc_text = f"[Document {i}] (Source: {source}, Score: {doc.score:.3f})\n{doc.content}\n"

            # Check if adding this document exceeds max length
            if current_length + len(doc_text) > max_length:
                logger.info(
                    f"Context length limit reached, using {i - 1}/{len(documents)} documents"
                )
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return "\n".join(context_parts)
