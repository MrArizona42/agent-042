"""Retrieval service that orchestrates vector search."""

from __future__ import annotations

import logging
from typing import List, Literal, Optional

from rag.embeddings import EmbeddingService
from rag.reranker import Reranker
from rag.vector_store import Document, QdrantVectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """High-level retrieval service for RAG pipeline."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
        reranker: Reranker | None = None,
    ):
        """Initialize retriever.

        Args:
            embedding_service: Service for generating embeddings
            vector_store: Vector database for similarity search
            reranker: Optional post-retrieval reranker
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
        strategy: Literal["dense", "hybrid", "sparse"] = "dense",
        task: Optional[str] = None,
    ) -> List[Document]:
        """Retrieve relevant documents for a query.

        Args:
            query: User query text
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score
            strategy: Retrieval strategy (only ``"dense"`` is implemented)
            task: Task type for filtering (chat, code, summarize)

        Returns:
            List of relevant documents with scores
        """
        if not query.strip():
            logger.warning("Empty query provided to retriever")
            return []

        if strategy != "dense":
            raise NotImplementedError(f"retrieval_strategy '{strategy}' not yet implemented")

        # Embed query
        logger.info(f"Embedding query: {query[:100]}...")
        query_embedding = self.embedding_service.embed_query(query)

        # Build filter if task specified
        filter_dict = None
        if task:
            filter_dict = {"must": [{"key": "task", "match": {"value": task}}]}

        # Search vector store
        logger.info(f"Searching for top {top_k} documents (threshold={score_threshold})")
        candidates = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_dict=filter_dict,
        )

        if self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates, top_k)

        logger.info(f"Retrieved {len(candidates)} documents")
        return candidates[:top_k]

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
