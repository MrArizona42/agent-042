"""Embedding service for document and query vectorization."""
from __future__ import annotations

import logging
from typing import List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Manages embedding model for converting text to vectors.

    Uses sentence-transformers for efficient embedding generation.
    Model runs on CPU by default to save GPU memory for vLLM.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        """Initialize embedding service.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on (cpu, cuda, mps)
        """
        logger.info(f"Loading embedding model: {model_name} on device: {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
        )
        return embedding.tolist()
