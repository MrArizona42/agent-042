"""Embedding service for document and query vectorization."""

from __future__ import annotations

import logging
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from shared.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Manages embedding model for converting text to vectors.

    Uses sentence-transformers for efficient embedding generation.
    Model runs on CPU by default to save GPU memory for vLLM.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        """Initialize embedding service.

        Args:
            model_name: HuggingFace model identifier (uses config default if None)
            device: Device to run model on - cpu, cuda, mps (uses config default if None)
            batch_size: Batch size for embedding documents (uses config default if None)
        """
        settings = get_settings()

        self._model_name = model_name or settings.embedding_model
        self._device = device or settings.embedding_device
        self._batch_size = batch_size or settings.embedding_batch_size

        logger.info(f"Loading embedding model: {self._model_name} on device: {self._device}")
        self.model = SentenceTransformer(self._model_name, device=self._device)
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
            batch_size=self._batch_size,
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
