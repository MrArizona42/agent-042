"""Embedding client for document and query vectorization.

Calls the standalone embeddings microservice over HTTP instead of loading
the model in-process, keeping heavy dependencies (PyTorch,
sentence-transformers) out of the gateway and Airflow containers.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import httpx

from app_config.runtime import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """HTTP client for the embeddings microservice.

    Drop-in replacement for the previous local-model implementation.
    The public interface (embed_documents, embed_query, dimension) is
    unchanged so that callers (Retriever, RAGService, build scripts) work
    without modification.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        embeddings_url: Optional[str] = None,
    ):
        """Initialize the embedding client.

        Args:
            model_name: Ignored (kept for backwards compatibility).
            device: Ignored (kept for backwards compatibility).
            batch_size: Ignored (kept for backwards compatibility).
            embeddings_url: Override for the embeddings service URL.
        """
        settings = get_settings()
        base_url = (embeddings_url or settings.platform.embeddings_url).rstrip("/")
        self._client = httpx.Client(
            base_url=base_url,
            timeout=settings.gateway.embeddings_timeout,
        )

        logger.info(f"Connecting to embeddings service at {base_url}")
        resp = self._client.get("/v1/dimension")
        resp.raise_for_status()
        data = resp.json()
        self.dimension: int = data["dimension"]
        logger.info(f"Embedding dimension: {self.dimension} (model: {data.get('model')})")

    # Maximum number of texts sent per HTTP request.  Large inputs (e.g.
    # eval corpora with 5 000+ docs) are split into chunks of this size so
    # individual requests stay well within the configured timeout.
    _EMBED_BATCH = 512

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        results: List[List[float]] = []
        for start in range(0, len(texts), self._EMBED_BATCH):
            batch = texts[start : start + self._EMBED_BATCH]
            resp = self._client.post("/v1/embeddings", json={"input": batch})
            resp.raise_for_status()
            data = resp.json()
            results.extend(item["embedding"] for item in data["data"])
            logger.debug(f"Embedded batch {start} to {start + len(batch)} / {len(texts)}")
        return results

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        resp = self._client.post("/v1/embeddings", json={"input": [text]})
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
