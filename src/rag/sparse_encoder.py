"""Sparse vector encoder (BM25).

HTTP client for the embeddings microservice ``/v1/sparse-embeddings`` endpoint.
Mirrors the structure of :class:`rag.embeddings.EmbeddingService` so callers
can treat both clients the same way.
"""

from __future__ import annotations

import logging

import httpx
from qdrant_client.models import SparseVector

from app_config.runtime import get_settings

logger = logging.getLogger(__name__)


class SparseEncoderService:
    """HTTP client for sparse (BM25) vector encoding via the embeddings microservice."""

    # Maximum number of texts sent per HTTP request — mirrors EmbeddingService._EMBED_BATCH.
    _ENCODE_BATCH = 512

    def __init__(self, embeddings_url: str | None = None) -> None:
        settings = get_settings()
        base_url = (embeddings_url or settings.platform.embeddings_url).rstrip("/")
        self._client = httpx.Client(
            base_url=base_url,
            timeout=settings.gateway.embeddings_timeout,
        )
        logger.info(f"SparseEncoderService connecting to {base_url}")

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        """Encode a list of documents into sparse vectors.

        Args:
            texts: Document texts to encode.

        Returns:
            List of :class:`qdrant_client.models.SparseVector` in the same order.
        """
        if not texts:
            return []

        results: list[SparseVector] = []
        for start in range(0, len(texts), self._ENCODE_BATCH):
            batch = texts[start : start + self._ENCODE_BATCH]
            resp = self._client.post("/v1/sparse-embeddings", json={"input": batch})
            resp.raise_for_status()
            data = resp.json()
            results.extend(
                SparseVector(indices=item["indices"], values=item["values"])
                for item in data["data"]
            )
            logger.debug(f"Sparse-encoded batch {start}–{start + len(batch)} / {len(texts)}")
        return results

    def encode_query(self, text: str) -> SparseVector:
        """Encode a single query text into a sparse vector.

        Args:
            text: Query text to encode.

        Returns:
            :class:`qdrant_client.models.SparseVector` for the query.
        """
        resp = self._client.post("/v1/sparse-embeddings", json={"input": [text]})
        resp.raise_for_status()
        item = resp.json()["data"][0]
        return SparseVector(indices=item["indices"], values=item["values"])

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
