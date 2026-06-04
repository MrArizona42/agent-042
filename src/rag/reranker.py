"""Post-retrieval cross-encoder reranker.

HTTP client for the reranker microservice ``/v1/rerank`` endpoint.
"""

from __future__ import annotations

import logging

import httpx

from rag.vector_store import Document
from shared.config import get_settings

logger = logging.getLogger(__name__)


class Reranker:
    """Post-retrieval cross-encoder reranker."""

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    """HTTP client for the reranker microservice.

    Calls ``POST /v1/rerank`` with the query and document passages, overwrites
    each ``Document.score`` with the returned cross-encoder score, and returns
    the full list sorted descending by that score.  No score-threshold filtering
    is applied here — that is the caller's responsibility (``Retriever.retrieve``).
    """

    def __init__(self, reranker_url: str) -> None:
        settings = get_settings()
        base_url = reranker_url.rstrip("/")
        self._client = httpx.Client(
            base_url=base_url,
            timeout=settings.gateway.embeddings_timeout,
        )
        logger.info(f"CrossEncoderReranker connecting to {base_url}")

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        """Rerank *docs* against *query* and return the full list sorted by score.

        Args:
            query: The user query string.
            docs: Candidate documents from first-stage retrieval.
            top_k: Unused here — caller truncates after filtering by score_threshold.

        Returns:
            *docs* with ``Document.score`` replaced by cross-encoder scores,
            sorted descending.
        """
        if not docs:
            return []

        passages = [doc.content for doc in docs]
        resp = self._client.post("/v1/rerank", json={"query": query, "passages": passages})
        resp.raise_for_status()
        scores: list[float] = resp.json()["scores"]

        for doc, score in zip(docs, scores):
            doc.score = score

        return sorted(docs, key=lambda d: d.score, reverse=True)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()


def get_reranker(model_name: str) -> Reranker:
    """Factory — returns a :class:`CrossEncoderReranker` using ``settings.rag.reranker_url``.

    The *model_name* parameter is accepted for future dispatch but is currently
    unused; the reranker service reads its model from ``RERANKER_MODEL`` at startup.
    """
    settings = get_settings()
    return CrossEncoderReranker(reranker_url=settings.rag.reranker_url)
