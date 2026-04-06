"""Post-retrieval cross-encoder reranker. Not yet implemented."""

from __future__ import annotations

from rag.vector_store import Document


class Reranker:
    """Post-retrieval cross-encoder reranker."""

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        raise NotImplementedError


def get_reranker(model_name: str) -> Reranker:
    """Factory — mirrors get_chunker(). Implement model dispatch here."""
    raise NotImplementedError(f"Reranker '{model_name}' not yet implemented")
