"""LlamaIndex embedding/sparse-encoder adapters over the project's embedding services.

These wrap the existing HTTP clients (``rag.embeddings.EmbeddingService`` and
``rag.sparse_encoder.SparseEncoderService``) so LlamaIndex's ``VectorStoreIndex``
and Qdrant vector store can use them directly, without LlamaIndex ever loading
a model in-process.
"""

from __future__ import annotations

from typing import Any, Protocol

from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.embeddings import BaseEmbedding
from pydantic import Field
from qdrant_client.models import SparseVector


class EmbeddingServiceProtocol(Protocol):
    """Dense embedding client contract used by :class:`ProjectEmbedding`."""

    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document texts."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        ...


class SparseEncoderServiceProtocol(Protocol):
    """Sparse encoder client contract used by :class:`ProjectSparseEncoder`."""

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        """Encode texts as sparse vectors."""
        ...


class ProjectEmbedding(BaseEmbedding):
    """LlamaIndex embedding adapter over the current embedding service."""

    embedding_client: Any = Field(exclude=True)

    def __init__(
        self,
        *,
        embedding_client: EmbeddingServiceProtocol,
        model_name: str = "project-embedding-service",
        **kwargs: Any,
    ) -> None:
        super().__init__(embedding_client=embedding_client, model_name=model_name, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "ProjectEmbedding"

    def _get_query_embedding(self, query: str) -> Embedding:
        return self.embedding_client.embed_query(query)

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self.embedding_client.embed_query(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self.embedding_client.embed_documents([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self.embedding_client.embed_documents(texts)

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self._get_text_embeddings(texts)


class ProjectSparseEncoder:
    """Adapter exposing ``sparse_doc_fn``/``sparse_query_fn`` callables.

    LlamaIndex's Qdrant vector store calls both with a batch of texts and
    expects ``(indices_per_text, values_per_text)`` back. The project's BM25
    sparse encoder treats documents and queries identically, so both
    callables delegate to the same client method.
    """

    def __init__(self, sparse_encoder_client: SparseEncoderServiceProtocol) -> None:
        self._client = sparse_encoder_client

    def _encode(self, texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
        vectors = self._client.encode_documents(texts)
        return [vector.indices for vector in vectors], [vector.values for vector in vectors]

    def sparse_doc_fn(self, texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
        """Encode document texts; matches LlamaIndex's ``SparseEncoderCallable``."""
        return self._encode(texts)

    def sparse_query_fn(self, texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
        """Encode query texts; matches LlamaIndex's ``SparseEncoderCallable``."""
        return self._encode(texts)
