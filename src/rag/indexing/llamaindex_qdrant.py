"""Project collection policy around LlamaIndex's Qdrant vector store."""

from __future__ import annotations

import asyncio
from typing import Any

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from rag.clients.qdrant import create_qdrant_clients
from rag.contracts import ReleaseAttestation
from rag.indexing.llamaindex_embeddings import ProjectSparseEncoder

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantCollectionManager:
    """Own collection lifecycle, attestation metadata, and alias operations."""

    def __init__(
        self,
        *,
        client: QdrantClient,
        collection_name: str,
        aclient: AsyncQdrantClient | None = None,
        owns_clients: bool = False,
    ) -> None:
        self.client = client
        self.aclient = aclient
        self.collection_name = collection_name
        self._owns_clients = owns_clients

    @classmethod
    def connect(
        cls,
        *,
        host: str,
        port: int,
        collection_name: str,
    ) -> "QdrantCollectionManager":
        client, aclient = create_qdrant_clients(host=host, port=port)
        return cls(
            client=client,
            aclient=aclient,
            collection_name=collection_name,
            owns_clients=True,
        )

    def close(self) -> None:
        """Close the sync client and, if present, the async client.

        Safe to call from a plain synchronous context (CLI commands) where no
        event loop is running. Use :meth:`aclose` instead from async contexts.
        """
        if not self._owns_clients:
            return
        self.client.close()
        if self.aclient is not None:
            asyncio.run(self.aclient.close())

    async def aclose(self) -> None:
        """Async counterpart to :meth:`close`, for use inside a running event loop."""
        if not self._owns_clients:
            return
        self.client.close()
        if self.aclient is not None:
            await self.aclient.close()

    def prepare_new_collection(self, *, force_recreate: bool) -> None:
        """Ensure materialization starts from a new physical collection."""
        if not self.collection_exists():
            return
        if not force_recreate:
            raise RuntimeError(
                f"Collection '{self.collection_name}' already exists; "
                "use force_recreate to replace it"
            )
        self.client.delete_collection(self.collection_name)

    def vector_store(
        self,
        *,
        vector_size: int,
        batch_size: int,
        enable_hybrid: bool,
        sparse_encoder: ProjectSparseEncoder | None,
    ) -> QdrantVectorStore:
        """Construct the LlamaIndex vector store for this physical collection."""
        if enable_hybrid and sparse_encoder is None:
            raise ValueError("hybrid vector store requires a sparse encoder")
        return QdrantVectorStore(
            collection_name=self.collection_name,
            client=self.client,
            aclient=self.aclient,
            batch_size=batch_size,
            dense_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            sparse_config=SparseVectorParams(index=SparseIndexParams()),
            enable_hybrid=enable_hybrid,
            sparse_doc_fn=(sparse_encoder.sparse_doc_fn if sparse_encoder else None),
            sparse_query_fn=(sparse_encoder.sparse_query_fn if sparse_encoder else None),
            dense_vector_name=DENSE_VECTOR_NAME,
            sparse_vector_name=SPARSE_VECTOR_NAME,
        )

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    def resolve_alias(self, alias_name: str) -> str | None:
        for alias in self.client.get_aliases().aliases:
            if alias.alias_name == alias_name:
                return alias.collection_name
        return None

    def vector_size(self) -> int | None:
        """Return the collection's dense vector dimension."""
        if not self.collection_exists():
            return None
        vectors = self.client.get_collection(self.collection_name).config.params.vectors
        if isinstance(vectors, dict):
            for params in vectors.values():
                if isinstance(params.size, int):
                    return params.size
            return None
        return vectors.size if isinstance(vectors.size, int) else None

    def write_release_attestation(self, attestation: ReleaseAttestation) -> None:
        """Persist schema-version-2 release attestation as Qdrant collection metadata."""
        if not self.collection_exists():
            raise RuntimeError(f"Collection '{self.collection_name}' does not exist")
        self.client.update_collection(
            collection_name=self.collection_name,
            metadata={
                "attestation": attestation.model_dump(mode="json", exclude_none=True),
            },
        )

    def read_release_attestation(self) -> ReleaseAttestation | None:
        """Read `.config.metadata.attestation` as a schema-version-2 release attestation."""
        if not self.collection_exists():
            return None
        metadata = self.client.get_collection(self.collection_name).config.metadata or {}
        payload: Any = metadata.get("attestation")
        if payload is None:
            return None
        return ReleaseAttestation.model_validate(payload)

    def update_alias(self, alias_name: str, collection_name: str) -> None:
        self.client.update_collection_aliases(
            [
                CreateAliasOperation(
                    create_alias=CreateAlias(
                        collection_name=collection_name,
                        alias_name=alias_name,
                    )
                )
            ]
        )
