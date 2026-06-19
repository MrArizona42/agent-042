"""Resolve catalog aliases into validated LlamaIndex Qdrant indexes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Literal

from llama_index.core import VectorStoreIndex
from qdrant_client import AsyncQdrantClient, QdrantClient

from rag.contracts import CollectionAttestation
from rag.indexing.llamaindex_embeddings import ProjectEmbedding, ProjectSparseEncoder
from rag.indexing.llamaindex_qdrant import QdrantCollectionManager

RetrievalStrategy = Literal["dense", "hybrid", "sparse"]


@dataclass(frozen=True, slots=True)
class RuntimeAliasState:
    kb_id: str
    alias: str
    qdrant_alias: str
    collection_name: str
    attestation: CollectionAttestation
    vector_size: int


class LlamaIndexRuntimeResolver:
    """Validate alias targets and reopen their LlamaIndex vector indexes."""

    def __init__(
        self,
        *,
        qdrant_client: QdrantClient,
        embedding_service,
        embedding_model: str,
        qdrant_batch_size: int,
        sparse_encoder_factory: Callable[[], object],
        qdrant_aclient: AsyncQdrantClient | None = None,
        collection_manager_factory: Callable[[str], QdrantCollectionManager] | None = None,
    ) -> None:
        self.qdrant_client = qdrant_client
        self.qdrant_aclient = qdrant_aclient
        self.embedding_service = embedding_service
        self.embedding_model = embedding_model
        self.qdrant_batch_size = qdrant_batch_size
        self._sparse_encoder_factory = sparse_encoder_factory
        self._manager_factory = collection_manager_factory or self._default_manager

    def _default_manager(self, collection_name: str) -> QdrantCollectionManager:
        return QdrantCollectionManager(
            client=self.qdrant_client,
            aclient=self.qdrant_aclient,
            collection_name=collection_name,
        )

    def close(self) -> None:
        """Close the resolver's shared sync/async Qdrant clients (sync context)."""
        self.qdrant_client.close()
        if self.qdrant_aclient is not None:
            asyncio.run(self.qdrant_aclient.close())

    async def aclose(self) -> None:
        """Close the resolver's shared sync/async Qdrant clients (async context)."""
        self.qdrant_client.close()
        if self.qdrant_aclient is not None:
            await self.qdrant_aclient.close()

    @staticmethod
    def _fail(message: str, *, strict: bool) -> None:
        if strict:
            raise RuntimeError(message)

    def alias_target(self, qdrant_alias: str) -> str | None:
        return self._manager_factory(qdrant_alias).resolve_alias(qdrant_alias)

    def resolve(
        self,
        *,
        kb_id: str,
        alias: str,
        qdrant_alias: str,
        strict: bool,
    ) -> RuntimeAliasState | None:
        collection_name = self.alias_target(qdrant_alias)
        if collection_name is None:
            self._fail(f"Qdrant alias '{qdrant_alias}' does not resolve", strict=strict)
            return None

        manager = self._manager_factory(collection_name)
        attestation = manager.read_attestation()
        if attestation is None:
            self._fail(
                f"Collection '{collection_name}' has no collection metadata attestation",
                strict=strict,
            )
            return None
        if attestation.kb_id != kb_id:
            self._fail(
                f"Collection '{collection_name}' belongs to KB '{attestation.kb_id}', "
                f"not requested KB '{kb_id}'",
                strict=strict,
            )
            return None
        if attestation.collection_name != collection_name:
            self._fail(
                f"Collection attestation names '{attestation.collection_name}', "
                f"not resolved collection '{collection_name}'",
                strict=strict,
            )
            return None
        if attestation.embedding_model != self.embedding_model:
            self._fail(
                f"Embedding model mismatch for '{collection_name}': "
                f"collection={attestation.embedding_model}, runtime={self.embedding_model}",
                strict=strict,
            )
            return None

        vector_size = manager.vector_size()
        runtime_dimension = getattr(self.embedding_service, "dimension", None)
        if vector_size is None or vector_size != runtime_dimension:
            self._fail(
                f"Embedding dimension mismatch for '{collection_name}': "
                f"collection={vector_size}, runtime={runtime_dimension}",
                strict=strict,
            )
            return None
        return RuntimeAliasState(
            kb_id=kb_id,
            alias=alias,
            qdrant_alias=qdrant_alias,
            collection_name=collection_name,
            attestation=attestation,
            vector_size=vector_size,
        )

    def open_index(
        self,
        state: RuntimeAliasState,
        *,
        strategy: RetrievalStrategy,
    ) -> VectorStoreIndex:
        sparse_encoder = (
            ProjectSparseEncoder(self._sparse_encoder_factory())
            if strategy in {"hybrid", "sparse"}
            else None
        )
        vector_store = self._manager_factory(state.collection_name).vector_store(
            vector_size=state.vector_size,
            batch_size=self.qdrant_batch_size,
            enable_hybrid=strategy in {"hybrid", "sparse"},
            sparse_encoder=sparse_encoder,
        )
        return VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=ProjectEmbedding(
                embedding_client=self.embedding_service,
                model_name=self.embedding_model,
            ),
        )
