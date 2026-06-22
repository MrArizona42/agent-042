"""Resolve applied alias deployments into validated LlamaIndex Qdrant indexes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Literal
from uuid import UUID

from llama_index.core import VectorStoreIndex
from qdrant_client import AsyncQdrantClient, QdrantClient

from app_config.catalog.schema import AliasRetrievalConfig
from rag.contracts import compare_release_attestation
from rag.control_plane.models import AliasDeployment, RagRelease
from rag.control_plane.repositories import ReleaseRepository
from rag.indexing.llamaindex_embeddings import ProjectEmbedding, ProjectSparseEncoder
from rag.indexing.llamaindex_qdrant import QdrantCollectionManager
from rag.indexing.materialize import qdrant_alias_name

RetrievalStrategy = Literal["dense", "hybrid", "sparse"]


@dataclass(frozen=True, slots=True)
class RuntimeAliasState:
    """A validated, queryable release target and its applied retrieval configuration."""

    kb_id: str
    alias: str
    collection_name: str
    vector_size: int
    release: RagRelease
    retrieval_config: AliasRetrievalConfig
    qdrant_alias: str | None = None
    deployment_id: UUID | None = None

    @property
    def manifest_id(self) -> str:
        return self.release.manifest_id

    @property
    def retrieval_capability(self) -> str:
        return "hybrid" if self.release.build_config.sparse_encoder is not None else "dense"


class LlamaIndexRuntimeResolver:
    """Validate the applied alias deployment and reopen its LlamaIndex vector index."""

    def __init__(
        self,
        *,
        qdrant_client: QdrantClient,
        embedding_service,
        qdrant_batch_size: int,
        sparse_encoder_factory: Callable[[], object],
        release_repo: ReleaseRepository,
        qdrant_aclient: AsyncQdrantClient | None = None,
        collection_manager_factory: Callable[[str], QdrantCollectionManager] | None = None,
    ) -> None:
        self.qdrant_client = qdrant_client
        self.qdrant_aclient = qdrant_aclient
        self.embedding_service = embedding_service
        self.qdrant_batch_size = qdrant_batch_size
        self._sparse_encoder_factory = sparse_encoder_factory
        self._release_repo = release_repo
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
        """Resolve the Qdrant alias mirror's current target collection.

        Diagnostic/inspection only -- not used by `resolve()`. Postgres
        (the active `AliasDeployment`), not the Qdrant alias, is the
        runtime serving source of truth.
        """
        return self._manager_factory(qdrant_alias).resolve_alias(qdrant_alias)

    def resolve(
        self,
        *,
        kb_id: str,
        alias: str,
        deployment: AliasDeployment,
        strict: bool,
    ) -> RuntimeAliasState | None:
        """Validate *deployment*'s release and attestation, returning a queryable state."""
        release = self._release_repo.get(deployment.release_id)
        if release is None:
            self._fail(
                f"Active deployment for kb='{kb_id}' alias='{alias}' references unknown "
                f"release '{deployment.release_id}'",
                strict=strict,
            )
            return None

        manager = self._manager_factory(release.collection_name)
        if not manager.collection_exists():
            self._fail(f"Collection '{release.collection_name}' does not exist", strict=strict)
            return None

        attestation = manager.read_release_attestation()
        if attestation is None:
            self._fail(
                f"Collection '{release.collection_name}' has no release attestation",
                strict=strict,
            )
            return None

        comparison = compare_release_attestation(release, attestation)
        if not comparison.matches:
            self._fail(
                f"Release attestation mismatch for '{release.collection_name}': "
                f"{comparison.mismatches}",
                strict=strict,
            )
            return None

        provider_model = getattr(self.embedding_service, "model", None)
        if release.build_config.dense_encoder.model != provider_model:
            self._fail(
                f"Embedding model mismatch for '{release.collection_name}': "
                f"release={release.build_config.dense_encoder.model}, "
                f"provider={provider_model}",
                strict=strict,
            )
            return None

        vector_size = manager.vector_size()
        runtime_dimension = getattr(self.embedding_service, "dimension", None)
        if vector_size is None or vector_size != runtime_dimension:
            self._fail(
                f"Embedding dimension mismatch for '{release.collection_name}': "
                f"collection={vector_size}, runtime={runtime_dimension}",
                strict=strict,
            )
            return None

        return RuntimeAliasState(
            kb_id=kb_id,
            alias=alias,
            collection_name=release.collection_name,
            vector_size=vector_size,
            qdrant_alias=qdrant_alias_name(kb_id=kb_id, alias=alias),
            deployment_id=deployment.id,
            release=release,
            retrieval_config=deployment.retrieval_config,
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
                model_name=self.embedding_service.model,
            ),
        )
