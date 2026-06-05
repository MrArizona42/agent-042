"""Project-owned runtime RAG retrieval service."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from rag.domain import CollectionAttestation, RetrievalHit, attestation_from_payload
from rag.embeddings import EmbeddingService
from rag.reranker import Reranker, get_reranker
from rag.retriever import Retriever
from rag.runtime.models import RagRuntimeResult, RagRuntimeSource, RuntimeSkippedSource
from rag.sources.materialize import qdrant_alias_name, validate_strategy_supported
from rag.sparse_encoder import SparseEncoderService
from rag.vector_store import Document, QdrantVectorStore
from shared.catalog import KBConfig, get_catalog, get_kb_config
from shared.config import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _RuntimeAliasState:
    kb_id: str
    alias: str
    qdrant_alias: str
    collection_name: str
    attestation: CollectionAttestation


class RagRuntime:
    """Resolve catalog KB aliases into Qdrant retrieval operations."""

    def __init__(
        self,
        *,
        settings=None,
        embedding_service: EmbeddingService | None = None,
        vector_store_factory: Callable[[str], QdrantVectorStore] | None = None,
        reranker_factory: Callable[[str], Reranker] = get_reranker,
        sparse_encoder_factory: Callable[[], SparseEncoderService] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.platform_settings = self.settings.platform
        self.rag_settings = self.settings.rag
        self.embedding_service = embedding_service or EmbeddingService(
            model_name=self.rag_settings.embedding_model,
            device=self.rag_settings.embedding_device,
            batch_size=self.rag_settings.build.embedding_batch_size,
        )
        self._vector_store_factory = vector_store_factory or self._default_vector_store
        self._reranker_factory = reranker_factory
        self._sparse_encoder_factory = sparse_encoder_factory or self._default_sparse_encoder
        self._retrievers: dict[str, Retriever] = {}
        self._alias_states: dict[str, _RuntimeAliasState] = {}

    def invalidate_caches(self) -> None:
        """Clear alias and retriever caches."""
        self._retrievers.clear()
        self._alias_states.clear()

    def _default_vector_store(self, collection_name: str) -> QdrantVectorStore:
        return QdrantVectorStore(
            host=self.platform_settings.qdrant_host,
            port=self.platform_settings.qdrant_port,
            collection_name=collection_name,
        )

    def _default_sparse_encoder(self) -> SparseEncoderService:
        return SparseEncoderService(embeddings_url=self.platform_settings.embeddings_url)

    @staticmethod
    def _cache_key(kb_id: str, alias: str) -> str:
        return qdrant_alias_name(kb_id=kb_id, alias=alias)

    @staticmethod
    def _effective_alias(kb_cfg: KBConfig, alias: str | None) -> str:
        return alias or kb_cfg.default_alias

    def _resolve_alias_state(
        self,
        *,
        kb_id: str,
        alias: str,
        strict: bool,
    ) -> _RuntimeAliasState | None:
        cache_key = self._cache_key(kb_id, alias)
        cached = self._alias_states.get(cache_key)
        if cached is not None:
            alias_store = self._vector_store_factory(cache_key)
            current_collection = alias_store.resolve_alias(cache_key)
            if current_collection == cached.collection_name:
                return cached
            self._retrievers.pop(cache_key, None)
            self._alias_states.pop(cache_key, None)

        alias_store = self._vector_store_factory(cache_key)
        if not alias_store.collection_exists():
            message = f"Qdrant alias '{cache_key}' does not resolve"
            if strict:
                raise RuntimeError(message)
            logger.warning(message)
            return None

        collection_name = alias_store.resolve_alias(cache_key) or cache_key
        collection_store = (
            alias_store
            if collection_name == cache_key
            else self._vector_store_factory(collection_name)
        )
        payload = collection_store.read_meta()
        if payload is None:
            message = f"Collection '{collection_name}' has no Qdrant attestation"
            if strict:
                raise RuntimeError(message)
            logger.warning(message)
            return None

        attestation = attestation_from_payload(payload)
        if attestation.kb_id != kb_id:
            message = (
                f"Collection '{collection_name}' belongs to KB '{attestation.kb_id}', "
                f"not requested KB '{kb_id}'"
            )
            if strict:
                raise RuntimeError(message)
            logger.warning(message)
            return None
        if attestation.collection_name != collection_name:
            message = (
                f"Collection attestation names '{attestation.collection_name}', "
                f"not resolved collection '{collection_name}'"
            )
            if strict:
                raise RuntimeError(message)
            logger.warning(message)
            return None

        vector_size = collection_store.get_collection_info().get("vector_size")
        runtime_dimension = getattr(self.embedding_service, "dimension", None)
        if (
            isinstance(vector_size, int)
            and isinstance(runtime_dimension, int)
            and vector_size != runtime_dimension
        ):
            message = (
                f"Embedding dimension mismatch for '{collection_name}': "
                f"collection={vector_size}, runtime={runtime_dimension}"
            )
            if strict:
                raise RuntimeError(message)
            logger.warning(message)
            return None

        state = _RuntimeAliasState(
            kb_id=kb_id,
            alias=alias,
            qdrant_alias=cache_key,
            collection_name=collection_name,
            attestation=attestation,
        )
        self._alias_states[cache_key] = state
        return state

    def _get_retriever(
        self,
        *,
        kb_cfg: KBConfig,
        alias: str,
        state: _RuntimeAliasState,
    ) -> Retriever:
        cache_key = state.qdrant_alias
        cached = self._retrievers.get(cache_key)
        if cached is not None:
            return cached

        alias_cfg = kb_cfg.aliases[alias]
        vector_store = self._vector_store_factory(state.collection_name)
        reranker = self._reranker_factory(alias_cfg.reranker) if alias_cfg.reranker else None
        sparse_encoder_service = (
            self._sparse_encoder_factory()
            if alias_cfg.retrieval_strategy in {"hybrid", "sparse"}
            else None
        )
        retriever = Retriever(
            embedding_service=self.embedding_service,
            vector_store=vector_store,
            reranker=reranker,
            sparse_encoder_service=sparse_encoder_service,
            reranker_multiplier=alias_cfg.reranker_multiplier,
        )
        self._retrievers[cache_key] = retriever
        return retriever

    def _hit_from_document(
        self,
        document: Document,
        *,
        state: _RuntimeAliasState,
    ) -> RetrievalHit:
        metadata = dict(document.metadata)
        chunk_id = str(metadata.get("chunk_id") or metadata.get("id") or "")
        document_id = str(metadata.get("document_id") or metadata.get("source_document_id") or "")
        title = str(metadata.get("title") or document_id or "Untitled source")
        uri = str(metadata.get("source_uri") or metadata.get("uri") or "unknown")
        source_type = str(metadata.get("source_type") or "unknown")
        metadata.update(
            {
                "kb_id": state.kb_id,
                "alias": state.alias,
                "qdrant_alias": state.qdrant_alias,
                "collection_name": state.collection_name,
                "manifest_id": state.attestation.manifest_id,
                "retrieval_capability": state.attestation.retrieval_capability.value,
            }
        )
        return RetrievalHit(
            chunk_id=chunk_id or f"{state.collection_name}:unknown",
            document_id=document_id or chunk_id or f"{state.collection_name}:unknown",
            text=document.content,
            score=float(document.score if document.score is not None else 0.0),
            source_type=source_type,
            title=title,
            uri=uri,
            section_title=metadata.get("section_title"),
            metadata=metadata,
        )

    def validate_aliases(self, *, strict: bool = False) -> None:
        """Validate every declared catalog alias that is currently available."""
        for task_cfg in get_catalog().values():
            for kb_cfg in task_cfg.knowledge_bases:
                for alias, alias_cfg in kb_cfg.aliases.items():
                    state = self._resolve_alias_state(
                        kb_id=kb_cfg.name,
                        alias=alias,
                        strict=strict,
                    )
                    if state is None:
                        continue
                    try:
                        validate_strategy_supported(
                            retrieval_strategy=alias_cfg.retrieval_strategy,
                            retrieval_capability=state.attestation.retrieval_capability.value,
                        )
                    except ValueError:
                        if strict:
                            raise
                        logger.warning(
                            "RAG alias is not compatible with its collection: "
                            "kb=%s alias=%s collection=%s strategy=%s capability=%s",
                            kb_cfg.name,
                            alias,
                            state.collection_name,
                            alias_cfg.retrieval_strategy,
                            state.attestation.retrieval_capability.value,
                            exc_info=True,
                        )

    def retrieve(
        self,
        *,
        query: str,
        sources: list[RagRuntimeSource],
    ) -> RagRuntimeResult:
        """Retrieve citation-ready hits from requested KB aliases."""
        result = RagRuntimeResult()
        if not query.strip():
            return result

        for source in sources:
            kb_cfg = get_kb_config(source.knowledge_base)
            if kb_cfg is None:
                result.skipped_sources.append(
                    RuntimeSkippedSource(
                        knowledge_base=source.knowledge_base,
                        alias=source.alias,
                        reason="unknown_knowledge_base",
                    )
                )
                continue

            alias = self._effective_alias(kb_cfg, source.alias)
            alias_cfg = kb_cfg.aliases.get(alias)
            if alias_cfg is None:
                result.skipped_sources.append(
                    RuntimeSkippedSource(
                        knowledge_base=source.knowledge_base,
                        alias=alias,
                        reason="unknown_alias",
                    )
                )
                continue

            state = self._resolve_alias_state(kb_id=kb_cfg.name, alias=alias, strict=False)
            if state is None:
                result.skipped_sources.append(
                    RuntimeSkippedSource(
                        knowledge_base=source.knowledge_base,
                        alias=alias,
                        reason="alias_unavailable",
                    )
                )
                continue

            try:
                validate_strategy_supported(
                    retrieval_strategy=alias_cfg.retrieval_strategy,
                    retrieval_capability=state.attestation.retrieval_capability.value,
                )
            except ValueError as exc:
                result.skipped_sources.append(
                    RuntimeSkippedSource(
                        knowledge_base=source.knowledge_base,
                        alias=alias,
                        reason=str(exc),
                    )
                )
                continue

            retriever = self._get_retriever(kb_cfg=kb_cfg, alias=alias, state=state)
            documents = retriever.retrieve(
                query=query,
                top_k=alias_cfg.top_k,
                score_threshold=alias_cfg.score_threshold,
                strategy=alias_cfg.retrieval_strategy,
            )
            result.provenance.append(
                {
                    "knowledge_base": kb_cfg.name,
                    "alias": alias,
                    "qdrant_alias": state.qdrant_alias,
                    "collection_name": state.collection_name,
                    "manifest_id": state.attestation.manifest_id,
                    "retrieval_strategy": alias_cfg.retrieval_strategy,
                    "retrieval_capability": state.attestation.retrieval_capability.value,
                    "hit_count": len(documents),
                }
            )
            result.hits.extend(
                self._hit_from_document(document, state=state) for document in documents
            )

        return result
