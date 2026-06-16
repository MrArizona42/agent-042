"""Backward-compatibility re-exports from rag.indexing.materialize."""

from rag.indexing.materialize import (
    AliasPromotionResult,
    EmbeddingClient,
    MaterializationResult,
    MaterializationSummary,
    RetrievalStrategy,
    SourceRetrievalCapability,
    SourceVectorStore,
    SparseEmbeddingClient,
    collection_name_for_build,
    materialize_kb_collection,
    promote_materialized_alias,
    qdrant_alias_name,
    retrieval_capability_for_strategy,
    source_snapshot_id,
    validate_strategy_supported,
)

__all__ = [
    "AliasPromotionResult",
    "EmbeddingClient",
    "MaterializationResult",
    "MaterializationSummary",
    "RetrievalStrategy",
    "SourceRetrievalCapability",
    "SourceVectorStore",
    "SparseEmbeddingClient",
    "collection_name_for_build",
    "materialize_kb_collection",
    "promote_materialized_alias",
    "qdrant_alias_name",
    "retrieval_capability_for_strategy",
    "source_snapshot_id",
    "validate_strategy_supported",
]
