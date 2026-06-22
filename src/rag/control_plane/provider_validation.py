"""Validate catalog-declared provider identity against live providers.

Orchestrates the per-client `validate_*_identity` checks (phase 1) for a
complete alias build/retrieve declaration, returning human-readable
mismatch descriptions rather than raising -- callers (alias diff, alias
apply) decide what to do with mismatches instead of having an exception
choose for them.
"""

from __future__ import annotations

from app_config.catalog.schema import AliasBuildConfig, AliasRetrievalConfig
from rag.embeddings import (
    EmbeddingIdentityMismatch,
    EmbeddingService,
    validate_dense_encoder_identity,
)
from rag.reranker import CrossEncoderReranker, RerankerIdentityMismatch, validate_reranker_identity
from rag.sparse_encoder import (
    SparseEncoderIdentityMismatch,
    SparseEncoderService,
    validate_sparse_encoder_identity,
)


def validate_build_provider_identity(
    build_config: AliasBuildConfig,
    *,
    embedding_client: EmbeddingService,
    sparse_encoder_client: SparseEncoderService | None,
) -> list[str]:
    """Return mismatch descriptions between *build_config* and live providers."""
    mismatches: list[str] = []
    try:
        validate_dense_encoder_identity(
            embedding_client,
            expected_model=build_config.dense_encoder.model,
            expected_dimension=build_config.dense_encoder.dimension,
        )
    except EmbeddingIdentityMismatch as exc:
        mismatches.append(str(exc))

    if build_config.sparse_encoder is not None:
        if sparse_encoder_client is None:
            mismatches.append(
                "build declares sparse_encoder but no sparse encoder provider is configured"
            )
        else:
            try:
                validate_sparse_encoder_identity(
                    sparse_encoder_client, expected_model=build_config.sparse_encoder.model
                )
            except SparseEncoderIdentityMismatch as exc:
                mismatches.append(str(exc))
    return mismatches


def validate_retrieval_provider_identity(
    retrieve_config: AliasRetrievalConfig,
    *,
    reranker_client: CrossEncoderReranker | None,
) -> list[str]:
    """Return mismatch descriptions between *retrieve_config* and the live reranker."""
    if retrieve_config.reranker is None:
        return []
    if reranker_client is None:
        return ["retrieve declares reranker but no reranker provider is configured"]
    try:
        validate_reranker_identity(reranker_client, expected_model=retrieve_config.reranker)
    except RerankerIdentityMismatch as exc:
        return [str(exc)]
    return []
