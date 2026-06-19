"""Project-owned RAG contracts.

These models describe collection manifests and Qdrant metadata.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RetrievalCapability(StrEnum):
    """Vector legs available in a materialized index."""

    DENSE = "dense"
    HYBRID = "hybrid"
    SPARSE = "sparse"


class CollectionAttestation(BaseModel):
    """Small Qdrant-side runtime metadata for a live collection.

    The attestation is not the full source of truth for build provenance. It is
    a compact copy of the manifest fields needed to validate an alias target at
    runtime.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    manifest_id: str
    kb_id: str
    collection_name: str
    embedding_model: str
    sparse_encoder: str | None = None
    retrieval_capability: RetrievalCapability
    chunk_count: int = Field(ge=0)
    created_at: datetime

    @field_validator("manifest_id", "kb_id", "collection_name", "embedding_model")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()


class IndexManifest(BaseModel):
    """Full artifact manifest for a materialized RAG collection."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    manifest_id: str | None = None
    kb_id: str
    collection_name: str
    alias: str | None = None
    source_snapshot_id: str
    source_manifest_ref: str | None = None
    source_manifest_digests: dict[str, str] = Field(default_factory=dict)
    source_adapter_versions: dict[str, str] = Field(default_factory=dict)
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    embedding_model: str
    vector_dimension: int | None = Field(default=None, gt=0)
    sparse_encoder: str | None = None
    retrieval_capability: RetrievalCapability
    chunking_config: dict[str, Any] = Field(default_factory=dict)
    extraction_config: dict[str, Any] = Field(default_factory=dict)
    build_config_ref: str | None = None
    build_config_digest: str | None = None
    build_profile_digest: str | None = None
    benchmark_scope: str | None = None
    eval_summary: dict[str, Any] | None = None
    created_at: datetime

    @field_validator(
        "kb_id",
        "collection_name",
        "source_snapshot_id",
        "embedding_model",
    )
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()

    @model_validator(mode="after")
    def _manifest_id_must_match_canonical_payload(self) -> "IndexManifest":
        if self.manifest_id is None:
            return self

        from rag.contracts.manifests import compute_manifest_id

        expected_manifest_id = compute_manifest_id(self)
        if self.manifest_id != expected_manifest_id:
            raise ValueError(
                f"manifest_id does not match manifest payload (expected {expected_manifest_id})"
            )
        return self

    def to_attestation(self) -> CollectionAttestation:
        """Return the compact Qdrant-side metadata for this manifest."""
        if self.manifest_id is None:
            from rag.contracts.manifests import with_manifest_id

            manifest = with_manifest_id(self)
        else:
            manifest = self

        return CollectionAttestation(
            manifest_id=manifest.manifest_id or "",
            kb_id=manifest.kb_id,
            collection_name=manifest.collection_name,
            embedding_model=manifest.embedding_model,
            sparse_encoder=manifest.sparse_encoder,
            retrieval_capability=manifest.retrieval_capability,
            chunk_count=manifest.chunk_count,
            created_at=manifest.created_at,
        )


class ManifestComparison(BaseModel):
    """Result of comparing artifact manifest and Qdrant attestation."""

    model_config = ConfigDict(extra="forbid")

    matches: bool
    mismatches: dict[str, tuple[Any, Any]] = Field(default_factory=dict)


CollectionAlias = Literal["champion", "challenger", "shadow"]
