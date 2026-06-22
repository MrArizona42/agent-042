"""Project-owned RAG contracts.

These models describe collection manifests and Qdrant metadata.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetrievalCapability(StrEnum):
    """Vector legs available in a materialized index."""

    DENSE = "dense"
    HYBRID = "hybrid"
    SPARSE = "sparse"


class ManifestComparison(BaseModel):
    """Result of comparing artifact manifest and Qdrant attestation."""

    model_config = ConfigDict(extra="forbid")

    matches: bool
    mismatches: dict[str, tuple[Any, Any]] = Field(default_factory=dict)


class ReleaseAttestation(BaseModel):
    """Small Qdrant-side runtime metadata for a collection backing an immutable release.

    This is the schema-version-2 attestation written by release builds for
    both durable KB releases and disposable benchmark releases.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    release_id: str
    manifest_id: str
    kb_id: str
    collection_name: str
    release_fingerprint: str
    build_config_digest: str
    source_snapshot_id: str
    dense_encoder_model: str
    dense_vector_dimension: int = Field(gt=0)
    sparse_encoder_model: str | None = None
    retrieval_capability: RetrievalCapability
    chunk_count: int = Field(ge=0)
    created_at: datetime

    @field_validator(
        "release_id",
        "manifest_id",
        "kb_id",
        "collection_name",
        "release_fingerprint",
        "build_config_digest",
        "source_snapshot_id",
        "dense_encoder_model",
    )
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()
