"""Project-owned RAG domain contracts.

These models describe the lifecycle data that crosses source connectors,
extractors, build pipelines, runtime retrieval, manifests, and Qdrant
collection metadata.
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


class DocumentSection(BaseModel):
    """A structured section extracted from a source document."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    text: str
    level: int | None = Field(default=None, ge=1)
    ordinal: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def _text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("section text must be non-empty")
        return value


class SourceDocument(BaseModel):
    """A selected source before extraction."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source_type: str
    uri: str
    title: str
    authors: list[str] = Field(default_factory=list)
    published_at: datetime | None = None
    raw_path: str | None = None
    checksum: str | None = None
    license: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "source_type", "uri", "title")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()

    @field_validator("authors")
    @classmethod
    def _authors_must_not_be_blank(cls, value: list[str]) -> list[str]:
        return [author.strip() for author in value if author.strip()]


class ExtractedDocument(BaseModel):
    """Text and structure extracted from a source document."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source_document_id: str
    text: str
    sections: list[DocumentSection] = Field(default_factory=list)
    extraction_method: str
    extraction_warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "source_document_id", "text", "extraction_method")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value


class Chunk(BaseModel):
    """A retrievable text unit derived from an extracted document."""

    model_config = ConfigDict(extra="forbid")

    id: str
    document_id: str
    source_document_id: str
    text: str
    section_title: str | None = None
    ordinal: int = Field(ge=0)
    token_count: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "document_id", "source_document_id", "text")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value


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
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    embedding_model: str
    sparse_encoder: str | None = None
    retrieval_capability: RetrievalCapability
    chunking_config: dict[str, Any] = Field(default_factory=dict)
    extraction_config: dict[str, Any] = Field(default_factory=dict)
    build_config_ref: str | None = None
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

        from rag.domain.manifests import compute_manifest_id

        expected_manifest_id = compute_manifest_id(self)
        if self.manifest_id != expected_manifest_id:
            raise ValueError(
                f"manifest_id does not match manifest payload "
                f"(expected {expected_manifest_id})"
            )
        return self

    def to_attestation(self) -> CollectionAttestation:
        """Return the compact Qdrant-side metadata for this manifest."""
        if self.manifest_id is None:
            from rag.domain.manifests import with_manifest_id

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


class RetrievalHit(BaseModel):
    """Citation-ready runtime retrieval hit."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    text: str
    score: float
    source_type: str
    title: str
    uri: str
    section_title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunk_id", "document_id", "text", "source_type", "title", "uri")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value


class ManifestComparison(BaseModel):
    """Result of comparing artifact manifest and Qdrant attestation."""

    model_config = ConfigDict(extra="forbid")

    matches: bool
    mismatches: dict[str, tuple[Any, Any]] = Field(default_factory=dict)


CollectionAlias = Literal["champion", "challenger", "shadow"]
