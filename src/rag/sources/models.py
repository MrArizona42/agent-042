"""Source manifest contracts for RAG data inputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag.contracts import SourceDocument

SourceType = str


class GenericSourceEntry(BaseModel):
    """Generic manifest entry for adapter-owned source families."""

    model_config = ConfigDict(extra="allow")

    id: str
    title: str
    uri: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "title")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()

    @field_validator("uri", "url")
    @classmethod
    def _blank_uri_means_missing(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    def to_source_document(self, source_type: str) -> SourceDocument:
        """Convert a generic manifest entry to a source document contract."""
        uri = self.uri or self.url or f"{source_type}:{self.id}"
        return SourceDocument(
            id=f"{source_type}:{self.id}",
            source_type=source_type,
            uri=uri,
            title=self.title,
            metadata=self.metadata,
        )


class SourceManifest(BaseModel):
    """One source-type manifest for one source instance."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    source_type: SourceType
    documents: list[GenericSourceEntry]

    @model_validator(mode="after")
    def _documents_must_have_unique_ids(self) -> "SourceManifest":
        seen_ids: set[str] = set()
        for document in self.documents:
            if document.id in seen_ids:
                raise ValueError(f"Duplicate source document id '{document.id}'")
            seen_ids.add(document.id)
        return self

    def to_source_documents(self) -> list[SourceDocument]:
        """Convert manifest entries to source document contracts."""
        return [document.to_source_document(self.source_type) for document in self.documents]


def source_manifest_from_raw(raw: dict[str, Any]) -> SourceManifest:
    """Validate a raw TOML payload as a typed source manifest."""
    documents = raw.get("documents", [])
    raw = {**raw, "documents": [GenericSourceEntry.model_validate(doc) for doc in documents]}
    return SourceManifest.model_validate(raw)
