"""Source manifest contracts for RAG data inputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag.contracts import SourceDocument

SourceType = Literal["arxiv_paper", "html_docs"]


class ArxivPaperEntry(BaseModel):
    """Curated ArXiv paper entry."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    url: str | None = None

    @field_validator("id", "title")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()

    @field_validator("url")
    @classmethod
    def _blank_url_means_missing(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    def to_source_document(self) -> SourceDocument:
        """Convert the manifest entry to a source document contract."""
        return SourceDocument(
            id=f"arxiv:{self.id}",
            source_type="arxiv_paper",
            uri=self.url or f"arxiv:{self.id}",
            title=self.title,
            metadata={"arxiv_id": self.id},
        )


class HtmlDocsEntry(BaseModel):
    """Curated HTML documentation page entry."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    url: str | None = None

    @field_validator("id", "title")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()

    @field_validator("url")
    @classmethod
    def _blank_url_means_missing(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    def to_source_document(self) -> SourceDocument:
        """Convert the manifest entry to a source document contract."""
        if self.url is None:
            raise ValueError(f"HTML docs entry '{self.id}' requires url before fetch")
        return SourceDocument(
            id=f"html:{self.id}",
            source_type="html_docs",
            uri=self.url,
            title=self.title,
            metadata={"page_id": self.id},
        )


class SourceManifest(BaseModel):
    """One source-type manifest for one source instance."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    source_type: SourceType
    documents: list[ArxivPaperEntry | HtmlDocsEntry]

    @model_validator(mode="after")
    def _documents_must_match_source_type(self) -> "SourceManifest":
        expected_type = ArxivPaperEntry if self.source_type == "arxiv_paper" else HtmlDocsEntry
        for document in self.documents:
            if not isinstance(document, expected_type):
                raise ValueError(
                    f"source_type '{self.source_type}' manifest contains incompatible document "
                    f"entry {document.__class__.__name__}"
                )
        seen_ids: set[str] = set()
        for document in self.documents:
            if document.id in seen_ids:
                raise ValueError(f"Duplicate source document id '{document.id}'")
            seen_ids.add(document.id)
        return self

    def to_source_documents(self) -> list[SourceDocument]:
        """Convert manifest entries to source document contracts."""
        return [document.to_source_document() for document in self.documents]


def source_manifest_from_raw(raw: dict[str, Any]) -> SourceManifest:
    """Validate a raw TOML payload as a typed source manifest."""
    source_type = raw.get("source_type")
    documents = raw.get("documents", [])
    if source_type == "arxiv_paper":
        raw = {**raw, "documents": [ArxivPaperEntry.model_validate(doc) for doc in documents]}
    elif source_type == "html_docs":
        raw = {**raw, "documents": [HtmlDocsEntry.model_validate(doc) for doc in documents]}
    return SourceManifest.model_validate(raw)
