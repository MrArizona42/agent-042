"""Persisted source extraction artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from rag.contracts import ExtractedDocument, SourceDocument
from rag.sources.cache import safe_document_id, write_json_immutable
from rag.sources.fetchers import SourceFetchResult


class RawArtifactRef(BaseModel):
    """Raw cache artifact used to produce an extracted document."""

    model_config = ConfigDict(extra="forbid")

    path: str
    checksum: str
    content_type: str | None = None


class ExtractionArtifactMeta(BaseModel):
    """Extraction method metadata."""

    model_config = ConfigDict(extra="forbid")

    method: str
    warnings: list[str] = Field(default_factory=list)


class ExtractedDocumentArtifact(BaseModel):
    """Stored extraction artifact with source and raw-cache provenance."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    kb_id: str
    source_instance_id: str
    source_type: str
    source_document: SourceDocument
    raw: RawArtifactRef
    extraction: ExtractionArtifactMeta
    document: ExtractedDocument


def extracted_artifact_path(
    *,
    rag_data_root: Path | str,
    kb_id: str,
    source_instance_id: str,
    source_document_id: str,
) -> Path:
    """Return the conventional extracted artifact path for one source document."""
    return (
        Path(rag_data_root)
        / kb_id
        / "extracted"
        / source_instance_id
        / f"{safe_document_id(source_document_id)}.json"
    )


def extracted_artifact_from_result(
    *,
    kb_id: str,
    source_instance_id: str,
    fetch_result: SourceFetchResult,
    extracted_document: ExtractedDocument,
) -> ExtractedDocumentArtifact:
    """Build a persisted extraction artifact from fetch and extraction results."""
    source_document = fetch_result.source_document
    return ExtractedDocumentArtifact(
        kb_id=kb_id,
        source_instance_id=source_instance_id,
        source_type=source_document.source_type,
        source_document=source_document,
        raw=RawArtifactRef(
            path=fetch_result.raw_path.as_posix(),
            checksum=fetch_result.checksum,
            content_type=fetch_result.content_type,
        ),
        extraction=ExtractionArtifactMeta(
            method=extracted_document.extraction_method,
            warnings=extracted_document.extraction_warnings,
        ),
        document=extracted_document,
    )


def write_extracted_artifact(
    path: Path,
    artifact: ExtractedDocumentArtifact,
    *,
    force: bool = False,
) -> None:
    """Write an extracted artifact once unless *force* is set."""
    write_json_immutable(path, artifact.model_dump(mode="json"), force=force)


def read_extracted_artifact(path: Path) -> ExtractedDocumentArtifact:
    """Read a persisted extracted artifact."""
    return ExtractedDocumentArtifact.model_validate(json.loads(path.read_text(encoding="utf-8")))
