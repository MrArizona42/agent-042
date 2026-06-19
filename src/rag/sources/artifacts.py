"""Persisted source extraction artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from llama_index.core import Document
from pydantic import BaseModel, ConfigDict, Field

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


class LlamaDocumentArtifact(BaseModel):
    """Stored extraction artifact with source and raw-cache provenance."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=2, ge=2)
    kb_id: str
    source_instance_id: str
    source_document: Document
    raw: RawArtifactRef
    extraction: ExtractionArtifactMeta
    document: Document


def extracted_artifact_path(
    *,
    rag_data_root: Path | str,
    kb_id: str,
    source_instance_id: str,
    source_document_id: str,
) -> Path:
    """Return the conventional extracted artifact path for one source document.

    Keyed by the globally unique source instance id, not by `kb_id`; `kb_id`
    is accepted for caller symmetry with sibling functions.
    """
    return (
        Path(rag_data_root)
        / "source_instances"
        / source_instance_id
        / "extracted"
        / f"{safe_document_id(source_document_id)}.json"
    )


def extracted_artifact_from_result(
    *,
    kb_id: str,
    source_instance_id: str,
    fetch_result: SourceFetchResult,
    extracted_document: Document,
) -> LlamaDocumentArtifact:
    """Build a persisted extraction artifact from fetch and extraction results."""
    source_document = fetch_result.source_document
    return LlamaDocumentArtifact(
        kb_id=kb_id,
        source_instance_id=source_instance_id,
        source_document=source_document,
        raw=RawArtifactRef(
            path=fetch_result.raw_path.as_posix(),
            checksum=fetch_result.checksum,
            content_type=fetch_result.content_type,
        ),
        extraction=ExtractionArtifactMeta(
            method=str(extracted_document.metadata["extraction_method"]),
            warnings=list(extracted_document.metadata.get("extraction_warnings", [])),
        ),
        document=extracted_document,
    )


def write_extracted_artifact(
    path: Path,
    artifact: LlamaDocumentArtifact,
    *,
    force: bool = False,
) -> None:
    """Write an extracted artifact once unless *force* is set."""
    write_json_immutable(path, artifact.model_dump(mode="json"), force=force)


def read_extracted_artifact(path: Path) -> LlamaDocumentArtifact:
    """Read a persisted extracted artifact."""
    return LlamaDocumentArtifact.model_validate(json.loads(path.read_text(encoding="utf-8")))
