"""Chunk extracted source artifacts into persisted project chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llama_index.core.node_parser import SentenceSplitter
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag.domain import Chunk, DocumentSection
from rag.sources.artifacts import ExtractedDocumentArtifact, read_extracted_artifact
from rag.sources.cache import safe_document_id, sha256_bytes, write_json_immutable

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
LLAMAINDEX_SENTENCE_SPLITTER = "llamaindex_sentence_splitter"


class ChunkingConfig(BaseModel):
    """Configuration for source artifact chunking."""

    model_config = ConfigDict(extra="forbid")

    method: str = LLAMAINDEX_SENTENCE_SPLITTER
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, gt=0)
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP, ge=0)

    @model_validator(mode="after")
    def _overlap_must_be_smaller_than_size(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class ChunkArtifact(BaseModel):
    """Stored chunks for one extracted source document."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    kb_id: str
    source_instance_id: str
    source_type: str
    source_document_id: str
    extracted_artifact_path: str
    extracted_checksum: str
    chunking: ChunkingConfig
    chunks: list[Chunk]


class SourceInstanceChunkingSummary(BaseModel):
    """Summary for chunking extracted artifacts from one source instance."""

    model_config = ConfigDict(extra="forbid")

    kb_id: str
    source_instance_id: str
    total_selected: int = Field(ge=0)
    chunked: int = Field(default=0, ge=0)
    from_cache: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)
    failed: list[dict[str, str]] = Field(default_factory=list)


def chunk_artifact_path(
    *,
    rag_data_root: Path | str,
    kb_id: str,
    source_instance_id: str,
    source_document_id: str,
) -> Path:
    """Return the conventional chunk artifact path for one source document."""
    return (
        Path(rag_data_root)
        / kb_id
        / "chunks"
        / source_instance_id
        / f"{safe_document_id(source_document_id)}.json"
    )


def write_chunk_artifact(path: Path, artifact: ChunkArtifact, *, force: bool = False) -> None:
    """Write a chunk artifact once unless *force* is set."""
    write_json_immutable(path, artifact.model_dump(mode="json"), force=force)


def read_chunk_artifact(path: Path) -> ChunkArtifact:
    """Read a persisted chunk artifact."""
    return ChunkArtifact.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _extracted_checksum(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _split_text(text: str, config: ChunkingConfig) -> list[str]:
    splitter = SentenceSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def _section_inputs(artifact: ExtractedDocumentArtifact) -> list[tuple[str, str, dict[str, Any]]]:
    sections = artifact.document.sections
    if not sections:
        return [
            (
                artifact.source_document.title,
                artifact.document.text,
                {"section_ordinal": None, "section_level": None},
            )
        ]

    result: list[tuple[str, str, dict[str, Any]]] = []
    for section in sections:
        result.append((_section_title(section), section.text, _section_metadata(section)))
    return result


def _section_title(section: DocumentSection) -> str:
    return section.title or f"Section {section.ordinal + 1}"


def _section_metadata(section: DocumentSection) -> dict[str, Any]:
    return {
        "section_ordinal": section.ordinal,
        "section_level": section.level,
        **section.metadata,
    }


def _chunk_id(source_document_id: str, ordinal: int) -> str:
    return f"{source_document_id}:chunk:{ordinal:04d}"


def _build_chunks(
    artifact: ExtractedDocumentArtifact,
    *,
    artifact_path: Path,
    config: ChunkingConfig,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section_title, text, section_metadata in _section_inputs(artifact):
        for chunk_text in _split_text(text, config):
            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    id=_chunk_id(artifact.document.id, ordinal),
                    document_id=artifact.document.id,
                    source_document_id=artifact.document.source_document_id,
                    text=chunk_text,
                    section_title=section_title,
                    ordinal=ordinal,
                    token_count=len(chunk_text.split()),
                    metadata={
                        "kb_id": artifact.kb_id,
                        "source_instance_id": artifact.source_instance_id,
                        "source_type": artifact.source_type,
                        "source_uri": artifact.source_document.uri,
                        "title": artifact.source_document.title,
                        "extracted_artifact_path": artifact_path.as_posix(),
                        **section_metadata,
                    },
                )
            )
    return chunks


def chunk_extracted_artifact(
    extracted_path: Path | str,
    *,
    rag_data_root: Path | str,
    config: ChunkingConfig | None = None,
    force: bool = False,
) -> ChunkArtifact:
    """Chunk one extracted artifact and persist the chunk artifact."""
    extracted_path = Path(extracted_path)
    artifact = read_extracted_artifact(extracted_path)
    output_path = chunk_artifact_path(
        rag_data_root=rag_data_root,
        kb_id=artifact.kb_id,
        source_instance_id=artifact.source_instance_id,
        source_document_id=artifact.document.source_document_id,
    )
    if output_path.exists() and not force:
        return read_chunk_artifact(output_path)

    chunking_config = config or ChunkingConfig()
    chunk_artifact = ChunkArtifact(
        kb_id=artifact.kb_id,
        source_instance_id=artifact.source_instance_id,
        source_type=artifact.source_type,
        source_document_id=artifact.document.source_document_id,
        extracted_artifact_path=extracted_path.as_posix(),
        extracted_checksum=_extracted_checksum(extracted_path),
        chunking=chunking_config,
        chunks=_build_chunks(artifact, artifact_path=extracted_path, config=chunking_config),
    )
    write_chunk_artifact(output_path, chunk_artifact, force=force)
    return chunk_artifact


def chunk_source_instance(
    *,
    rag_data_root: Path | str,
    kb_id: str,
    source_instance_id: str,
    document_ids: list[str] | None = None,
    limit: int | None = None,
    config: ChunkingConfig | None = None,
    force: bool = False,
) -> SourceInstanceChunkingSummary:
    """Chunk extracted artifacts for one source instance."""
    root = Path(rag_data_root)
    extracted_dir = root / kb_id / "extracted" / source_instance_id
    extracted_paths = sorted(extracted_dir.glob("*.json"))
    if document_ids is not None:
        selected_ids = {safe_document_id(document_id) for document_id in document_ids}
        extracted_paths = [path for path in extracted_paths if path.stem in selected_ids]
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        extracted_paths = extracted_paths[:limit]

    summary = SourceInstanceChunkingSummary(
        kb_id=kb_id,
        source_instance_id=source_instance_id,
        total_selected=len(extracted_paths),
    )
    for extracted_path in extracted_paths:
        try:
            artifact = read_extracted_artifact(extracted_path)
            output_path = chunk_artifact_path(
                rag_data_root=root,
                kb_id=artifact.kb_id,
                source_instance_id=artifact.source_instance_id,
                source_document_id=artifact.document.source_document_id,
            )
            from_cache = output_path.exists() and not force
            chunk_artifact = chunk_extracted_artifact(
                extracted_path,
                rag_data_root=root,
                config=config,
                force=force,
            )
            summary.chunk_count += len(chunk_artifact.chunks)
            if from_cache:
                summary.from_cache += 1
            else:
                summary.chunked += 1
        except Exception as exc:  # noqa: BLE001 - lifecycle summary owns per-doc failures.
            summary.failed.append(
                {
                    "artifact_path": extracted_path.as_posix(),
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                }
            )
    return summary
