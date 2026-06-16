"""Strict chunk artifact collection for materialization."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from rag.contracts import Chunk
from rag.sources.cache import safe_document_id, sha256_bytes
from rag.sources.chunks import ChunkArtifact, read_chunk_artifact


class SourceChunkBundle(BaseModel):
    """Materialization-ready chunks from one source instance."""

    model_config = ConfigDict(extra="forbid")

    kb_id: str
    source_instance_id: str
    source_types: list[str]
    chunk_artifact_paths: list[str]
    chunk_artifact_checksums: dict[str, str]
    chunks: list[Chunk]
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)


def _chunk_artifact_paths(
    *,
    rag_data_root: Path | str,
    kb_id: str,
    source_instance_id: str,
    document_ids: list[str] | None,
    limit: int | None,
) -> list[Path]:
    chunk_dir = Path(rag_data_root) / kb_id / "chunks" / source_instance_id
    paths = sorted(chunk_dir.glob("*.json"))
    if document_ids is not None:
        selected_ids = {safe_document_id(document_id) for document_id in document_ids}
        paths = [path for path in paths if path.stem in selected_ids]
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        paths = paths[:limit]
    return paths


def collect_source_chunks(
    *,
    rag_data_root: Path | str,
    kb_id: str,
    source_instance_id: str,
    document_ids: list[str] | None = None,
    limit: int | None = None,
) -> SourceChunkBundle:
    """Collect valid chunk artifacts for one source instance.

    This intentionally fails on invalid/corrupt artifacts. Processing and
    chunking stages collect per-document failures; materialization input should
    be strict.
    """
    paths = _chunk_artifact_paths(
        rag_data_root=rag_data_root,
        kb_id=kb_id,
        source_instance_id=source_instance_id,
        document_ids=document_ids,
        limit=limit,
    )
    artifacts: list[ChunkArtifact] = []
    checksums: dict[str, str] = {}
    for path in paths:
        artifacts.append(read_chunk_artifact(path))
        checksums[path.as_posix()] = sha256_bytes(path.read_bytes())

    source_types = sorted({artifact.source_type for artifact in artifacts})
    chunks = [chunk for artifact in artifacts for chunk in artifact.chunks]
    document_ids_seen = {artifact.source_document_id for artifact in artifacts}
    return SourceChunkBundle(
        kb_id=kb_id,
        source_instance_id=source_instance_id,
        source_types=source_types,
        chunk_artifact_paths=[path.as_posix() for path in paths],
        chunk_artifact_checksums=checksums,
        chunks=chunks,
        document_count=len(document_ids_seen),
        chunk_count=len(chunks),
    )
