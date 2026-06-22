"""Strict native-node artifact collection for materialization."""

from __future__ import annotations

from pathlib import Path

from llama_index.core.schema import TextNode
from pydantic import BaseModel, ConfigDict, Field

from rag.sources.cache import safe_document_id, sha256_bytes
from rag.sources.chunks import NodeArtifact, read_chunk_artifact


class SourceNodeBundle(BaseModel):
    """Materialization-ready LlamaIndex nodes from one source instance."""

    model_config = ConfigDict(extra="forbid")

    kb_id: str
    source_instance_id: str
    node_artifact_paths: list[str]
    node_artifact_checksums: dict[str, str]
    nodes: list[TextNode]
    document_count: int = Field(ge=0)
    node_count: int = Field(ge=0)


def _node_artifact_paths(
    *,
    rag_data_root: Path | str,
    source_instance_id: str,
    document_ids: list[str] | None,
    limit: int | None,
    transformation_digest: str | None = None,
) -> list[Path]:
    chunks_dir = Path(rag_data_root) / "source_instances" / source_instance_id / "chunks"
    if transformation_digest is not None:
        from rag.control_plane.fingerprints import digest_directory_name

        chunks_dir = chunks_dir / digest_directory_name(transformation_digest)
    paths = sorted(chunks_dir.glob("*.json"))
    if document_ids is not None:
        selected_ids = {
            safe_document_id(candidate)
            for document_id in document_ids
            for candidate in (document_id, f"{source_instance_id}:{document_id}")
        }
        paths = [path for path in paths if path.stem in selected_ids]
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        paths = paths[:limit]
    return paths


def collect_source_nodes(
    *,
    rag_data_root: Path | str,
    kb_id: str,
    source_instance_id: str,
    document_ids: list[str] | None = None,
    limit: int | None = None,
    transformation_digest: str | None = None,
) -> SourceNodeBundle:
    """Collect valid native node artifacts for one source instance."""
    paths = _node_artifact_paths(
        rag_data_root=rag_data_root,
        source_instance_id=source_instance_id,
        document_ids=document_ids,
        limit=limit,
        transformation_digest=transformation_digest,
    )
    artifacts: list[NodeArtifact] = []
    checksums: dict[str, str] = {}
    for path in paths:
        artifacts.append(read_chunk_artifact(path))
        checksums[path.as_posix()] = sha256_bytes(path.read_bytes())

    nodes = [node for artifact in artifacts for node in artifact.nodes]
    document_ids_seen = {artifact.source_document_id for artifact in artifacts}
    return SourceNodeBundle(
        kb_id=kb_id,
        source_instance_id=source_instance_id,
        node_artifact_paths=[path.as_posix() for path in paths],
        node_artifact_checksums=checksums,
        nodes=nodes,
        document_count=len(document_ids_seen),
        node_count=len(nodes),
    )


def collect_source_bundles(
    *,
    rag_data_root: Path | str,
    kb_id: str,
    source_instance_ids: list[str],
    document_ids: list[str] | None = None,
    limit: int | None = None,
    transformation_digest: str | None = None,
) -> list[SourceNodeBundle]:
    return [
        collect_source_nodes(
            rag_data_root=rag_data_root,
            kb_id=kb_id,
            source_instance_id=source_instance_id,
            document_ids=document_ids,
            limit=limit,
            transformation_digest=transformation_digest,
        )
        for source_instance_id in source_instance_ids
    ]
