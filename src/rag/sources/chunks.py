"""Parse extracted LlamaIndex documents into persisted native text nodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode, TextNode
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag.contracts.metadata import (
    node_id_for_chunk,
    require_document_metadata,
    require_node_metadata,
)
from rag.sources.artifacts import ExtractedDocumentArtifact, read_extracted_artifact
from rag.sources.cache import safe_document_id, sha256_bytes, write_json_immutable

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
LLAMAINDEX_SENTENCE_SPLITTER = "llamaindex_sentence_splitter"


class ChunkingConfig(BaseModel):
    """Configuration for LlamaIndex node parsing."""

    model_config = ConfigDict(extra="forbid")

    method: str = LLAMAINDEX_SENTENCE_SPLITTER
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE, gt=0)
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP, ge=0)

    @model_validator(mode="after")
    def _overlap_must_be_smaller_than_size(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class NodeArtifact(BaseModel):
    """Native LlamaIndex text nodes derived from one extracted document."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=2, ge=2)
    kb_id: str
    source_instance_id: str
    source_document_id: str
    extracted_artifact_path: str
    extracted_checksum: str
    chunking: ChunkingConfig
    nodes: list[TextNode]


class SourceInstanceChunkingSummary(BaseModel):
    """Summary for parsing one source instance into text nodes."""

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
    """Return the transitional native-node artifact path."""
    del kb_id
    return (
        Path(rag_data_root)
        / "source_instances"
        / source_instance_id
        / "chunks"
        / f"{safe_document_id(source_document_id)}.json"
    )


def write_chunk_artifact(path: Path, artifact: NodeArtifact, *, force: bool = False) -> None:
    write_json_immutable(path, artifact.model_dump(mode="json"), force=force)


def read_chunk_artifact(path: Path) -> NodeArtifact:
    return NodeArtifact.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _section_documents(document: Document) -> list[Document]:
    require_document_metadata(document)
    raw_sections = document.metadata.get("sections") or []
    base_metadata = {key: value for key, value in document.metadata.items() if key != "sections"}
    if not raw_sections:
        metadata = {
            **base_metadata,
            "section_title": None,
            "section_ordinal": None,
            "section_level": None,
        }
        return [
            Document(
                text=document.text,
                id_=document.id_,
                metadata=metadata,
                excluded_embed_metadata_keys=list(metadata),
                excluded_llm_metadata_keys=list(metadata),
            )
        ]

    sections: list[Document] = []
    for raw in raw_sections:
        section = dict(raw)
        section_metadata = dict(section.get("metadata") or {})
        metadata = {
            **base_metadata,
            **section_metadata,
            "section_title": section.get("title"),
            "section_ordinal": section.get("ordinal"),
            "section_level": section.get("level"),
        }
        sections.append(
            Document(
                text=str(section["text"]),
                id_=document.id_,
                metadata=metadata,
                excluded_embed_metadata_keys=list(metadata),
                excluded_llm_metadata_keys=list(metadata),
            )
        )
    return sections


def _build_nodes(
    artifact: ExtractedDocumentArtifact,
    *,
    artifact_path: Path,
    config: ChunkingConfig,
) -> list[TextNode]:
    parser = SentenceSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        include_metadata=False,
    )
    parsed_with_metadata: list[tuple[TextNode, dict[str, Any]]] = []
    for section_document in _section_documents(artifact.document):
        for parsed_node in parser.get_nodes_from_documents([section_document]):
            parsed_with_metadata.append((parsed_node, section_document.metadata))
    nodes: list[TextNode] = []
    for ordinal, (parsed_node, section_metadata) in enumerate(parsed_with_metadata):
        metadata: dict[str, Any] = {
            **section_metadata,
            "chunk_id": f"{artifact.document.id_}:chunk:{ordinal:04d}",
            "document_id": artifact.document.id_,
            "source_document_id": artifact.document.metadata["source_document_id"],
            "ordinal": ordinal,
            "token_count": len(parsed_node.get_content(metadata_mode=MetadataMode.NONE).split()),
            "extracted_artifact_path": artifact_path.as_posix(),
        }
        node = TextNode(
            id_=node_id_for_chunk(str(metadata["chunk_id"])),
            text=parsed_node.get_content(metadata_mode=MetadataMode.NONE),
            metadata=metadata,
            relationships=parsed_node.relationships,
        )
        require_node_metadata(node)
        nodes.append(node)
    return nodes


def chunk_extracted_artifact(
    extracted_path: Path | str,
    *,
    rag_data_root: Path | str,
    config: ChunkingConfig | None = None,
    force: bool = False,
) -> NodeArtifact:
    """Parse one extracted document and persist native LlamaIndex nodes."""
    extracted_path = Path(extracted_path)
    artifact = read_extracted_artifact(extracted_path)
    source_document_id = str(artifact.document.metadata["source_document_id"])
    output_path = chunk_artifact_path(
        rag_data_root=rag_data_root,
        kb_id=artifact.kb_id,
        source_instance_id=artifact.source_instance_id,
        source_document_id=source_document_id,
    )
    if output_path.exists() and not force:
        return read_chunk_artifact(output_path)

    chunking_config = config or ChunkingConfig()
    result = NodeArtifact(
        kb_id=artifact.kb_id,
        source_instance_id=artifact.source_instance_id,
        source_document_id=source_document_id,
        extracted_artifact_path=extracted_path.as_posix(),
        extracted_checksum=sha256_bytes(extracted_path.read_bytes()),
        chunking=chunking_config,
        nodes=_build_nodes(artifact, artifact_path=extracted_path, config=chunking_config),
    )
    write_chunk_artifact(output_path, result, force=force)
    return result


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
    """Parse extracted documents for one source instance into text nodes."""
    root = Path(rag_data_root)
    extracted_paths = sorted(
        (root / "source_instances" / source_instance_id / "extracted").glob("*.json")
    )
    if document_ids is not None:
        selected_ids = {
            safe_document_id(candidate)
            for document_id in document_ids
            for candidate in (document_id, f"{source_instance_id}:{document_id}")
        }
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
            source_document_id = str(artifact.document.metadata["source_document_id"])
            output_path = chunk_artifact_path(
                rag_data_root=root,
                kb_id=artifact.kb_id,
                source_instance_id=artifact.source_instance_id,
                source_document_id=source_document_id,
            )
            from_cache = output_path.exists() and not force
            node_artifact = chunk_extracted_artifact(
                extracted_path,
                rag_data_root=root,
                config=config,
                force=force,
            )
            summary.chunk_count += len(node_artifact.nodes)
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
