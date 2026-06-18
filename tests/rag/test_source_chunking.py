from __future__ import annotations

import uuid
from pathlib import Path

from llama_index.core import Document

from rag.contracts.metadata import source_document
from rag.sources.artifacts import (
    ExtractedDocumentArtifact,
    ExtractionArtifactMeta,
    RawArtifactRef,
    extracted_artifact_path,
    write_extracted_artifact,
)
from rag.sources.chunks import (
    ChunkingConfig,
    chunk_artifact_path,
    chunk_extracted_artifact,
    chunk_source_instance,
    read_chunk_artifact,
)


def _write_extracted_artifact(
    root: Path,
    *,
    source_document_id: str = "docs:tensors",
    text: str = "First sentence. Second sentence. Third sentence.",
) -> Path:
    local_id = source_document_id.rsplit(":", 1)[-1]
    source = source_document(
        local_document_id=local_id,
        title="Tensors",
        source_uri="https://docs.test/tensors.html",
        kb_id="pytorch_reference",
        source_instance_id="docs",
        adapter_id="generic.http_html",
        adapter_version="1",
        manifest_digest="sha256:manifest",
    )
    extracted = Document(
        id_=source.id_,
        text=text,
        metadata={
            **source.metadata,
            "extraction_method": "html_bs4",
            "extraction_warnings": [],
            "sections": [
                {
                    "title": "Overview",
                    "text": text,
                    "level": 1,
                    "ordinal": 0,
                    "metadata": {"anchor": "overview"},
                }
            ],
        },
    )
    artifact = ExtractedDocumentArtifact(
        kb_id="pytorch_reference",
        source_instance_id="docs",
        source_document=source,
        raw=RawArtifactRef(path=(root / "raw.html").as_posix(), checksum="sha256:raw"),
        extraction=ExtractionArtifactMeta(method="html_bs4"),
        document=extracted,
    )
    path = extracted_artifact_path(
        rag_data_root=root,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        source_document_id=source.id_,
    )
    write_extracted_artifact(path, artifact)
    return path


def test_chunk_extracted_artifact_writes_native_text_nodes(tmp_path: Path) -> None:
    extracted_path = _write_extracted_artifact(tmp_path)
    artifact = chunk_extracted_artifact(
        extracted_path,
        rag_data_root=tmp_path,
        config=ChunkingConfig(chunk_size=16, chunk_overlap=2),
    )
    restored = read_chunk_artifact(
        chunk_artifact_path(
            rag_data_root=tmp_path,
            kb_id="pytorch_reference",
            source_instance_id="docs",
            source_document_id="docs:tensors",
        )
    )

    assert artifact.schema_version == 2
    assert restored.source_document_id == "docs:tensors"
    assert restored.nodes[0].metadata["chunk_id"] == "docs:tensors:chunk:0000"
    assert restored.nodes[0].metadata["document_id"] == "docs:tensors"
    assert restored.nodes[0].metadata["section_title"] == "Overview"
    assert restored.nodes[0].metadata["anchor"] == "overview"
    assert str(uuid.UUID(restored.nodes[0].id_)) == restored.nodes[0].id_


def test_chunk_extracted_artifact_reuses_cache_unless_forced(tmp_path: Path) -> None:
    extracted_path = _write_extracted_artifact(tmp_path)
    config = ChunkingConfig(chunk_size=16, chunk_overlap=2)
    first = chunk_extracted_artifact(extracted_path, rag_data_root=tmp_path, config=config)
    cached = chunk_extracted_artifact(extracted_path, rag_data_root=tmp_path, config=config)
    forced = chunk_extracted_artifact(
        extracted_path,
        rag_data_root=tmp_path,
        config=config,
        force=True,
    )

    assert cached.nodes[0].id_ == first.nodes[0].id_
    assert forced.nodes[0].id_ == first.nodes[0].id_


def test_chunk_source_instance_summarizes_cache_and_filters(tmp_path: Path) -> None:
    _write_extracted_artifact(tmp_path, source_document_id="docs:tensors")
    _write_extracted_artifact(tmp_path, source_document_id="docs:torch", text="Torch sentence.")
    config = ChunkingConfig(chunk_size=16, chunk_overlap=2)

    first = chunk_source_instance(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        document_ids=["docs:tensors"],
        config=config,
    )
    second = chunk_source_instance(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        document_ids=["docs:tensors"],
        config=config,
    )

    assert first.total_selected == 1
    assert first.chunked == 1
    assert first.chunk_count >= 1
    assert second.from_cache == 1


def test_chunk_source_instance_records_failures(tmp_path: Path) -> None:
    extracted_dir = tmp_path / "source_instances" / "docs" / "extracted"
    extracted_dir.mkdir(parents=True)
    (extracted_dir / "broken.json").write_text('{"broken": true}\n', encoding="utf-8")

    summary = chunk_source_instance(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="docs",
    )

    assert summary.chunked == 0
    assert summary.failed[0]["error_type"] == "ValidationError"
