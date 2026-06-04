from __future__ import annotations

from pathlib import Path

from rag.domain import DocumentSection, ExtractedDocument, SourceDocument
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
    source_document_id: str = "html:tensors",
    text: str = "First sentence. Second sentence. Third sentence.",
) -> Path:
    source_document = SourceDocument(
        id=source_document_id,
        source_type="html_docs",
        uri="https://docs.test/tensors.html",
        title="Tensors",
    )
    document = ExtractedDocument(
        id=source_document_id,
        source_document_id=source_document_id,
        text=text,
        sections=[
            DocumentSection(
                title="Overview",
                text=text,
                level=1,
                ordinal=0,
                metadata={"anchor": "overview"},
            )
        ],
        extraction_method="html_bs4",
    )
    artifact = ExtractedDocumentArtifact(
        kb_id="pytorch_reference",
        source_instance_id="docs",
        source_type="html_docs",
        source_document=source_document,
        raw=RawArtifactRef(
            path=(root / "raw.html").as_posix(),
            checksum="sha256:raw",
            content_type="text/html",
        ),
        extraction=ExtractionArtifactMeta(method="html_bs4"),
        document=document,
    )
    path = extracted_artifact_path(
        rag_data_root=root,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        source_document_id=source_document_id,
    )
    write_extracted_artifact(path, artifact)
    return path


def test_chunk_extracted_artifact_writes_project_chunks(tmp_path: Path) -> None:
    extracted_path = _write_extracted_artifact(tmp_path)

    artifact = chunk_extracted_artifact(
        extracted_path,
        rag_data_root=tmp_path,
        config=ChunkingConfig(chunk_size=16, chunk_overlap=2),
    )
    path = chunk_artifact_path(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        source_document_id="html:tensors",
    )
    restored = read_chunk_artifact(path)

    assert path.as_posix().endswith("pytorch_reference/chunks/docs/html_tensors.json")
    assert artifact.chunking.method == "llamaindex_sentence_splitter"
    assert restored.source_document_id == "html:tensors"
    assert restored.extracted_artifact_path == extracted_path.as_posix()
    assert restored.extracted_checksum.startswith("sha256:")
    assert len(restored.chunks) >= 1
    assert restored.chunks[0].id == "html:tensors:chunk:0000"
    assert restored.chunks[0].document_id == "html:tensors"
    assert restored.chunks[0].section_title == "Overview"
    assert restored.chunks[0].metadata["anchor"] == "overview"
    assert restored.chunks[0].metadata["source_type"] == "html_docs"


def test_chunk_extracted_artifact_reuses_cache_unless_forced(tmp_path: Path) -> None:
    extracted_path = _write_extracted_artifact(tmp_path)
    config = ChunkingConfig(chunk_size=16, chunk_overlap=2)
    chunk_extracted_artifact(extracted_path, rag_data_root=tmp_path, config=config)
    path = chunk_artifact_path(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        source_document_id="html:tensors",
    )
    original_text = read_chunk_artifact(path).chunks[0].text

    extracted_path.write_text(extracted_path.read_text(encoding="utf-8"), encoding="utf-8")
    cached = chunk_extracted_artifact(extracted_path, rag_data_root=tmp_path, config=config)
    forced = chunk_extracted_artifact(
        extracted_path,
        rag_data_root=tmp_path,
        config=config,
        force=True,
    )

    assert cached.chunks[0].text == original_text
    assert forced.chunks[0].text == original_text


def test_chunk_source_instance_summarizes_cache_and_filters(tmp_path: Path) -> None:
    _write_extracted_artifact(tmp_path, source_document_id="html:tensors")
    _write_extracted_artifact(
        tmp_path,
        source_document_id="html:torch",
        text="Torch sentence. Another torch sentence.",
    )
    config = ChunkingConfig(chunk_size=16, chunk_overlap=2)

    first = chunk_source_instance(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        document_ids=["html:tensors"],
        config=config,
    )
    second = chunk_source_instance(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        document_ids=["html:tensors"],
        config=config,
    )
    forced = chunk_source_instance(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        document_ids=["html:tensors"],
        config=config,
        force=True,
    )

    assert first.total_selected == 1
    assert first.chunked == 1
    assert first.from_cache == 0
    assert first.chunk_count >= 1
    assert first.failed == []
    assert second.chunked == 0
    assert second.from_cache == 1
    assert forced.chunked == 1
    assert forced.from_cache == 0


def test_chunk_source_instance_records_failures(tmp_path: Path) -> None:
    extracted_dir = tmp_path / "pytorch_reference" / "extracted" / "docs"
    extracted_dir.mkdir(parents=True)
    (extracted_dir / "broken.json").write_text('{"broken": true}\n', encoding="utf-8")

    summary = chunk_source_instance(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="docs",
    )

    assert summary.total_selected == 1
    assert summary.chunked == 0
    assert summary.chunk_count == 0
    assert len(summary.failed) == 1
    assert summary.failed[0]["error_type"] == "ValidationError"
