from __future__ import annotations

from pathlib import Path

import pytest

from rag.domain import DocumentSection, ExtractedDocument, SourceDocument
from rag.sources.artifacts import (
    ExtractedDocumentArtifact,
    ExtractionArtifactMeta,
    RawArtifactRef,
    extracted_artifact_path,
    write_extracted_artifact,
)
from rag.sources.bundles import collect_source_chunks
from rag.sources.chunks import ChunkingConfig, chunk_extracted_artifact


def _write_chunk_artifact(
    root: Path,
    *,
    source_document_id: str,
    text: str,
) -> None:
    source_document = SourceDocument(
        id=source_document_id,
        source_type="html_docs",
        uri="https://docs.test/page.html",
        title=source_document_id,
    )
    extracted = ExtractedDocument(
        id=source_document_id,
        source_document_id=source_document_id,
        text=text,
        sections=[DocumentSection(title="Overview", text=text, level=1, ordinal=0)],
        extraction_method="html_bs4",
    )
    artifact = ExtractedDocumentArtifact(
        kb_id="pytorch_reference",
        source_instance_id="docs",
        source_type="html_docs",
        source_document=source_document,
        raw=RawArtifactRef(path=(root / "raw.html").as_posix(), checksum="sha256:raw"),
        extraction=ExtractionArtifactMeta(method="html_bs4"),
        document=extracted,
    )
    path = extracted_artifact_path(
        rag_data_root=root,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        source_document_id=source_document_id,
    )
    write_extracted_artifact(path, artifact)
    chunk_extracted_artifact(
        path,
        rag_data_root=root,
        config=ChunkingConfig(chunk_size=32, chunk_overlap=4),
    )


def test_collect_source_chunks_returns_materialization_bundle(tmp_path: Path) -> None:
    _write_chunk_artifact(
        tmp_path,
        source_document_id="html:tensors",
        text="Tensor text. More tensor text.",
    )
    _write_chunk_artifact(
        tmp_path,
        source_document_id="html:torch",
        text="Torch text. More torch text.",
    )

    bundle = collect_source_chunks(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="docs",
    )

    assert bundle.kb_id == "pytorch_reference"
    assert bundle.source_instance_id == "docs"
    assert bundle.source_types == ["html_docs"]
    assert bundle.document_count == 2
    assert bundle.chunk_count == len(bundle.chunks)
    assert len(bundle.chunk_artifact_paths) == 2
    assert all(
        checksum.startswith("sha256:")
        for checksum in bundle.chunk_artifact_checksums.values()
    )


def test_collect_source_chunks_filters_document_ids_and_limit(tmp_path: Path) -> None:
    _write_chunk_artifact(tmp_path, source_document_id="html:tensors", text="Tensor text.")
    _write_chunk_artifact(tmp_path, source_document_id="html:torch", text="Torch text.")

    bundle = collect_source_chunks(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        document_ids=["html:tensors", "html:missing"],
        limit=1,
    )

    assert bundle.document_count == 1
    assert {chunk.source_document_id for chunk in bundle.chunks} == {"html:tensors"}


def test_collect_source_chunks_is_strict_for_corrupt_artifacts(tmp_path: Path) -> None:
    chunk_dir = tmp_path / "pytorch_reference" / "chunks" / "docs"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "broken.json").write_text('{"broken": true}\n', encoding="utf-8")

    with pytest.raises(ValueError):
        collect_source_chunks(
            rag_data_root=tmp_path,
            kb_id="pytorch_reference",
            source_instance_id="docs",
        )
