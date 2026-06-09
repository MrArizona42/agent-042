from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from rag.domain import (
    Chunk,
    DocumentSection,
    ExtractedDocument,
    IndexManifest,
    RetrievalCapability,
    SourceDocument,
    compare_manifest_attestation,
    manifest_path,
    read_index_manifest,
    with_manifest_id,
    write_index_manifest,
)
from rag.domain.llamaindex import (
    chunk_to_text_node,
    extracted_document_to_llama_document,
)


def _created_at() -> datetime:
    return datetime(2026, 6, 3, 12, 0, tzinfo=timezone.utc)


def _manifest() -> IndexManifest:
    return IndexManifest(
        kb_id="ml_papers_core",
        collection_name="ml_papers_core_20260603_120000",
        alias="challenger",
        source_snapshot_id="sources:abc123",
        source_manifest_ref="assets/rag_data/ml_papers_core/sources.toml",
        document_count=2,
        chunk_count=12,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        sparse_encoder="fastembed/bm25",
        retrieval_capability=RetrievalCapability.HYBRID,
        chunking_config={"strategy": "section_aware", "chunk_size": 512},
        extraction_config={"method": "pdf_text"},
        eval_summary={"smoke_passed": True},
        created_at=_created_at(),
    )


def test_source_and_extracted_document_contracts_validate_required_text() -> None:
    source = SourceDocument(
        id="arxiv:1706.03762",
        source_type="arxiv_paper",
        uri="https://arxiv.org/abs/1706.03762",
        title="Attention Is All You Need",
        authors=[" Ashish Vaswani ", ""],
        metadata={"tags": ["transformer"]},
    )
    extracted = ExtractedDocument(
        id="doc-1",
        source_document_id=source.id,
        text="## Introduction\nTransformer models...",
        sections=[
            DocumentSection(
                title="Introduction",
                text="Transformer models...",
                level=2,
                ordinal=0,
            )
        ],
        extraction_method="pdf_text",
    )

    assert source.authors == ["Ashish Vaswani"]
    assert extracted.sections[0].title == "Introduction"

    with pytest.raises(ValidationError, match="section text"):
        DocumentSection(text=" ", ordinal=0)


def test_manifest_round_trip_sets_and_validates_deterministic_manifest_id(tmp_path) -> None:
    path = manifest_path(
        rag_data_root=tmp_path,
        kb_id="ml_papers_core",
        collection_name="ml_papers_core_20260603_120000",
    )

    written = write_index_manifest(path, _manifest())
    loaded = read_index_manifest(path)

    assert path.as_posix().endswith("ml_papers_core/manifests/ml_papers_core_20260603_120000.json")
    assert written.manifest_id is not None
    assert loaded == written

    payload = loaded.model_dump(mode="json", exclude_none=True)
    payload["chunk_count"] = 13
    with pytest.raises(ValidationError, match="manifest_id does not match"):
        IndexManifest.model_validate(payload)


def test_collection_attestation_is_compact_runtime_metadata() -> None:
    manifest = with_manifest_id(_manifest())
    attestation = manifest.to_attestation()

    assert attestation.manifest_id == manifest.manifest_id
    assert attestation.kb_id == "ml_papers_core"
    assert attestation.collection_name == "ml_papers_core_20260603_120000"
    assert attestation.embedding_model == manifest.embedding_model
    assert attestation.chunk_count == 12
    assert not hasattr(attestation, "source_manifest_ref")


def test_manifest_attestation_comparison_reports_drift() -> None:
    manifest = with_manifest_id(_manifest())
    matching = compare_manifest_attestation(manifest, manifest.to_attestation())

    drifted = manifest.to_attestation().model_copy(
        update={"embedding_model": "other-embedding-model"}
    )
    comparison = compare_manifest_attestation(manifest, drifted)

    assert matching.matches is True
    assert comparison.matches is False
    assert comparison.mismatches["embedding_model"] == (
        "sentence-transformers/all-MiniLM-L6-v2",
        "other-embedding-model",
    )


def test_llamaindex_adapters_keep_project_metadata() -> None:
    extracted = ExtractedDocument(
        id="doc-1",
        source_document_id="arxiv:1706.03762",
        text="Transformer text",
        extraction_method="pdf_text",
        metadata={"title": "Attention Is All You Need"},
    )
    chunk = Chunk(
        id="chunk-1",
        document_id="doc-1",
        source_document_id="arxiv:1706.03762",
        text="Self-attention text",
        section_title="Attention",
        ordinal=3,
        token_count=42,
        metadata={"uri": "https://arxiv.org/abs/1706.03762"},
    )

    llama_doc = extracted_document_to_llama_document(extracted)
    node = chunk_to_text_node(chunk)

    assert llama_doc.text == "Transformer text"
    assert llama_doc.metadata["source_document_id"] == "arxiv:1706.03762"
    assert node.text == "Self-attention text"
    assert node.metadata["chunk_id"] == "chunk-1"
    assert node.metadata["section_title"] == "Attention"
