"""Tests for the preferred RAG contracts import surface."""

from __future__ import annotations

from datetime import datetime, timezone


def test_rag_contracts_are_legacy_domain_compatible():
    from rag.contracts import IndexManifest, RetrievalCapability, SourceDocument, with_manifest_id
    from rag.contracts.manifests import compute_manifest_id
    from rag.domain import IndexManifest as LegacyIndexManifest
    from rag.domain import SourceDocument as LegacySourceDocument
    from rag.domain.manifests import compute_manifest_id as legacy_compute_manifest_id
    from rag.domain.manifests import with_manifest_id as legacy_with_manifest_id

    manifest = IndexManifest(
        kb_id="pytorch_reference",
        collection_name="rag__pytorch_reference__test",
        source_snapshot_id="sha256:snapshot",
        document_count=1,
        chunk_count=2,
        embedding_model="BAAI/bge-small-en-v1.5",
        retrieval_capability=RetrievalCapability.DENSE,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert SourceDocument is LegacySourceDocument
    assert IndexManifest is LegacyIndexManifest
    assert compute_manifest_id is legacy_compute_manifest_id
    assert with_manifest_id is legacy_with_manifest_id
    assert with_manifest_id(manifest).manifest_id == legacy_with_manifest_id(manifest).manifest_id
