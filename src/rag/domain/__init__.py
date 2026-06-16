"""Compatibility exports for RAG contracts.

New production code should import from ``rag.contracts``. This package remains
as a stable compatibility path while callers migrate away from ``rag.domain``.
"""

from rag.contracts import (
    Chunk,
    CollectionAttestation,
    DocumentSection,
    ExtractedDocument,
    IndexManifest,
    ManifestComparison,
    RetrievalCapability,
    RetrievalHit,
    SourceDocument,
    attestation_from_payload,
    attestation_payload,
    canonical_manifest_payload,
    compare_manifest_attestation,
    compute_manifest_id,
    manifest_path,
    read_index_manifest,
    with_manifest_id,
    write_index_manifest,
)

__all__ = [
    "Chunk",
    "CollectionAttestation",
    "DocumentSection",
    "ExtractedDocument",
    "IndexManifest",
    "ManifestComparison",
    "RetrievalCapability",
    "RetrievalHit",
    "SourceDocument",
    "attestation_from_payload",
    "attestation_payload",
    "canonical_manifest_payload",
    "compare_manifest_attestation",
    "compute_manifest_id",
    "manifest_path",
    "read_index_manifest",
    "with_manifest_id",
    "write_index_manifest",
]
