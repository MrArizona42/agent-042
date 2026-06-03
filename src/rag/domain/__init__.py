"""RAG domain contracts and artifact helpers."""

from rag.domain.manifests import (
    attestation_from_legacy_collection_meta,
    attestation_from_payload,
    attestation_payload,
    canonical_manifest_payload,
    compare_manifest_attestation,
    compute_manifest_id,
    manifest_from_legacy_collection_meta,
    manifest_path,
    read_index_manifest,
    with_manifest_id,
    write_index_manifest,
)
from rag.domain.models import (
    Chunk,
    CollectionAttestation,
    DocumentSection,
    ExtractedDocument,
    IndexManifest,
    ManifestComparison,
    RetrievalCapability,
    RetrievalHit,
    SourceDocument,
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
    "attestation_from_legacy_collection_meta",
    "attestation_from_payload",
    "attestation_payload",
    "canonical_manifest_payload",
    "compare_manifest_attestation",
    "compute_manifest_id",
    "manifest_from_legacy_collection_meta",
    "manifest_path",
    "read_index_manifest",
    "with_manifest_id",
    "write_index_manifest",
]
