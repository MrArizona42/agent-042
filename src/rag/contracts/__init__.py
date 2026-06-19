"""Stable RAG data contracts and artifact helpers."""

from rag.contracts.manifests import (
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
from rag.contracts.models import (
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
from rag.contracts.prompts import (
    DEFAULT_RAG_QUERY_PROMPTS,
    ProjectQueryPrompts,
    PromptIdentity,
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
    "DEFAULT_RAG_QUERY_PROMPTS",
    "ProjectQueryPrompts",
    "PromptIdentity",
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
