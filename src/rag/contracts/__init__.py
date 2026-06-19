"""Stable RAG data contracts and artifact helpers."""

from rag.contracts.manifests import (
    canonical_manifest_payload,
    compare_manifest_attestation,
    compute_manifest_id,
    manifest_path,
    read_index_manifest,
    with_manifest_id,
    write_index_manifest,
)
from rag.contracts.models import (
    CollectionAttestation,
    IndexManifest,
    ManifestComparison,
    RetrievalCapability,
)
from rag.contracts.prompts import (
    DEFAULT_RAG_QUERY_PROMPTS,
    ProjectQueryPrompts,
    PromptIdentity,
)

__all__ = [
    "CollectionAttestation",
    "IndexManifest",
    "ManifestComparison",
    "RetrievalCapability",
    "DEFAULT_RAG_QUERY_PROMPTS",
    "ProjectQueryPrompts",
    "PromptIdentity",
    "canonical_manifest_payload",
    "compare_manifest_attestation",
    "compute_manifest_id",
    "manifest_path",
    "read_index_manifest",
    "with_manifest_id",
    "write_index_manifest",
]
