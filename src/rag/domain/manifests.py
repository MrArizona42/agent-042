"""Compatibility exports for RAG manifest helpers."""

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

__all__ = [
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
