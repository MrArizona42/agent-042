"""Manifest artifact and collection-attestation helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rag.contracts.models import (
    CollectionAttestation,
    IndexManifest,
    ManifestComparison,
)


def canonical_manifest_payload(manifest: IndexManifest) -> dict[str, Any]:
    """Return the stable JSON payload used for manifest hashing."""
    return manifest.model_dump(
        mode="json",
        exclude={"manifest_id"},
        exclude_none=True,
    )


def compute_manifest_id(manifest: IndexManifest) -> str:
    """Compute a deterministic id from the manifest payload."""
    payload = canonical_manifest_payload(manifest)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def with_manifest_id(manifest: IndexManifest) -> IndexManifest:
    """Return a copy of *manifest* with its deterministic manifest id set."""
    manifest_id = compute_manifest_id(manifest)
    return manifest.model_copy(update={"manifest_id": manifest_id})


def manifest_path(
    *,
    rag_data_root: Path | str,
    kb_id: str,
    collection_name: str,
) -> Path:
    """Return the conventional artifact path for a collection manifest."""
    return Path(rag_data_root) / kb_id / "manifests" / f"{collection_name}.json"


def write_index_manifest(path: Path | str, manifest: IndexManifest) -> IndexManifest:
    """Write a manifest JSON artifact and return the id-bearing manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = with_manifest_id(manifest)
    path.write_text(
        json.dumps(
            manifest.model_dump(mode="json", exclude_none=True),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def read_index_manifest(path: Path | str) -> IndexManifest:
    """Read and validate a manifest JSON artifact."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return IndexManifest.model_validate(payload)


def compare_manifest_attestation(
    manifest: IndexManifest,
    attestation: CollectionAttestation,
) -> ManifestComparison:
    """Compare manifest artifact provenance to Qdrant runtime attestation."""
    manifest = with_manifest_id(manifest) if manifest.manifest_id is None else manifest
    expected = manifest.to_attestation()
    mismatches: dict[str, tuple[Any, Any]] = {}
    for field_name in (
        "manifest_id",
        "kb_id",
        "collection_name",
        "embedding_model",
        "sparse_encoder",
        "retrieval_capability",
        "chunk_count",
    ):
        expected_value = getattr(expected, field_name)
        actual_value = getattr(attestation, field_name)
        if expected_value != actual_value:
            mismatches[field_name] = (expected_value, actual_value)
    return ManifestComparison(matches=not mismatches, mismatches=mismatches)


def attestation_payload(attestation: CollectionAttestation) -> dict[str, Any]:
    """Return the Qdrant payload for collection metadata."""
    return {
        "metadata_kind": "collection_attestation",
        **attestation.model_dump(mode="json", exclude_none=True),
    }


def attestation_from_payload(payload: dict[str, Any]) -> CollectionAttestation:
    """Parse Qdrant collection metadata payload into an attestation."""
    payload = dict(payload)
    payload.pop("metadata_kind", None)
    payload.pop("type", None)
    return CollectionAttestation.model_validate(payload)
