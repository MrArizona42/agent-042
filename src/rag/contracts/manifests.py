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
    ReleaseAttestation,
)
from rag.control_plane.fingerprints import canonical_digest
from rag.control_plane.models import RagRelease


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
    return Path(rag_data_root) / "knowledge_bases" / kb_id / "manifests" / f"{collection_name}.json"


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


# ---------------------------------------------------------------------------
# Release manifests (rag.control_plane.models.RagRelease): immutable,
# content-identified successor to IndexManifest. The functions above stay in
# place for the old materialize_kb_collection_llamaindex() path until it is
# retired; new build code uses these instead.
# ---------------------------------------------------------------------------


def canonical_release_payload(release: RagRelease) -> dict[str, Any]:
    """Return the stable JSON payload used for release manifest hashing."""
    return release.model_dump(mode="json", exclude={"manifest_id"})


def compute_release_manifest_id(release: RagRelease) -> str:
    """Compute the deterministic manifest id from the release payload."""
    return canonical_digest(canonical_release_payload(release))


def with_release_manifest_id(release: RagRelease) -> RagRelease:
    """Return a copy of *release* with its deterministic manifest id set."""
    return release.model_copy(update={"manifest_id": compute_release_manifest_id(release)})


def release_manifest_path(
    *,
    rag_data_root: Path | str,
    kb_id: str,
    release_id: str,
) -> Path:
    """Return the conventional artifact path for a release manifest."""
    return Path(rag_data_root) / "knowledge_bases" / kb_id / "releases" / f"{release_id}.json"


def write_release_manifest(path: Path | str, release: RagRelease) -> RagRelease:
    """Write an immutable release manifest.

    Writing a different payload to an existing release id is an error: the
    manifest at *path* never changes once written. Writing an identical
    payload again is a no-op (idempotent reuse).
    """
    path = Path(path)
    release = with_release_manifest_id(release) if not release.manifest_id else release
    if path.exists():
        existing = read_release_manifest(path)
        if existing.manifest_id != release.manifest_id:
            raise ValueError(
                f"release manifest at {path} is immutable: existing manifest_id "
                f"{existing.manifest_id!r} does not match new payload's "
                f"{release.manifest_id!r}"
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(release.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return release


def read_release_manifest(path: Path | str) -> RagRelease:
    """Read and validate a release manifest JSON artifact."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return RagRelease.model_validate(payload)


def release_to_attestation(release: RagRelease) -> ReleaseAttestation:
    """Return the compact Qdrant-side metadata for a release."""
    return ReleaseAttestation(
        release_id=release.id,
        manifest_id=release.manifest_id,
        kb_id=release.kb_id,
        collection_name=release.collection_name,
        release_fingerprint=release.release_fingerprint,
        build_config_digest=release.build_config_digest,
        source_snapshot_id=release.source_snapshot_id,
        dense_encoder_model=release.build_config.dense_encoder.model,
        dense_vector_dimension=release.build_config.dense_encoder.dimension,
        sparse_encoder_model=(
            release.build_config.sparse_encoder.model
            if release.build_config.sparse_encoder
            else None
        ),
        retrieval_capability=(
            "hybrid" if release.build_config.sparse_encoder is not None else "dense"
        ),
        chunk_count=release.chunk_count,
        created_at=release.created_at,
    )


def compare_release_attestation(
    release: RagRelease,
    attestation: ReleaseAttestation,
) -> ManifestComparison:
    """Compare a release manifest's provenance to its Qdrant runtime attestation."""
    expected = release_to_attestation(release)
    mismatches: dict[str, tuple[Any, Any]] = {}
    for field_name in (
        "release_id",
        "manifest_id",
        "kb_id",
        "collection_name",
        "release_fingerprint",
        "build_config_digest",
        "source_snapshot_id",
        "dense_encoder_model",
        "dense_vector_dimension",
        "sparse_encoder_model",
        "retrieval_capability",
        "chunk_count",
    ):
        expected_value = getattr(expected, field_name)
        actual_value = getattr(attestation, field_name)
        if expected_value != actual_value:
            mismatches[field_name] = (expected_value, actual_value)
    return ManifestComparison(matches=not mismatches, mismatches=mismatches)
