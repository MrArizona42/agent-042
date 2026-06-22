"""Manifest artifact and collection-attestation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.contracts.models import ManifestComparison, ReleaseAttestation
from rag.control_plane.fingerprints import canonical_digest
from rag.control_plane.models import RagRelease


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
