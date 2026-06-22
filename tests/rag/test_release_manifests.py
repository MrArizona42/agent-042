"""Contract tests for release manifest immutability and attestation v2.

Covers `rag.contracts.manifests`' release-manifest functions: immutability on
write, manifest/attestation round-tripping, and that a schema-version-1
attestation payload is rejected by the new `ReleaseAttestation` model.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from app_config.catalog.schema import AliasBuildConfig
from rag.contracts.manifests import (
    compare_release_attestation,
    read_release_manifest,
    release_to_attestation,
    with_release_manifest_id,
    write_release_manifest,
)
from rag.contracts.models import ReleaseAttestation
from rag.control_plane.models import RagRelease


def _build_config(**overrides) -> AliasBuildConfig:
    defaults = {
        "chunking": {"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 64},
        "dense_encoder": {"model": "minilm", "dimension": 384},
    }
    defaults.update(overrides)
    return AliasBuildConfig(**defaults)


def _release(**overrides) -> RagRelease:
    defaults = dict(
        id="ragrel_pytorch_reference_abc123",
        kb_id="pytorch_reference",
        collection_name="rag__pytorch_reference__abc123",
        manifest_id="",
        release_fingerprint="sha256:f",
        catalog_digest="sha256:a",
        build_config_digest="sha256:b",
        source_declaration_digest="sha256:d",
        source_snapshot_id="sha256:s",
        build_config=_build_config(),
        source_manifest_digests={"pytorch_reference.docs": "sha256:x"},
        source_adapter_versions={"generic.http_html": "1"},
        document_count=10,
        chunk_count=100,
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return RagRelease(**defaults)


class TestWriteReleaseManifestImmutability:
    def test_first_write_creates_file(self, tmp_path: Path):
        path = tmp_path / "release.json"
        release = with_release_manifest_id(_release())

        written = write_release_manifest(path, release)

        assert path.exists()
        assert written.manifest_id == release.manifest_id

    def test_rewriting_identical_payload_is_a_noop(self, tmp_path: Path):
        path = tmp_path / "release.json"
        created_at = datetime.now(timezone.utc)
        release = with_release_manifest_id(_release(created_at=created_at))

        write_release_manifest(path, release)
        second = write_release_manifest(path, _release(created_at=created_at))

        assert second.manifest_id == release.manifest_id

    def test_rewriting_different_payload_raises(self, tmp_path: Path):
        path = tmp_path / "release.json"
        write_release_manifest(path, _release())

        with pytest.raises(ValueError, match="immutable"):
            write_release_manifest(path, _release(chunk_count=999))

    def test_round_trip_read(self, tmp_path: Path):
        path = tmp_path / "release.json"
        written = write_release_manifest(path, _release())

        loaded = read_release_manifest(path)

        assert loaded == written


class TestReleaseToAttestation:
    def test_dense_release_maps_to_attestation(self):
        release = with_release_manifest_id(_release())

        attestation = release_to_attestation(release)

        assert attestation.schema_version == 2
        assert attestation.release_id == release.id
        assert attestation.retrieval_capability == "dense"
        assert attestation.sparse_encoder_model is None

    def test_hybrid_release_maps_sparse_encoder(self):
        release = with_release_manifest_id(
            _release(build_config=_build_config(sparse_encoder={"model": "bm25"}))
        )

        attestation = release_to_attestation(release)

        assert attestation.retrieval_capability == "hybrid"
        assert attestation.sparse_encoder_model == "bm25"

    def test_compare_release_attestation_matches(self):
        release = with_release_manifest_id(_release())
        attestation = release_to_attestation(release)

        comparison = compare_release_attestation(release, attestation)

        assert comparison.matches is True
        assert comparison.mismatches == {}

    def test_compare_release_attestation_detects_drift(self):
        release = with_release_manifest_id(_release())
        attestation = release_to_attestation(release).model_copy(
            update={"chunk_count": release.chunk_count + 1}
        )

        comparison = compare_release_attestation(release, attestation)

        assert comparison.matches is False
        assert "chunk_count" in comparison.mismatches


class TestAttestationV1Rejected:
    def test_v1_shaped_payload_is_rejected_by_release_attestation(self):
        v1_payload = {
            "schema_version": 1,
            "manifest_id": "sha256:" + "a" * 64,
            "kb_id": "pytorch_reference",
            "collection_name": "rag__pytorch_reference__20260101_000000",
            "embedding_model": "minilm",
            "retrieval_capability": "dense",
            "chunk_count": 10,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        with pytest.raises(ValidationError):
            ReleaseAttestation.model_validate(v1_payload)
