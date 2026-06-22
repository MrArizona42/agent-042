"""Tests for rag.control_plane.postgres's domain <-> ORM row conversions.

These exercise the conversion functions directly (constructing ORM objects
in memory, no engine or database connection involved) since this repo has
no live Postgres available in its test environment. The repository classes'
SQL/transaction behavior is covered behaviorally by the fake repositories in
tests/rag/control_plane_fakes.py against the same Protocol contracts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app_config.catalog.schema import AliasBuildConfig, AliasRetrievalConfig
from rag.control_plane.models import AliasDeployment, RagRelease, ReleaseBuildAttempt
from rag.control_plane.postgres import (
    _attempt_to_row,
    _deployment_to_row,
    _release_to_row,
    _row_to_attempt,
    _row_to_deployment,
    _row_to_release,
)


def _build_config() -> AliasBuildConfig:
    return AliasBuildConfig(
        chunking={"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 64},
        dense_encoder={"model": "minilm", "dimension": 384},
    )


class TestReleaseBuildAttemptConversion:
    def test_round_trip(self):
        attempt = ReleaseBuildAttempt(
            id=uuid4(),
            kb_id="pytorch_reference",
            requested_alias="challenger",
            status="running",
            catalog_digest="sha256:a",
            build_config_digest="sha256:b",
            retrieval_config_digest="sha256:c",
            source_declaration_digest="sha256:d",
            started_at=datetime.now(timezone.utc),
        )

        row = _attempt_to_row(attempt)
        restored = _row_to_attempt(row)

        assert restored == attempt


class TestRagReleaseConversion:
    def test_round_trip(self):
        release = RagRelease(
            id="ragrel_pytorch_reference_abc123",
            kb_id="pytorch_reference",
            collection_name="rag__pytorch_reference__abc123",
            manifest_id="sha256:m",
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

        row = _release_to_row(release, manifest_path="assets/rag_data/knowledge_bases/x.json")
        restored = _row_to_release(row)

        assert restored == release
        assert row.manifest_path == "assets/rag_data/knowledge_bases/x.json"

    def test_row_has_no_alias_attribute(self):
        from shared.db.models import RagReleaseRow

        assert not hasattr(RagReleaseRow, "alias")


class TestAliasDeploymentConversion:
    def test_round_trip(self):
        deployment = AliasDeployment(
            id=uuid4(),
            kb_id="pytorch_reference",
            alias="challenger",
            release_id="ragrel_pytorch_reference_abc123",
            collection_name="rag__pytorch_reference__abc123",
            catalog_digest="sha256:a",
            build_config_digest="sha256:b",
            retrieval_config_digest="sha256:c",
            retrieval_config=AliasRetrievalConfig(strategy="dense", top_k=5, score_threshold=0.35),
            status="active",
        )

        row = _deployment_to_row(deployment)
        restored = _row_to_deployment(row)

        assert restored == deployment
