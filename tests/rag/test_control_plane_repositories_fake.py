"""Behavioral tests for the RAG control-plane repository contract.

Exercised against the in-memory fakes (tests/rag/control_plane_fakes.py)
since this repo has no live Postgres in its test environment. These prove
the *contract* phase 3 promises -- the real Postgres-backed implementation
in rag.control_plane.postgres must uphold the same contract (single-
transaction supersede+activate, immutable release rows, no release row on a
failed attempt) and additionally relies on the partial unique index in
rag_alias_deployments.sql as a second guard against cross-process races,
which only a live database can enforce.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app_config.catalog.schema import AliasBuildConfig, AliasRetrievalConfig
from rag.control_plane.models import AliasDeployment, RagRelease, ReleaseBuildAttempt
from tests.rag.control_plane_fakes import (
    FakeAliasDeploymentRepository,
    FakeReleaseBuildRepository,
    FakeReleaseRepository,
)


def _build_config() -> AliasBuildConfig:
    return AliasBuildConfig(
        chunking={"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 64},
        dense_encoder={"model": "minilm", "dimension": 384},
    )


def _release(release_id: str = "ragrel_pytorch_reference_abc123") -> RagRelease:
    return RagRelease(
        id=release_id,
        kb_id="pytorch_reference",
        collection_name=f"rag__pytorch_reference__{release_id[-6:]}",
        manifest_id=f"sha256:{release_id}",
        release_fingerprint=f"sha256:fp-{release_id}",
        catalog_digest="sha256:a",
        build_config_digest="sha256:b",
        source_declaration_digest="sha256:d",
        source_snapshot_id="sha256:s",
        build_config=_build_config(),
        source_manifest_digests={},
        source_adapter_versions={},
        document_count=1,
        chunk_count=1,
        created_at=datetime.now(timezone.utc),
    )


def _deployment(*, release_id: str, status: str = "pending") -> AliasDeployment:
    return AliasDeployment(
        id=uuid4(),
        kb_id="pytorch_reference",
        alias="challenger",
        release_id=release_id,
        collection_name=f"rag__pytorch_reference__{release_id[-6:]}",
        catalog_digest="sha256:a",
        build_config_digest="sha256:b",
        retrieval_config_digest="sha256:c",
        retrieval_config=AliasRetrievalConfig(strategy="dense", top_k=5, score_threshold=0.35),
        status=status,
    )


class TestReleaseBuildRepositoryContract:
    def test_failed_attempt_never_creates_a_release(self):
        builds = FakeReleaseBuildRepository()
        releases = FakeReleaseRepository()
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
        builds.create(attempt)

        builds.mark_failed(attempt.id, error="boom", finished_at=datetime.now(timezone.utc))

        assert builds.get(attempt.id).status == "failed"
        assert releases.list_for_kb("pytorch_reference") == []

    def test_completed_attempt_records_release_identity(self):
        builds = FakeReleaseBuildRepository()
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
        builds.create(attempt)

        builds.mark_completed(
            attempt.id,
            release_id="ragrel_pytorch_reference_abc123",
            collection_name="rag__pytorch_reference__abc123",
            source_snapshot_id="sha256:s",
            finished_at=datetime.now(timezone.utc),
        )

        completed = builds.get(attempt.id)
        assert completed.status == "completed"
        assert completed.release_id == "ragrel_pytorch_reference_abc123"


class TestReleaseRepositoryImmutability:
    def test_release_rows_are_immutable(self):
        releases = FakeReleaseRepository()
        releases.insert(_release(), manifest_path="x.json")

        with pytest.raises(ValueError, match="immutable"):
            releases.insert(_release(), manifest_path="x.json")

    def test_find_reusable_matches_on_build_and_source_identity(self):
        releases = FakeReleaseRepository()
        release = _release()
        releases.insert(release, manifest_path="x.json")

        found = releases.find_reusable(
            build_config_digest=release.build_config_digest,
            source_declaration_digest=release.source_declaration_digest,
            source_snapshot_id=release.source_snapshot_id,
        )

        assert [r.id for r in found] == [release.id]

    def test_find_reusable_does_not_match_different_source_snapshot(self):
        releases = FakeReleaseRepository()
        release = _release()
        releases.insert(release, manifest_path="x.json")

        found = releases.find_reusable(
            build_config_digest=release.build_config_digest,
            source_declaration_digest=release.source_declaration_digest,
            source_snapshot_id="sha256:different",
        )

        assert found == []


class TestAliasDeploymentRepositoryUniqueness:
    def test_only_one_active_deployment_per_kb_alias(self):
        deployments = FakeAliasDeploymentRepository()
        first = _deployment(release_id="ragrel_a")
        deployments.create_pending(first)
        deployments.activate(first.id, applied_at=datetime.now(timezone.utc))

        second = _deployment(release_id="ragrel_b")
        deployments.create_pending(second)
        deployments.activate(second.id, applied_at=datetime.now(timezone.utc))

        active = [
            row
            for row in deployments.list_history(kb_id="pytorch_reference", alias="challenger")
            if row.status == "active"
        ]
        assert len(active) == 1
        assert active[0].id == second.id

    def test_activating_supersedes_previous_active_deployment(self):
        deployments = FakeAliasDeploymentRepository()
        first = _deployment(release_id="ragrel_a")
        deployments.create_pending(first)
        deployments.activate(first.id, applied_at=datetime.now(timezone.utc))

        second = _deployment(release_id="ragrel_b")
        deployments.create_pending(second)
        deployments.activate(second.id, applied_at=datetime.now(timezone.utc))

        previous = deployments.get(first.id)
        assert previous.status == "superseded"
        assert previous.superseded_at is not None

    def test_no_drift_apply_is_idempotent_no_active_deployment_disappears(self):
        deployments = FakeAliasDeploymentRepository()
        deployment = _deployment(release_id="ragrel_a")
        deployments.create_pending(deployment)
        deployments.activate(deployment.id, applied_at=datetime.now(timezone.utc))

        # Re-activating the same deployment (idempotent re-apply) must not
        # supersede itself.
        deployments.activate(deployment.id, applied_at=datetime.now(timezone.utc))

        assert deployments.get_active(kb_id="pytorch_reference", alias="challenger").id == (
            deployment.id
        )

    def test_failed_pending_deployment_does_not_affect_active_one(self):
        deployments = FakeAliasDeploymentRepository()
        active = _deployment(release_id="ragrel_a")
        deployments.create_pending(active)
        deployments.activate(active.id, applied_at=datetime.now(timezone.utc))

        retry = _deployment(release_id="ragrel_b")
        deployments.create_pending(retry)
        deployments.mark_failed(retry.id, error="qdrant unavailable")

        assert deployments.get_active(kb_id="pytorch_reference", alias="challenger").id == (
            active.id
        )
        assert deployments.get(retry.id).status == "failed"

    def test_different_alias_does_not_collide(self):
        deployments = FakeAliasDeploymentRepository()
        champion = _deployment(release_id="ragrel_a")
        champion = champion.model_copy(update={"alias": "champion"})
        deployments.create_pending(champion)
        deployments.activate(champion.id, applied_at=datetime.now(timezone.utc))

        challenger = _deployment(release_id="ragrel_b")
        deployments.create_pending(challenger)
        deployments.activate(challenger.id, applied_at=datetime.now(timezone.utc))

        assert deployments.get_active(kb_id="pytorch_reference", alias="champion").id == (
            champion.id
        )
        assert deployments.get_active(kb_id="pytorch_reference", alias="challenger").id == (
            challenger.id
        )

    def test_list_history_orders_newest_first(self):
        deployments = FakeAliasDeploymentRepository()
        first = _deployment(release_id="ragrel_a")
        deployments.create_pending(first)
        second = _deployment(release_id="ragrel_b")
        deployments.create_pending(second)

        history = deployments.list_history(kb_id="pytorch_reference", alias="challenger")

        assert [row.id for row in history] == [second.id, first.id]
