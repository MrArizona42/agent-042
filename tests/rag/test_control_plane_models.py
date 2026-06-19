"""Contract tests for `rag.control_plane.models`.

Covers the alias build/retrieve schema-v4 shapes and the release/build-attempt/
deployment/diff contracts: every model forbids unknown fields, and validators
described by the declarative alias workflow plan are enforced.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Catalog-shape types (app_config.catalog.schema), exercised here because the
# control plane contracts embed them.
# ---------------------------------------------------------------------------


class TestAliasChunkingConfig:
    def test_overlap_smaller_than_size_is_valid(self):
        from app_config.catalog.schema import AliasChunkingConfig

        cfg = AliasChunkingConfig(strategy="sentence", chunk_size=512, chunk_overlap=64)
        assert cfg.chunk_overlap == 64

    def test_overlap_equal_to_size_is_rejected(self):
        from app_config.catalog.schema import AliasChunkingConfig

        with pytest.raises(ValidationError, match="chunk_overlap"):
            AliasChunkingConfig(strategy="sentence", chunk_size=512, chunk_overlap=512)

    def test_overlap_greater_than_size_is_rejected(self):
        from app_config.catalog.schema import AliasChunkingConfig

        with pytest.raises(ValidationError, match="chunk_overlap"):
            AliasChunkingConfig(strategy="sentence", chunk_size=512, chunk_overlap=600)

    def test_unknown_field_is_rejected(self):
        from app_config.catalog.schema import AliasChunkingConfig

        with pytest.raises(ValidationError):
            AliasChunkingConfig(
                strategy="sentence", chunk_size=512, chunk_overlap=64, extra_field="x"
            )


class TestAliasBuildConfig:
    def test_dense_only_build_is_valid(self):
        from app_config.catalog.schema import AliasBuildConfig

        cfg = AliasBuildConfig(
            chunking={"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 64},
            dense_encoder={"model": "minilm", "dimension": 384},
        )
        assert cfg.sparse_encoder is None

    def test_hybrid_build_with_sparse_encoder_is_valid(self):
        from app_config.catalog.schema import AliasBuildConfig

        cfg = AliasBuildConfig(
            chunking={"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 64},
            dense_encoder={"model": "minilm", "dimension": 384},
            sparse_encoder={"model": "Qdrant/bm25"},
        )
        assert cfg.sparse_encoder.model == "Qdrant/bm25"

    def test_dense_encoder_dimension_must_be_positive(self):
        from app_config.catalog.schema import AliasBuildConfig

        with pytest.raises(ValidationError, match="dimension"):
            AliasBuildConfig(
                chunking={"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 64},
                dense_encoder={"model": "minilm", "dimension": 0},
            )

    def test_dense_encoder_is_required(self):
        from app_config.catalog.schema import AliasBuildConfig

        with pytest.raises(ValidationError, match="dense_encoder"):
            AliasBuildConfig(
                chunking={"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 64},
            )


class TestAliasRetrievalConfig:
    def test_omitted_reranker_requires_multiplier_one(self):
        from app_config.catalog.schema import AliasRetrievalConfig

        with pytest.raises(ValidationError, match="reranker_multiplier"):
            AliasRetrievalConfig(
                strategy="dense", top_k=5, score_threshold=0.35, reranker_multiplier=4
            )

    def test_omitted_reranker_with_default_multiplier_is_valid(self):
        from app_config.catalog.schema import AliasRetrievalConfig

        cfg = AliasRetrievalConfig(strategy="dense", top_k=5, score_threshold=0.35)
        assert cfg.reranker_multiplier == 1

    def test_reranker_with_multiplier_is_valid(self):
        from app_config.catalog.schema import AliasRetrievalConfig

        cfg = AliasRetrievalConfig(
            strategy="hybrid",
            top_k=5,
            score_threshold=0.01,
            reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
            reranker_multiplier=4,
        )
        assert cfg.reranker_multiplier == 4

    def test_top_k_must_be_positive(self):
        from app_config.catalog.schema import AliasRetrievalConfig

        with pytest.raises(ValidationError, match="top_k"):
            AliasRetrievalConfig(strategy="dense", top_k=0, score_threshold=0.35)


# ---------------------------------------------------------------------------
# Control-plane contracts
# ---------------------------------------------------------------------------


def _build_config():
    from app_config.catalog.schema import AliasBuildConfig

    return AliasBuildConfig(
        chunking={"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 64},
        dense_encoder={"model": "minilm", "dimension": 384},
    )


def _retrieval_config():
    from app_config.catalog.schema import AliasRetrievalConfig

    return AliasRetrievalConfig(strategy="dense", top_k=5, score_threshold=0.35)


class TestReleaseBuildAttempt:
    def test_complete_attempt_is_valid(self):
        from rag.control_plane.models import ReleaseBuildAttempt

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
        assert attempt.status == "running"
        assert attempt.release_id is None

    def test_unknown_field_is_rejected(self):
        from rag.control_plane.models import ReleaseBuildAttempt

        with pytest.raises(ValidationError):
            ReleaseBuildAttempt(
                id=uuid4(),
                kb_id="pytorch_reference",
                requested_alias="challenger",
                status="running",
                catalog_digest="sha256:a",
                build_config_digest="sha256:b",
                retrieval_config_digest="sha256:c",
                source_declaration_digest="sha256:d",
                started_at=datetime.now(timezone.utc),
                unknown="x",
            )

    def test_invalid_status_is_rejected(self):
        from rag.control_plane.models import ReleaseBuildAttempt

        with pytest.raises(ValidationError, match="status"):
            ReleaseBuildAttempt(
                id=uuid4(),
                kb_id="pytorch_reference",
                requested_alias="challenger",
                status="queued",
                catalog_digest="sha256:a",
                build_config_digest="sha256:b",
                retrieval_config_digest="sha256:c",
                source_declaration_digest="sha256:d",
                started_at=datetime.now(timezone.utc),
            )


class TestRagRelease:
    def test_complete_release_is_valid(self):
        from rag.control_plane.models import RagRelease

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
        assert release.schema_version == 1

    def test_no_alias_field_exists(self):
        from rag.control_plane.models import RagRelease

        assert "alias" not in RagRelease.model_fields


class TestAliasDeployment:
    def test_complete_deployment_is_valid(self):
        from rag.control_plane.models import AliasDeployment

        deployment = AliasDeployment(
            id=uuid4(),
            kb_id="pytorch_reference",
            alias="challenger",
            release_id="ragrel_pytorch_reference_abc123",
            collection_name="rag__pytorch_reference__abc123",
            catalog_digest="sha256:a",
            build_config_digest="sha256:b",
            retrieval_config_digest="sha256:c",
            retrieval_config=_retrieval_config(),
            status="active",
        )
        assert deployment.status == "active"

    def test_invalid_status_is_rejected(self):
        from rag.control_plane.models import AliasDeployment

        with pytest.raises(ValidationError, match="status"):
            AliasDeployment(
                id=uuid4(),
                kb_id="pytorch_reference",
                alias="challenger",
                release_id="ragrel_pytorch_reference_abc123",
                collection_name="rag__pytorch_reference__abc123",
                catalog_digest="sha256:a",
                build_config_digest="sha256:b",
                retrieval_config_digest="sha256:c",
                retrieval_config=_retrieval_config(),
                status="promoted",
            )


class TestAliasDiff:
    def test_no_drift_diff_is_valid(self):
        from rag.control_plane.models import AliasDiff

        diff = AliasDiff(
            kb_id="pytorch_reference",
            alias="champion",
            desired_catalog_digest="sha256:a",
            desired_build_config_digest="sha256:b",
            desired_retrieval_config_digest="sha256:c",
            applied_deployment_id=uuid4(),
            applied_release_id="ragrel_pytorch_reference_abc123",
            build_drift=False,
            retrieval_drift=False,
            source_declaration_drift=False,
            provider_mismatches=[],
            reusable_release_ids=[],
        )
        assert diff.build_drift is False

    def test_unapplied_alias_allows_none_ids(self):
        from rag.control_plane.models import AliasDiff

        diff = AliasDiff(
            kb_id="pytorch_reference",
            alias="challenger",
            desired_catalog_digest="sha256:a",
            desired_build_config_digest="sha256:b",
            desired_retrieval_config_digest="sha256:c",
            applied_deployment_id=None,
            applied_release_id=None,
            build_drift=True,
            retrieval_drift=True,
            source_declaration_drift=False,
            provider_mismatches=[],
            reusable_release_ids=[],
        )
        assert diff.applied_deployment_id is None
