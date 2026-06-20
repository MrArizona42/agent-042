"""Tests for `rag alias diff/apply/status`, with an injected fake AliasService.

Per the plan's CLI testing strategy: typer.testing.CliRunner with injected
application services, not real Postgres/Qdrant/provider connections.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from uuid import uuid4

from typer.testing import CliRunner

import rag.cli.alias as alias_cli
from app_config.catalog.schema import AliasRetrievalConfig
from rag.cli.app import app
from rag.control_plane.models import AliasDeployment, AliasDiff, RagRelease

runner = CliRunner()


def _write_catalog(path: Path) -> Path:
    path.write_text(
        dedent(
            """
            schema_version = 4

            [[knowledge_bases]]
            id = "pytorch_reference"
            description = "PyTorch docs"
            default_alias = "champion"

            [knowledge_bases.aliases.champion.build.chunking]
            strategy = "sentence"
            chunk_size = 512
            chunk_overlap = 64

            [knowledge_bases.aliases.champion.build.dense_encoder]
            model = "test-embedding"
            dimension = 3

            [knowledge_bases.aliases.champion.retrieve]
            strategy = "dense"
            top_k = 5
            score_threshold = 0.35
            reranker_multiplier = 1
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _diff(
    *,
    build_drift: bool,
    retrieval_drift: bool = False,
    provider_mismatches: list[str] | None = None,
) -> AliasDiff:
    return AliasDiff(
        kb_id="pytorch_reference",
        alias="champion",
        desired_catalog_digest="sha256:a",
        desired_build_config_digest="sha256:b",
        desired_retrieval_config_digest="sha256:c",
        applied_deployment_id=None if build_drift else uuid4(),
        applied_release_id=None if build_drift else "ragrel_x",
        build_drift=build_drift,
        retrieval_drift=retrieval_drift,
        source_declaration_drift=False,
        provider_mismatches=provider_mismatches or [],
        reusable_release_ids=[],
    )


def _release() -> RagRelease:
    from app_config.catalog.schema import AliasBuildConfig

    return RagRelease(
        id="ragrel_pytorch_reference_abc123",
        kb_id="pytorch_reference",
        collection_name="rag__pytorch_reference__abc123",
        manifest_id="sha256:m",
        release_fingerprint="sha256:f",
        catalog_digest="sha256:a",
        build_config_digest="sha256:b",
        source_declaration_digest="sha256:d",
        source_snapshot_id="sha256:s",
        build_config=AliasBuildConfig(
            chunking={"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 64},
            dense_encoder={"model": "test-embedding", "dimension": 3},
        ),
        source_manifest_digests={},
        source_adapter_versions={},
        document_count=1,
        chunk_count=1,
        created_at=datetime.now(timezone.utc),
    )


class _FakeAliasService:
    def __init__(self, *, diff_result=None, diff_error=None, apply_result=None, apply_error=None):
        self._diff_result = diff_result
        self._diff_error = diff_error
        self._apply_result = apply_result
        self._apply_error = apply_error
        self.apply_calls = []

    def diff(self, request):
        if self._diff_error is not None:
            raise self._diff_error
        return self._diff_result

    def apply(self, request):
        self.apply_calls.append(request)
        if self._apply_error is not None:
            raise self._apply_error
        return self._apply_result


def test_diff_exits_zero_when_no_drift(tmp_path, monkeypatch):
    catalog_path = _write_catalog(tmp_path / "catalog.toml")
    fake = _FakeAliasService(diff_result=_diff(build_drift=False, retrieval_drift=False))
    monkeypatch.setattr(alias_cli, "build_alias_service", lambda ctx: fake)

    result = runner.invoke(
        app, ["--catalog", str(catalog_path), "alias", "diff", "pytorch_reference", "champion"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["build_drift"] is False


def test_diff_exits_one_when_drift_exists(tmp_path, monkeypatch):
    catalog_path = _write_catalog(tmp_path / "catalog.toml")
    fake = _FakeAliasService(diff_result=_diff(build_drift=True))
    monkeypatch.setattr(alias_cli, "build_alias_service", lambda ctx: fake)

    result = runner.invoke(
        app, ["--catalog", str(catalog_path), "alias", "diff", "pytorch_reference", "champion"]
    )

    assert result.exit_code == 1


def test_diff_exits_one_when_provider_identity_mismatches(tmp_path, monkeypatch):
    catalog_path = _write_catalog(tmp_path / "catalog.toml")
    fake = _FakeAliasService(
        diff_result=_diff(
            build_drift=False,
            provider_mismatches=["dense encoder provider mismatch"],
        )
    )
    monkeypatch.setattr(alias_cli, "build_alias_service", lambda ctx: fake)

    result = runner.invoke(
        app, ["--catalog", str(catalog_path), "alias", "diff", "pytorch_reference", "champion"]
    )

    assert result.exit_code == 1


def test_diff_exits_two_on_unknown_kb(tmp_path, monkeypatch):
    from rag.control_plane.alias_service import AliasApplyError

    catalog_path = _write_catalog(tmp_path / "catalog.toml")
    fake = _FakeAliasService(diff_error=AliasApplyError("Unknown KB 'nope'"))
    monkeypatch.setattr(alias_cli, "build_alias_service", lambda ctx: fake)

    result = runner.invoke(
        app, ["--catalog", str(catalog_path), "alias", "diff", "nope", "champion"]
    )

    assert result.exit_code == 2


def test_apply_exits_zero_and_prints_deployment(tmp_path, monkeypatch):
    from rag.control_plane.alias_service import AliasApplyResult

    catalog_path = _write_catalog(tmp_path / "catalog.toml")
    release = _release()
    deployment = AliasDeployment(
        id=uuid4(),
        kb_id="pytorch_reference",
        alias="champion",
        release_id=release.id,
        collection_name=release.collection_name,
        catalog_digest="sha256:a",
        build_config_digest="sha256:b",
        retrieval_config_digest="sha256:c",
        retrieval_config=AliasRetrievalConfig(strategy="dense", top_k=5, score_threshold=0.35),
        status="active",
    )
    fake = _FakeAliasService(
        apply_result=AliasApplyResult(deployment=deployment, release=release, action="no_drift")
    )
    monkeypatch.setattr(alias_cli, "build_alias_service", lambda ctx: fake)

    result = runner.invoke(
        app, ["--catalog", str(catalog_path), "alias", "apply", "pytorch_reference", "champion"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["action"] == "no_drift"
    assert payload["release"]["id"] == release.id
    assert len(fake.apply_calls) == 1
    assert fake.apply_calls[0].allow_unevaluated is False


def test_apply_passes_through_release_and_override_flags(tmp_path, monkeypatch):
    from rag.control_plane.alias_service import AliasApplyResult

    catalog_path = _write_catalog(tmp_path / "catalog.toml")
    release = _release()
    deployment = AliasDeployment(
        id=uuid4(),
        kb_id="pytorch_reference",
        alias="champion",
        release_id=release.id,
        collection_name=release.collection_name,
        catalog_digest="sha256:a",
        build_config_digest="sha256:b",
        retrieval_config_digest="sha256:c",
        retrieval_config=AliasRetrievalConfig(strategy="dense", top_k=5, score_threshold=0.35),
        status="active",
    )
    fake = _FakeAliasService(
        apply_result=AliasApplyResult(
            deployment=deployment, release=release, action="reused_release"
        )
    )
    monkeypatch.setattr(alias_cli, "build_alias_service", lambda ctx: fake)

    result = runner.invoke(
        app,
        [
            "--catalog",
            str(catalog_path),
            "alias",
            "apply",
            "pytorch_reference",
            "champion",
            "--release",
            release.id,
            "--allow-unevaluated",
            "--allow-build-default",
        ],
    )

    assert result.exit_code == 0
    request = fake.apply_calls[0]
    assert request.release_id == release.id
    assert request.allow_unevaluated is True
    assert request.allow_build_default is True


def test_apply_refusal_maps_to_conflict_exit_code(tmp_path, monkeypatch):
    from rag.control_plane.alias_service import AliasApplyError

    catalog_path = _write_catalog(tmp_path / "catalog.toml")
    fake = _FakeAliasService(
        apply_error=AliasApplyError("multiple releases match the desired build and source state")
    )
    monkeypatch.setattr(alias_cli, "build_alias_service", lambda ctx: fake)

    result = runner.invoke(
        app, ["--catalog", str(catalog_path), "alias", "apply", "pytorch_reference", "champion"]
    )

    assert result.exit_code == 3


def test_status_aggregates_diff_across_all_declared_aliases(tmp_path, monkeypatch):
    catalog_path = _write_catalog(tmp_path / "catalog.toml")
    fake = _FakeAliasService(diff_result=_diff(build_drift=True))
    monkeypatch.setattr(alias_cli, "build_alias_service", lambda ctx, catalog_cfg=None: fake)

    result = runner.invoke(
        app, ["--catalog", str(catalog_path), "alias", "status", "pytorch_reference"]
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert len(payload) == 1
    assert payload[0]["alias"] == "champion"


def test_status_exits_two_for_unknown_kb(tmp_path):
    catalog_path = _write_catalog(tmp_path / "catalog.toml")

    result = runner.invoke(app, ["--catalog", str(catalog_path), "alias", "status", "nope"])

    assert result.exit_code == 2


def test_status_exits_four_when_an_alias_check_fails(tmp_path, monkeypatch):
    catalog_path = _write_catalog(tmp_path / "catalog.toml")
    fake = _FakeAliasService(diff_error=RuntimeError("database unavailable"))
    monkeypatch.setattr(alias_cli, "build_alias_service", lambda ctx, catalog_cfg=None: fake)

    result = runner.invoke(
        app, ["--catalog", str(catalog_path), "alias", "status", "pytorch_reference"]
    )

    assert result.exit_code == 4
    payload = json.loads(result.stdout)
    assert payload[0]["error"] == "database unavailable"


def test_nested_help_works_without_any_settings(tmp_path):
    result = runner.invoke(app, ["alias", "apply", "--help"])

    assert result.exit_code == 0
    assert "Make this KB alias match its catalog declaration" in result.stdout
