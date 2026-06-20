"""Tests for `rag release list/show`, with an injected fake ReleaseRepository."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from typer.testing import CliRunner

import rag.cli.release as release_cli
from app_config.catalog.schema import AliasBuildConfig
from rag.cli.app import app
from rag.control_plane.models import RagRelease

runner = CliRunner()


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


class _FakeReleaseRepository:
    def __init__(self, releases: list[RagRelease]) -> None:
        self._releases = {release.id: release for release in releases}

    def list_for_kb(self, kb_id: str) -> list[RagRelease]:
        return [r for r in self._releases.values() if r.kb_id == kb_id]

    def get(self, release_id: str) -> RagRelease | None:
        return self._releases.get(release_id)


def test_list_returns_releases_for_kb(monkeypatch):
    release = _release()
    monkeypatch.setattr(
        release_cli, "build_release_repository", lambda ctx: _FakeReleaseRepository([release])
    )

    result = runner.invoke(app, ["release", "list", "--kb", "pytorch_reference"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert len(payload) == 1
    assert payload[0]["id"] == release.id


def test_show_returns_full_release(monkeypatch):
    release = _release()
    monkeypatch.setattr(
        release_cli, "build_release_repository", lambda ctx: _FakeReleaseRepository([release])
    )

    result = runner.invoke(app, ["release", "show", release.id])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["release_fingerprint"] == release.release_fingerprint


def test_show_exits_two_when_release_not_found(monkeypatch):
    monkeypatch.setattr(
        release_cli, "build_release_repository", lambda ctx: _FakeReleaseRepository([])
    )

    result = runner.invoke(app, ["release", "show", "ragrel_missing"])

    assert result.exit_code == 2
