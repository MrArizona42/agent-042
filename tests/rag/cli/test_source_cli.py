"""Tests for `rag source inspect/rebuild`, expert-diagnostics commands."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from typer.testing import CliRunner

import rag.cli.source as source_cli
from rag.cli.app import app

runner = CliRunner()


def _write_catalog(path: Path) -> Path:
    path.write_text(
        dedent(
            """
            schema_version = 4

            [[source_adapters]]
            id = "generic.http_html"
            version = "1"
            description = "Fetches HTML pages."
            factory = "rag.adapters.sources:make_http_html_adapter"

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

            [[source_instances]]
            id = "pytorch_reference.docs"
            description = "Official docs."
            role = "corpus"
            knowledge_base = "pytorch_reference"
            adapter = { id = "generic.http_html", version = "1" }
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_inspect_reports_no_cache_for_a_fresh_source_instance(tmp_path):
    catalog_path = _write_catalog(tmp_path / "catalog.toml")
    data_root = tmp_path / "data"

    result = runner.invoke(
        app,
        [
            "--catalog",
            str(catalog_path),
            "--data-root",
            str(data_root),
            "source",
            "inspect",
            "pytorch_reference.docs",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["manifest_exists"] is False
    assert payload["raw_count"] == 0
    assert payload["chunk_digest_dirs"] == []


def test_inspect_exits_two_for_unknown_source_instance(tmp_path):
    catalog_path = _write_catalog(tmp_path / "catalog.toml")

    result = runner.invoke(
        app, ["--catalog", str(catalog_path), "source", "inspect", "unknown.instance"]
    )

    assert result.exit_code == 2


def test_rebuild_invokes_build_source_instance_by_global_id(tmp_path, monkeypatch):
    catalog_path = _write_catalog(tmp_path / "catalog.toml")
    calls = []

    def _fake_build(**kwargs):
        calls.append(kwargs)
        from rag.sources.build import GlobalSourceBuildSummary, SourceBuildSummary
        from rag.sources.bundles import SourceNodeBundle  # noqa: F401
        from rag.sources.chunks import SourceInstanceChunkingSummary
        from rag.sources.processing import SourceProcessingSummary

        return GlobalSourceBuildSummary(
            catalog_path=str(catalog_path),
            source_instance_id="pytorch_reference.docs",
            role="corpus",
            build=SourceBuildSummary(
                kb_id="pytorch_reference",
                source_instance_id="pytorch_reference.docs",
                adapter_id="generic.http_html",
                status="empty",
                processing=SourceProcessingSummary(
                    kb_id="pytorch_reference",
                    source_instance_id="pytorch_reference.docs",
                    adapter_id="generic.http_html",
                    total_selected=0,
                ),
                chunking=SourceInstanceChunkingSummary(
                    kb_id="pytorch_reference",
                    source_instance_id="pytorch_reference.docs",
                    total_selected=0,
                ),
            ),
        )

    monkeypatch.setattr(source_cli, "build_source_instance_by_global_id", _fake_build)

    result = runner.invoke(
        app,
        [
            "--catalog",
            str(catalog_path),
            "source",
            "rebuild",
            "pytorch_reference.docs",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["force_fetch"] is True
    assert calls[0]["force_extract"] is True
    assert calls[0]["force_chunk"] is True
