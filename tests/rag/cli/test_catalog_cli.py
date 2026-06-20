"""Tests for `rag catalog validate`, using typer.testing.CliRunner."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from typer.testing import CliRunner

from rag.cli.app import app

runner = CliRunner()


def _write_catalog(path: Path, schema_version: int = 4) -> Path:
    path.write_text(
        dedent(
            f"""
            schema_version = {schema_version}

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


def test_validate_succeeds_for_a_well_formed_catalog(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.toml")

    result = runner.invoke(app, ["--catalog", str(catalog_path), "catalog", "validate"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["schema_version"] == 4
    assert payload["knowledge_base_count"] == 1


def test_validate_fails_for_schema_version_3(tmp_path: Path) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.toml", schema_version=3)

    result = runner.invoke(app, ["--catalog", str(catalog_path), "catalog", "validate"])

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["valid"] is False
    assert "schema_version 3" in payload["error"]


def test_validate_fails_for_missing_catalog_file(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.toml"

    result = runner.invoke(app, ["--catalog", str(missing), "catalog", "validate"])

    assert result.exit_code == 2


def test_help_does_not_require_a_catalog_path() -> None:
    result = runner.invoke(app, ["catalog", "validate", "--help"])

    assert result.exit_code == 0
    assert "Validate the catalog file" in result.stdout


def test_logs_go_to_stderr_not_stdout(tmp_path: Path, monkeypatch) -> None:
    """stdout stays pure, parseable JSON even when the command path logs."""
    import logging

    import rag.cli.catalog as catalog_cli

    catalog_path = _write_catalog(tmp_path / "catalog.toml")
    real_load = catalog_cli.load_catalog_config

    def _logging_load(ctx):
        logging.getLogger("rag.cli.test_probe").info(
            "a log emitted during command execution must never land in stdout"
        )
        return real_load(ctx)

    monkeypatch.setattr(catalog_cli, "load_catalog_config", _logging_load)

    result = runner.invoke(app, ["--catalog", str(catalog_path), "catalog", "validate"])

    assert result.exit_code == 0
    # The whole of stdout must parse as exactly one JSON document.
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert "must never land in stdout" not in result.stdout
    assert "must never land in stdout" in result.stderr
