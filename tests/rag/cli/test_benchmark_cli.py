"""Tests for `rag benchmark run/list/show`, with an injected fake runner."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from typer.testing import CliRunner

import rag.cli.benchmark as benchmark_cli
from rag.cli.app import app
from rag.evaluation.runner import BenchmarkRunSummary

runner = CliRunner()


def _write_catalog(path: Path) -> Path:
    path.write_text(
        dedent(
            """
            schema_version = 4

            [[benchmark_adapters]]
            id = "benchmark.fake"
            version = "1"
            description = "Fake."
            factory = "tests.rag.test_benchmark_prep:_fake_qa_benchmark_adapter_factory"

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
            id = "pytorch_reference.qa_benchmark"
            description = "Benchmark."
            role = "benchmark"
            knowledge_base = "pytorch_reference"
            adapter = { id = "benchmark.fake", version = "1" }

            [source_instances.benchmark]
            suites = ["retrieval_quality"]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _summary(source_instance_id: str) -> BenchmarkRunSummary:
    return BenchmarkRunSummary(
        source_instance_id=source_instance_id,
        knowledge_base="pytorch_reference",
        alias="champion",
        collection_name="rag__pytorch_reference__abc123",
        case_count=1,
        metric_values={"document_hit_rate": 1.0},
    )


def test_run_requires_exactly_one_of_positional_or_kb(tmp_path):
    catalog_path = _write_catalog(tmp_path / "catalog.toml")

    result = runner.invoke(
        app, ["--catalog", str(catalog_path), "benchmark", "run", "--alias", "champion"]
    )

    assert result.exit_code == 2


def test_run_with_explicit_source_instance(tmp_path, monkeypatch):
    catalog_path = _write_catalog(tmp_path / "catalog.toml")
    monkeypatch.setattr(
        benchmark_cli,
        "_run_one",
        lambda *, ctx, source_instance_id, alias: _summary(source_instance_id),
    )

    result = runner.invoke(
        app,
        [
            "--catalog",
            str(catalog_path),
            "benchmark",
            "run",
            "pytorch_reference.qa_benchmark",
            "--alias",
            "champion",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["source_instance_id"] == "pytorch_reference.qa_benchmark"


def test_run_with_kb_runs_every_attached_benchmark(tmp_path, monkeypatch):
    catalog_path = _write_catalog(tmp_path / "catalog.toml")
    calls = []

    def _fake_run_one(*, ctx, source_instance_id, alias):
        calls.append(source_instance_id)
        return _summary(source_instance_id)

    monkeypatch.setattr(benchmark_cli, "_run_one", _fake_run_one)

    result = runner.invoke(
        app,
        [
            "--catalog",
            str(catalog_path),
            "benchmark",
            "run",
            "--kb",
            "pytorch_reference",
            "--alias",
            "champion",
        ],
    )

    assert result.exit_code == 0
    assert calls == ["pytorch_reference.qa_benchmark"]


def test_list_returns_persisted_evaluation_runs(tmp_path, monkeypatch):
    catalog_path = _write_catalog(tmp_path / "catalog.toml")
    monkeypatch.setattr(
        benchmark_cli,
        "list_evaluation_runs",
        lambda **kwargs: [
            {
                "id": "1aa97378-dca6-4f32-8148-426b8e17b78b",
                "knowledge_base": kwargs["knowledge_base"],
                "metric_name": "document_hit_rate",
            }
        ],
    )

    result = runner.invoke(
        app, ["--catalog", str(catalog_path), "benchmark", "list", "--kb", "pytorch_reference"]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload[0]["knowledge_base"] == "pytorch_reference"
    assert payload[0]["metric_name"] == "document_hit_rate"


def test_show_exits_two_when_run_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_cli, "get_evaluation_run", lambda **kwargs: None)
    result = runner.invoke(
        app,
        [
            "--data-root",
            str(tmp_path),
            "benchmark",
            "show",
            "1aa97378-dca6-4f32-8148-426b8e17b78b",
        ],
    )

    assert result.exit_code == 2


def test_show_returns_run_and_samples(tmp_path, monkeypatch):
    run_id = "1aa97378-dca6-4f32-8148-426b8e17b78b"
    monkeypatch.setattr(
        benchmark_cli,
        "get_evaluation_run",
        lambda **kwargs: {
            "run": {"id": kwargs["eval_run_id"], "metric_name": "document_hit_rate"},
            "samples": [{"sample_idx": 0, "sample_id": "case-1"}],
        },
    )

    result = runner.invoke(app, ["benchmark", "show", run_id])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["run"]["id"] == run_id
    assert payload["samples"][0]["sample_id"] == "case-1"
