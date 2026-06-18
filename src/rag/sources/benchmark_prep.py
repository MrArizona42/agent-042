"""Benchmark preparation: validate a benchmark manifest, emit normalized cases/labels."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from app_config.catalog import (
    CatalogConfig,
    build_source_instance_index,
    conventional_manifest_path,
    materialize_catalog,
)
from rag.evaluation.models import BenchmarkCase, BenchmarkLabel, BenchmarkPreparedArtifacts
from rag.ingest import SourceAdapterRegistry, build_catalog_adapter_registry
from rag.sources.manifests import load_source_manifest


class BenchmarkPrepSummary(BaseModel):
    """Summary for one benchmark source instance preparation run."""

    model_config = ConfigDict(extra="forbid")

    source_instance_id: str
    knowledge_base: str
    adapter_id: str
    adapter_version: str
    case_count: int
    label_count: int


def _load_catalog_config(catalog_path: Path | str) -> CatalogConfig:
    path = Path(catalog_path)
    if path.suffix.lower() != ".toml":
        raise ValueError(f"Catalog must be a TOML file (got '{path.name}')")
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    catalog = CatalogConfig(**raw)
    materialize_catalog(catalog)
    return catalog


def benchmark_artifact_dir(rag_data_root: Path | str, source_instance_id: str) -> Path:
    """Return the conventional benchmark artifact directory for a source instance."""
    return Path(rag_data_root) / "source_instances" / source_instance_id / "benchmark"


def cases_artifact_path(rag_data_root: Path | str, source_instance_id: str) -> Path:
    """Return the conventional path for a benchmark source instance's cases artifact."""
    return benchmark_artifact_dir(rag_data_root, source_instance_id) / "cases.jsonl"


def labels_artifact_path(rag_data_root: Path | str, source_instance_id: str) -> Path:
    """Return the conventional path for a benchmark source instance's labels artifact."""
    return benchmark_artifact_dir(rag_data_root, source_instance_id) / "labels.jsonl"


def metadata_artifact_path(rag_data_root: Path | str, source_instance_id: str) -> Path:
    """Return the conventional path for a benchmark source instance's metadata artifact."""
    return benchmark_artifact_dir(rag_data_root, source_instance_id) / "metadata.json"


def _write_jsonl(path: Path, models: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps(model.model_dump(mode="json"), sort_keys=True) for model in models)
    path.write_text("\n".join(lines) + ("\n" if models else ""), encoding="utf-8")


def _read_jsonl(path: Path, model_cls: type[BaseModel]) -> list[Any]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [model_cls.model_validate_json(line) for line in text.splitlines()]


_PRODUCED_CONTAINS_CHECKS = {
    "qrels": lambda label: bool(label.qrels),
    "answers": lambda label: bool(label.reference_answers or label.reference_answer_ids),
    "scores": lambda label: bool(label.scores),
    "evidence_refs": lambda label: bool(label.evidence_refs),
    "rubrics": lambda label: bool(label.rubrics),
}


def _validate_contains(
    artifacts: BenchmarkPreparedArtifacts,
    *,
    contains: list[str],
    source_instance_id: str,
) -> None:
    """Raise if produced labels carry fields not declared in `benchmark.contains`."""
    declared = set(contains)
    undeclared = sorted(
        name
        for name, produced in _PRODUCED_CONTAINS_CHECKS.items()
        if name not in declared and any(produced(label) for label in artifacts.labels)
    )
    if undeclared:
        raise ValueError(
            f"Benchmark source instance '{source_instance_id}' produced label fields "
            f"{undeclared} not declared in benchmark.contains {sorted(declared)}"
        )


def prepare_benchmark_source_instance(
    *,
    catalog_path: Path | str,
    source_instance_id: str,
    rag_data_root: Path | str,
    adapter_registry: SourceAdapterRegistry | None = None,
) -> BenchmarkPrepSummary:
    """Validate a benchmark manifest and write its normalized cases/labels artifacts.

    Rejects targets that are not declared `role = "benchmark"` source
    instances. Benchmark preparation never touches corpus chunk artifacts,
    so labels can be revised without forcing a collection rebuild.
    """
    catalog_path = Path(catalog_path)
    catalog = _load_catalog_config(catalog_path)
    index = build_source_instance_index(catalog)
    instance = index.get(source_instance_id)

    if instance.role != "benchmark":
        raise ValueError(
            f"Source instance '{source_instance_id}' has role '{instance.role}'; "
            "prepare-benchmark only runs against role 'benchmark' instances."
        )
    if instance.benchmark is None:
        raise ValueError(f"Source instance '{source_instance_id}' has no benchmark block")

    registry = adapter_registry or build_catalog_adapter_registry(catalog)
    adapter = registry.get(instance.adapter.id, version=instance.adapter.version)
    if "benchmark" not in adapter.capabilities:
        raise ValueError(
            f"Adapter '{instance.adapter.id}@{instance.adapter.version}' for source instance "
            f"'{source_instance_id}' is not benchmark-capable"
        )

    manifest_path = conventional_manifest_path(rag_data_root, source_instance_id)
    manifest = adapter.validate_manifest(load_source_manifest(manifest_path))
    artifacts = adapter.prepare_benchmark(manifest)

    _validate_contains(
        artifacts,
        contains=instance.benchmark.contains,
        source_instance_id=source_instance_id,
    )

    _write_jsonl(cases_artifact_path(rag_data_root, source_instance_id), artifacts.cases)
    _write_jsonl(labels_artifact_path(rag_data_root, source_instance_id), artifacts.labels)

    summary = BenchmarkPrepSummary(
        source_instance_id=source_instance_id,
        knowledge_base=instance.knowledge_base,
        adapter_id=instance.adapter.id,
        adapter_version=instance.adapter.version,
        case_count=len(artifacts.cases),
        label_count=len(artifacts.labels),
    )
    metadata_path = metadata_artifact_path(rag_data_root, source_instance_id)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def read_prepared_benchmark_artifacts(
    rag_data_root: Path | str,
    source_instance_id: str,
) -> BenchmarkPreparedArtifacts:
    """Read back a previously prepared benchmark source instance's cases and labels."""
    cases = _read_jsonl(cases_artifact_path(rag_data_root, source_instance_id), BenchmarkCase)
    labels = _read_jsonl(labels_artifact_path(rag_data_root, source_instance_id), BenchmarkLabel)
    return BenchmarkPreparedArtifacts(cases=cases, labels=labels)
