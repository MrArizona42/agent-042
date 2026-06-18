"""Tests for the schema-version-3 source-instance catalog model.

Covers `[[source_adapters]]` / `[[benchmark_adapters]]` / `[[source_instances]]`
schema validation and the SourceInstanceIndex used to query source instances
by role and KB.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

V3_BASE = """
schema_version = 3

[[knowledge_bases]]
id = "pytorch_reference"
description = "PyTorch API reference."
default_alias = "champion"

[knowledge_bases.aliases.champion]
top_k = 5
score_threshold = 0.35
retrieval_strategy = "dense"
reranker_multiplier = 1

[[source_adapters]]
id = "generic.http_html"
version = "1"
description = "Fetches HTTP HTML pages."
factory = "rag.ingest.adapters:make_http_html_adapter"

[[benchmark_adapters]]
id = "benchmark.pytorch_qa"
version = "1"
description = "Loads QA examples for PyTorch docs."
factory = "tests.rag.test_adapter_loading:_benchmark_adapter"

[[source_instances]]
id = "pytorch_reference.docs"
description = "Official PyTorch documentation pages."
role = "corpus"
knowledge_base = "pytorch_reference"
adapter = { id = "generic.http_html", version = "1" }

[[source_instances]]
id = "pytorch_reference.qa_benchmark"
description = "QA benchmark cases for PyTorch documentation."
role = "benchmark"
knowledge_base = "pytorch_reference"
adapter = { id = "benchmark.pytorch_qa", version = "1" }

[source_instances.benchmark]
suites = ["context_quality", "generation_quality"]
"""


def _write(path: Path, content: str) -> Path:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return path


class TestSchemaVersion3SourceInstances:
    def test_current_catalog_toml_still_loads(self):
        from app_config.catalog import load_catalog

        load_catalog(Path("catalog.toml"))

    def test_v3_sample_loads(self, tmp_path: Path):
        from app_config.catalog import load_catalog

        path = _write(tmp_path / "catalog.toml", V3_BASE)
        load_catalog(path)

    def test_corpus_and_benchmark_instances_are_partitioned_by_role(self, tmp_path: Path):
        from app_config.catalog import load_source_instance_index

        path = _write(tmp_path / "catalog.toml", V3_BASE)
        index = load_source_instance_index(path)

        corpus = index.corpus_for_kb("pytorch_reference")
        benchmark = index.benchmark_for_kb("pytorch_reference")
        assert [s.id for s in corpus] == ["pytorch_reference.docs"]
        assert [s.id for s in benchmark] == ["pytorch_reference.qa_benchmark"]

    def test_manifest_path_is_derived_from_source_instance_id(self):
        from app_config.catalog import conventional_manifest_path

        path = conventional_manifest_path("assets/rag_data", "pytorch_reference.docs")
        expected = Path("assets/rag_data/source_instances/pytorch_reference.docs/manifest.toml")
        assert path == expected

    def test_duplicate_source_instance_id_is_rejected(self, tmp_path: Path):
        from app_config.catalog import load_catalog

        content = V3_BASE + dedent(
            """

            [[source_instances]]
            id = "pytorch_reference.docs"
            description = "Duplicate id."
            role = "corpus"
            knowledge_base = "pytorch_reference"
            adapter = { id = "generic.http_html", version = "1" }
            """
        )
        path = _write(tmp_path / "catalog.toml", content)

        with pytest.raises(ValueError, match="Duplicate source instance id"):
            load_catalog(path)

    def test_source_instance_unknown_kb_is_rejected(self, tmp_path: Path):
        from app_config.catalog import load_catalog

        content = V3_BASE.replace(
            'knowledge_base = "pytorch_reference"\nadapter = { id = "generic.http_html"',
            'knowledge_base = "missing_kb"\nadapter = { id = "generic.http_html"',
        )
        path = _write(tmp_path / "catalog.toml", content)

        with pytest.raises(ValueError, match="unknown KB 'missing_kb'"):
            load_catalog(path)

    def test_source_instance_undeclared_adapter_is_rejected(self, tmp_path: Path):
        from app_config.catalog import load_catalog

        content = V3_BASE.replace(
            'adapter = { id = "generic.http_html", version = "1" }\n\n[[source_instances]]',
            'adapter = { id = "generic.missing_adapter", version = "1" }\n\n[[source_instances]]',
        )
        path = _write(tmp_path / "catalog.toml", content)

        with pytest.raises(ValueError, match="undeclared adapter"):
            load_catalog(path)

    def test_benchmark_role_requires_benchmark_capable_adapter(self, tmp_path: Path):
        from pydantic import ValidationError

        from app_config.catalog import load_catalog

        content = V3_BASE.replace(
            'adapter = { id = "benchmark.pytorch_qa", version = "1" }',
            'adapter = { id = "generic.http_html", version = "1" }',
        )
        path = _write(tmp_path / "catalog.toml", content)

        with pytest.raises((ValueError, ValidationError), match="benchmark-capable"):
            load_catalog(path)

    def test_benchmark_role_without_benchmark_block_is_rejected(self):
        from pydantic import ValidationError

        from app_config.catalog import SourceInstanceAdapterRef, SourceInstanceConfig

        with pytest.raises(ValidationError, match="benchmark"):
            SourceInstanceConfig(
                id="kb.benchmark",
                description="Missing benchmark block.",
                role="benchmark",
                knowledge_base="kb",
                adapter=SourceInstanceAdapterRef(id="benchmark.x", version="1"),
            )

    def test_corpus_role_with_benchmark_block_is_rejected(self):
        from pydantic import ValidationError

        from app_config.catalog import (
            BenchmarkSourceConfig,
            SourceInstanceAdapterRef,
            SourceInstanceConfig,
        )

        with pytest.raises(ValidationError, match="must not have a benchmark block"):
            SourceInstanceConfig(
                id="kb.corpus",
                description="Corpus with a stray benchmark block.",
                role="corpus",
                knowledge_base="kb",
                adapter=SourceInstanceAdapterRef(id="generic.x", version="1"),
                benchmark=BenchmarkSourceConfig(
                    suites=["retrieval_quality"],
                ),
            )

    def test_benchmark_suites_rejects_unknown_vocabulary(self):
        from pydantic import ValidationError

        from app_config.catalog import BenchmarkSourceConfig

        with pytest.raises(ValidationError, match="Input should be"):
            BenchmarkSourceConfig(
                suites=["retrieval_quality", "not_a_real_suite"],
            )

    def test_benchmark_suites_rejects_duplicates(self):
        from pydantic import ValidationError

        from app_config.catalog import BenchmarkSourceConfig

        with pytest.raises(ValidationError, match="must not contain duplicate"):
            BenchmarkSourceConfig(
                suites=["retrieval_quality", "retrieval_quality"],
            )
