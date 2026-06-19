"""Tests for benchmark preparation (Phase 5)."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from llama_index.core.schema import Document

from rag.evaluation.models import BenchmarkCase, BenchmarkLabel, BenchmarkPreparedArtifacts, Qrel
from rag.sources.benchmark_prep import (
    prepare_benchmark_source_instance,
    read_prepared_benchmark_artifacts,
)


class _FakeQABenchmarkAdapter:
    adapter_id = "benchmark.fake_qa"
    version = "1"
    capabilities = frozenset({"source", "benchmark"})

    def validate_manifest(self, manifest):
        return manifest

    def list_documents(self, manifest, *, context):
        del manifest, context
        return []

    def fetcher(self):
        raise NotImplementedError

    def extractor(self):
        raise NotImplementedError

    def prepare_benchmark(self, manifest) -> BenchmarkPreparedArtifacts:
        cases = []
        labels = []
        for document in manifest.documents:
            case_id = f"case:{document.id}"
            cases.append(
                BenchmarkCase(
                    id=case_id,
                    benchmark_source_instance_id="pytorch_reference.qa_benchmark",
                    query=document.title,
                    metadata={"source_doc_id": document.id},
                )
            )
            labels.append(
                BenchmarkLabel(
                    case_id=case_id,
                    reference_answers=[document.metadata.get("answer", "")],
                )
            )
        return BenchmarkPreparedArtifacts(cases=cases, labels=labels)


class _FakeQABenchmarkAdapterWithUndeclaredQrels(_FakeQABenchmarkAdapter):
    def prepare_benchmark(self, manifest) -> BenchmarkPreparedArtifacts:
        artifacts = super().prepare_benchmark(manifest)
        artifacts.labels[0].qrels.append(Qrel(entity_id="doc:q1", relevance_grade=1))
        return artifacts


class _FakeQABenchmarkAdapterWithCorpus(_FakeQABenchmarkAdapter):
    def prepare_benchmark(self, manifest) -> BenchmarkPreparedArtifacts:
        artifacts = super().prepare_benchmark(manifest)
        artifacts.documents.append(
            Document(
                id_="doc:q1",
                text="A tensor is a multi-dimensional array.",
                metadata={"document_id": "doc:q1"},
            )
        )
        return artifacts


def _fake_qa_benchmark_adapter_factory() -> _FakeQABenchmarkAdapter:
    return _FakeQABenchmarkAdapter()


def _fake_qa_benchmark_adapter_with_qrels_factory() -> _FakeQABenchmarkAdapterWithUndeclaredQrels:
    return _FakeQABenchmarkAdapterWithUndeclaredQrels()


def _fake_qa_benchmark_adapter_with_corpus_factory() -> _FakeQABenchmarkAdapterWithCorpus:
    return _FakeQABenchmarkAdapterWithCorpus()


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return path


def _catalog(tmp_path: Path, *, factory: str) -> Path:
    return _write(
        tmp_path / "catalog.toml",
        f"""
        schema_version = 3

        [[benchmark_adapters]]
        id = "benchmark.fake_qa"
        version = "1"
        description = "Fake QA benchmark adapter."
        factory = "{factory}"

        [[knowledge_bases]]
        id = "pytorch_reference"
        description = "PyTorch API reference."
        default_alias = "champion"

        [knowledge_bases.aliases.champion]
        top_k = 5
        score_threshold = 0.35
        retrieval_strategy = "dense"
        reranker_multiplier = 1

        [[source_instances]]
        id = "pytorch_reference.qa_benchmark"
        description = "QA benchmark cases for PyTorch documentation."
        role = "benchmark"
        knowledge_base = "pytorch_reference"
        adapter = {{ id = "benchmark.fake_qa", version = "1" }}

        [source_instances.benchmark]
        suites = ["generation_quality"]
        """,
    )


def _manifest(tmp_path: Path, source_instance_id: str) -> Path:
    return _write(
        tmp_path / "source_instances" / source_instance_id / "manifest.toml",
        """
        schema_version = 1
        [[documents]]
        id = "q1"
        title = "What is a tensor?"
        metadata = { answer = "A tensor is a multi-dimensional array." }
        """,
    )


def test_prepare_benchmark_writes_and_reads_back_cases_and_labels(tmp_path: Path) -> None:
    _manifest(tmp_path, "pytorch_reference.qa_benchmark")
    catalog_path = _catalog(
        tmp_path, factory="tests.rag.test_benchmark_prep:_fake_qa_benchmark_adapter_factory"
    )

    summary = prepare_benchmark_source_instance(
        catalog_path=catalog_path,
        source_instance_id="pytorch_reference.qa_benchmark",
        rag_data_root=tmp_path,
    )

    assert summary.source_instance_id == "pytorch_reference.qa_benchmark"
    assert summary.knowledge_base == "pytorch_reference"
    assert summary.case_count == 1
    assert summary.label_count == 1

    artifacts = read_prepared_benchmark_artifacts(tmp_path, "pytorch_reference.qa_benchmark")
    assert len(artifacts.cases) == 1
    assert artifacts.cases[0].query == "What is a tensor?"
    assert artifacts.labels[0].reference_answers == ["A tensor is a multi-dimensional array."]


def test_prepare_benchmark_rejects_non_benchmark_role_target(tmp_path: Path) -> None:
    _manifest(tmp_path, "pytorch_reference.qa_benchmark")
    content = _catalog(
        tmp_path, factory="tests.rag.test_benchmark_prep:_fake_qa_benchmark_adapter_factory"
    ).read_text(encoding="utf-8")
    content = content.replace('role = "benchmark"', 'role = "corpus"').replace(
        '\n        [source_instances.benchmark]\n        suites = ["generation_quality"]\n',
        "\n",
    )
    catalog_path = _write(tmp_path / "catalog.toml", content)

    with pytest.raises(ValueError, match="has role 'corpus'"):
        prepare_benchmark_source_instance(
            catalog_path=catalog_path,
            source_instance_id="pytorch_reference.qa_benchmark",
            rag_data_root=tmp_path,
        )


def test_prepare_benchmark_allows_adapter_to_emit_any_normalized_label_channels(
    tmp_path: Path,
) -> None:
    _manifest(tmp_path, "pytorch_reference.qa_benchmark")
    catalog_path = _catalog(
        tmp_path,
        factory="tests.rag.test_benchmark_prep:_fake_qa_benchmark_adapter_with_qrels_factory",
    )

    summary = prepare_benchmark_source_instance(
        catalog_path=catalog_path,
        source_instance_id="pytorch_reference.qa_benchmark",
        rag_data_root=tmp_path,
    )

    assert summary.label_count == 1


def test_read_prepared_benchmark_artifacts_returns_empty_when_not_prepared(
    tmp_path: Path,
) -> None:
    artifacts = read_prepared_benchmark_artifacts(tmp_path, "pytorch_reference.qa_benchmark")

    assert artifacts.cases == []
    assert artifacts.labels == []


def test_prepare_benchmark_round_trips_normalized_llamaindex_corpus(tmp_path: Path) -> None:
    _manifest(tmp_path, "pytorch_reference.qa_benchmark")
    catalog_path = _catalog(
        tmp_path,
        factory="tests.rag.test_benchmark_prep:_fake_qa_benchmark_adapter_with_corpus_factory",
    )

    summary = prepare_benchmark_source_instance(
        catalog_path=catalog_path,
        source_instance_id="pytorch_reference.qa_benchmark",
        rag_data_root=tmp_path,
    )
    artifacts = read_prepared_benchmark_artifacts(
        tmp_path,
        "pytorch_reference.qa_benchmark",
    )

    assert summary.document_count == 1
    assert set(summary.artifact_digests) == {"corpus", "cases", "labels"}
    assert artifacts.documents[0].id_ == "doc:q1"
    assert artifacts.documents[0].metadata["document_id"] == "doc:q1"
