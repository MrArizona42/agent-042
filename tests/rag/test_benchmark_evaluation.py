"""Phase 5 benchmark evaluator and runner tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from rag.evaluation import runner
from rag.evaluation.models import BenchmarkCase, BenchmarkLabel, Qrel
from rag.evaluation.retrieval import ProjectRetrieverEvaluator, graded_ndcg
from rag.evaluation.runner import run_benchmark
from rag.sources.benchmark_prep import (
    cases_artifact_path,
    labels_artifact_path,
    metadata_artifact_path,
)


class _Retriever(BaseRetriever):
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        del query_bundle
        return [
            NodeWithScore(
                node=TextNode(
                    id_="chunk-1",
                    text="relevant",
                    metadata={"chunk_id": "chunk-1", "document_id": "doc-1"},
                ),
                score=0.9,
            ),
            NodeWithScore(
                node=TextNode(
                    id_="chunk-2",
                    text="less relevant",
                    metadata={"chunk_id": "chunk-2", "document_id": "doc-2"},
                ),
                score=0.4,
            ),
        ]


def test_project_retriever_evaluator_keeps_nodes_and_supports_graded_qrels() -> None:
    evaluator = ProjectRetrieverEvaluator.from_metric_names(
        ["hit_rate", "recall", "ndcg"],
        retriever=_Retriever(),
    )
    result = evaluator.evaluate(
        query="question",
        expected_ids=["chunk-1"],
    )
    assert result.metric_dict["hit_rate"].score == 1.0

    import asyncio

    project = asyncio.run(
        evaluator.aevaluate_project(
            query="question",
            qrels=[
                Qrel(entity_type="document", entity_id="doc-1", relevance_grade=3),
                Qrel(entity_type="document", entity_id="doc-2", relevance_grade=1),
            ],
            entity_type="document",
        )
    )
    assert project.retrieved_ids == ["doc-1", "doc-2"]
    assert project.retrieved_nodes[0].score == 0.9
    assert project.metric_scores["graded_ndcg@2"] == pytest.approx(1.0)


def test_graded_ndcg_respects_relevance_grades() -> None:
    qrels = [
        Qrel(entity_id="best", relevance_grade=3),
        Qrel(entity_id="okay", relevance_grade=1),
    ]
    assert graded_ndcg(qrels, ["best", "okay"], k=2) == pytest.approx(1.0)
    assert graded_ndcg(qrels, ["okay", "best"], k=2) < 1.0


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_run_benchmark_requires_alias_and_persists_metric_and_artifact_identity(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.toml"
    _write(
        catalog,
        """
schema_version = 3
[[benchmark_adapters]]
id = "benchmark.fake"
description = "Fake benchmark."
factory = "tests.rag.test_benchmark_prep:_fake_qa_benchmark_adapter_factory"
[[knowledge_bases]]
id = "kb"
description = "KB."
default_alias = "champion"
[knowledge_bases.aliases.champion]
top_k = 2
score_threshold = 0.1
retrieval_strategy = "dense"
reranker_multiplier = 1
[[source_instances]]
id = "kb.benchmark"
description = "Benchmark."
role = "benchmark"
knowledge_base = "kb"
adapter = { id = "benchmark.fake" }
[source_instances.benchmark]
suites = ["retrieval_quality"]
""".strip()
        + "\n",
    )
    case = BenchmarkCase(
        id="case-1",
        benchmark_source_instance_id="kb.benchmark",
        query="question",
    )
    label = BenchmarkLabel(
        case_id="case-1",
        qrels=[Qrel(entity_id="doc-1", relevance_grade=2)],
    )
    _write(cases_artifact_path(tmp_path, "kb.benchmark"), case.model_dump_json() + "\n")
    _write(labels_artifact_path(tmp_path, "kb.benchmark"), label.model_dump_json() + "\n")
    _write(
        metadata_artifact_path(tmp_path, "kb.benchmark"),
        json.dumps({"artifact_digests": {"cases": "sha256:cases"}}),
    )

    class FakeTarget:
        source_instance_id = "kb.benchmark"
        kb = SimpleNamespace(name="kb")
        alias = "champion"
        alias_config = SimpleNamespace(
            top_k=2,
            score_threshold=0.1,
            reranker=None,
            retrieval_strategy="dense",
        )
        state = SimpleNamespace(
            qdrant_alias="rag__kb__champion",
            collection_name="eval__kb__one",
            attestation=SimpleNamespace(
                manifest_id="sha256:manifest",
                embedding_model="embedding",
                retrieval_capability=SimpleNamespace(value="dense"),
            ),
        )
        parameter_state = SimpleNamespace(
            collection_name="rag__kb__production",
            attestation=SimpleNamespace(manifest_id="sha256:production"),
        )
        build_profile = SimpleNamespace(
            chunking_config={"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 64}
        )
        retriever = _Retriever()
        node_postprocessors = []
        closed = False

        def close(self):
            self.closed = True

    target = FakeTarget()
    writes = []

    def target_factory(**kwargs):
        del kwargs
        return target

    def writer(rows, *, db_url, sample_rows):
        writes.append((rows, db_url, sample_rows))

    summary = run_benchmark(
        catalog_path=catalog,
        source_instance_id="kb.benchmark",
        alias="champion",
        rag_data_root=tmp_path,
        db_url="postgresql://db",
        runtime=object(),
        base_model="model",
        target_factory=target_factory,
        writer=writer,
    )

    assert summary.collection_name == "eval__kb__one"
    assert summary.metric_values["document_hit_rate"] == 1.0
    assert writes[0][0][0]["extra"]["benchmark_artifact_digests"] == {"cases": "sha256:cases"}
    assert writes[0][2][0]["detail"]["retrieved_scores"] == [0.9, 0.4]
    assert target.closed is True

    with pytest.raises(ValueError, match="explicit alias"):
        run_benchmark(
            catalog_path=catalog,
            source_instance_id="kb.benchmark",
            alias="",
            rag_data_root=tmp_path,
            db_url="postgresql://db",
            runtime=object(),
            base_model="model",
            target_factory=target_factory,
            writer=writer,
        )


def test_run_benchmark_retrieval_only_never_constructs_judges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for the BenchmarkJudges eager-construction bug.

    BenchmarkJudges eagerly builds LlamaIndex's Faithfulness/Relevancy/
    Correctness evaluators, which read ``judge_llm.metadata`` at construction
    time. For a self-hosted model name that crashes immediately, regardless
    of whether the run actually needs a judge. A retrieval_quality-only run
    must never construct or touch the judge LLM at all, even though the CLI
    always supplies one (generation and judge clients are built unconditionally
    by ``RagRuntime``, independent of the benchmark's declared suites).
    """
    catalog = tmp_path / "catalog.toml"
    _write(
        catalog,
        """
schema_version = 3
[[benchmark_adapters]]
id = "benchmark.fake"
description = "Fake benchmark."
factory = "tests.rag.test_benchmark_prep:_fake_qa_benchmark_adapter_factory"
[[knowledge_bases]]
id = "kb"
description = "KB."
default_alias = "champion"
[knowledge_bases.aliases.champion]
top_k = 2
score_threshold = 0.1
retrieval_strategy = "dense"
reranker_multiplier = 1
[[source_instances]]
id = "kb.benchmark"
description = "Benchmark."
role = "benchmark"
knowledge_base = "kb"
adapter = { id = "benchmark.fake" }
[source_instances.benchmark]
suites = ["retrieval_quality"]
""".strip()
        + "\n",
    )
    case = BenchmarkCase(
        id="case-1",
        benchmark_source_instance_id="kb.benchmark",
        query="question",
    )
    label = BenchmarkLabel(
        case_id="case-1",
        qrels=[Qrel(entity_id="doc-1", relevance_grade=2)],
    )
    _write(cases_artifact_path(tmp_path, "kb.benchmark"), case.model_dump_json() + "\n")
    _write(labels_artifact_path(tmp_path, "kb.benchmark"), label.model_dump_json() + "\n")
    _write(metadata_artifact_path(tmp_path, "kb.benchmark"), json.dumps({"artifact_digests": {}}))

    class FakeTarget:
        source_instance_id = "kb.benchmark"
        kb = SimpleNamespace(name="kb")
        alias = "champion"
        alias_config = SimpleNamespace(
            top_k=2,
            score_threshold=0.1,
            reranker=None,
            retrieval_strategy="dense",
        )
        state = SimpleNamespace(
            qdrant_alias="rag__kb__champion",
            collection_name="eval__kb__one",
            attestation=SimpleNamespace(
                manifest_id="sha256:manifest",
                embedding_model="embedding",
                retrieval_capability=SimpleNamespace(value="dense"),
            ),
        )
        parameter_state = SimpleNamespace(
            collection_name="rag__kb__production",
            attestation=SimpleNamespace(manifest_id="sha256:production"),
        )
        build_profile = SimpleNamespace(
            chunking_config={"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 64}
        )
        retriever = _Retriever()
        node_postprocessors = []

        def close(self):
            pass

    target = FakeTarget()

    def target_factory(**kwargs):
        del kwargs
        return target

    def writer(rows, *, db_url, sample_rows):
        del rows, db_url, sample_rows

    def _explode(*args, **kwargs):
        raise AssertionError(
            "BenchmarkJudges must not be constructed for a retrieval_quality-only suite"
        )

    monkeypatch.setattr(runner, "BenchmarkJudges", _explode)

    class _PoisonedJudgeLLM:
        """Stands in for a real LLM client; touching it at all is the bug."""

        @property
        def metadata(self):
            raise AssertionError("judge LLM must not be touched for a retrieval_quality-only run")

    summary = run_benchmark(
        catalog_path=catalog,
        source_instance_id="kb.benchmark",
        alias="champion",
        rag_data_root=tmp_path,
        db_url="postgresql://db",
        runtime=object(),
        base_model="model",
        judge_llm=_PoisonedJudgeLLM(),
        target_factory=target_factory,
        writer=writer,
    )

    assert summary.metric_values["document_hit_rate"] == 1.0


def test_run_benchmark_persists_resolved_judge_identity_for_context_quality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: persisted judge_backend/judge_model must reflect the
    actually configured judge, not the generation model or a hardcoded string.

    Before this fix, the CLI always passed ``judge_model=settings.vllm.model``
    (the generation model) and the runner persisted a hardcoded
    ``judge_backend="llamaindex"``, so the DB could never distinguish a real
    judge backend/model from the generation model.
    """
    catalog = tmp_path / "catalog.toml"
    _write(
        catalog,
        """
schema_version = 3
[[benchmark_adapters]]
id = "benchmark.fake"
description = "Fake benchmark."
factory = "tests.rag.test_benchmark_prep:_fake_qa_benchmark_adapter_factory"
[[knowledge_bases]]
id = "kb"
description = "KB."
default_alias = "champion"
[knowledge_bases.aliases.champion]
top_k = 2
score_threshold = 0.1
retrieval_strategy = "dense"
reranker_multiplier = 1
[[source_instances]]
id = "kb.benchmark"
description = "Benchmark."
role = "benchmark"
knowledge_base = "kb"
adapter = { id = "benchmark.fake" }
[source_instances.benchmark]
suites = ["context_quality"]
""".strip()
        + "\n",
    )
    case = BenchmarkCase(
        id="case-1",
        benchmark_source_instance_id="kb.benchmark",
        query="question",
    )
    label = BenchmarkLabel(case_id="case-1")
    _write(cases_artifact_path(tmp_path, "kb.benchmark"), case.model_dump_json() + "\n")
    _write(labels_artifact_path(tmp_path, "kb.benchmark"), label.model_dump_json() + "\n")
    _write(metadata_artifact_path(tmp_path, "kb.benchmark"), json.dumps({"artifact_digests": {}}))

    class FakeTarget:
        source_instance_id = "kb.benchmark"
        kb = SimpleNamespace(name="kb")
        alias = "champion"
        alias_config = SimpleNamespace(
            top_k=2,
            score_threshold=0.1,
            reranker=None,
            retrieval_strategy="dense",
        )
        state = SimpleNamespace(
            qdrant_alias="rag__kb__champion",
            collection_name="eval__kb__one",
            attestation=SimpleNamespace(
                manifest_id="sha256:manifest",
                embedding_model="embedding",
                retrieval_capability=SimpleNamespace(value="dense"),
            ),
        )
        parameter_state = SimpleNamespace(
            collection_name="rag__kb__production",
            attestation=SimpleNamespace(manifest_id="sha256:production"),
        )
        build_profile = SimpleNamespace(
            chunking_config={"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 64}
        )
        retriever = _Retriever()
        node_postprocessors = []

        def retrieve(self, query: str):
            del query
            return self.retriever._retrieve(QueryBundle("question"))

        def close(self):
            pass

    target = FakeTarget()

    def target_factory(**kwargs):
        del kwargs
        return target

    class _FakeJudges:
        def __init__(self, llm):
            del llm

        async def evaluate_context(self, *, query, contexts):
            del query, contexts
            return SimpleNamespace(
                metric_name="context_relevancy",
                score=0.8,
                passing=True,
                feedback=None,
            )

    monkeypatch.setattr(runner, "BenchmarkJudges", _FakeJudges)

    writes: list[tuple] = []

    def writer(rows, *, db_url, sample_rows):
        writes.append((rows, db_url, sample_rows))

    run_benchmark(
        catalog_path=catalog,
        source_instance_id="kb.benchmark",
        alias="champion",
        rag_data_root=tmp_path,
        db_url="postgresql://db",
        runtime=object(),
        base_model="model-served-for-generation",
        judge_llm=object(),
        judge_model="gpt-4o-mini",
        judge_backend="openai_compatible",
        target_factory=target_factory,
        writer=writer,
    )

    row = writes[0][0][0]
    assert row["judge_backend"] == "openai_compatible"
    assert row["judge_model"] == "gpt-4o-mini"
    assert row["judge_model"] != "model-served-for-generation"
