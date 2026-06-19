"""Catalog-aware RAG benchmark execution and database persistence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from llama_index.core.llms import LLM
from pydantic import BaseModel, ConfigDict, Field

from app_config.catalog import load_catalog_with_source_index
from rag.contracts import DEFAULT_RAG_QUERY_PROMPTS
from rag.evaluation.judges import BenchmarkJudges
from rag.evaluation.models import (
    AnswerEvalObservation,
    BenchmarkCase,
    BenchmarkLabel,
    GenerationObservation,
    RetrievalEvalObservation,
    RetrievedChunk,
)
from rag.evaluation.retrieval import ProjectRetrievalEvalResult, ProjectRetrieverEvaluator
from rag.evaluation.target import BenchmarkTarget, materialize_benchmark_target
from rag.sources.benchmark_prep import (
    metadata_artifact_path,
    read_prepared_benchmark_artifacts,
)
from shared.db.eval_writer import write_evaluation_results

_BINARY_RETRIEVAL_METRICS = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]


class BenchmarkRunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_instance_id: str
    knowledge_base: str
    alias: str
    collection_name: str
    case_count: int = Field(ge=0)
    metric_values: dict[str, float] = Field(default_factory=dict)


def _case_query(case: BenchmarkCase) -> str:
    if case.query:
        return case.query
    for message in reversed(case.messages or []):
        if message.get("role") == "user" and str(message.get("content", "")).strip():
            return str(message["content"])
    raise ValueError(f"Benchmark case '{case.id}' has no usable query")


def _prepared_metadata(rag_data_root: Path | str, source_instance_id: str) -> dict[str, Any]:
    path = metadata_artifact_path(rag_data_root, source_instance_id)
    if not path.is_file():
        raise ValueError(
            f"Benchmark '{source_instance_id}' is not prepared; missing '{path}'"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _retrieval_observation(
    *,
    target: BenchmarkTarget,
    case_id: str,
    result: ProjectRetrievalEvalResult,
) -> RetrievalEvalObservation:
    retrieved = []
    for rank, node in enumerate(result.retrieved_nodes):
        metadata = node.node.metadata
        text = node.node.get_content()
        retrieved.append(
            RetrievedChunk(
                rank=rank,
                chunk_id=str(metadata.get("chunk_id") or node.node.node_id),
                document_id=str(
                    metadata.get("document_id")
                    or metadata.get("source_document_id")
                    or node.node.node_id
                ),
                source_instance_id=str(
                    metadata.get("source_instance_id") or target.source_instance_id
                ),
                score=float(node.score or 0.0),
                title=metadata.get("title"),
                uri=metadata.get("source_uri"),
                text_digest=f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}",
            )
        )
    return RetrievalEvalObservation(
        case_id=case_id,
        knowledge_base=target.kb.name,
        alias=target.alias,
        qdrant_alias=target.state.qdrant_alias,
        collection_name=target.state.collection_name,
        manifest_id=target.state.attestation.manifest_id,
        retrieval_strategy=target.alias_config.retrieval_strategy,
        retrieval_capability=target.state.attestation.retrieval_capability.value,
        top_k=target.alias_config.top_k,
        score_threshold=target.alias_config.score_threshold,
        reranker=target.alias_config.reranker,
        retrieved=retrieved,
    )


async def _execute(
    *,
    target: BenchmarkTarget,
    cases: list[BenchmarkCase],
    labels: dict[str, BenchmarkLabel],
    suites: list[str],
    generation_llm: LLM | None,
    judge_llm: LLM | None,
) -> tuple[dict[str, list[float]], dict[str, list[dict[str, Any]]]]:
    scores: dict[str, list[float]] = defaultdict(list)
    details: dict[str, list[dict[str, Any]]] = defaultdict(list)
    retrieval_evaluator = ProjectRetrieverEvaluator.from_metric_names(
        _BINARY_RETRIEVAL_METRICS,
        retriever=target.retriever,
        node_postprocessors=target.node_postprocessors,
    )
    assert isinstance(retrieval_evaluator, ProjectRetrieverEvaluator)
    needs_judges = "context_quality" in suites or "generation_quality" in suites
    judges = BenchmarkJudges(judge_llm) if needs_judges and judge_llm is not None else None

    for case in cases:
        query = _case_query(case)
        label = labels.get(case.id, BenchmarkLabel(case_id=case.id))
        if "retrieval_quality" in suites:
            entity_types = sorted({qrel.entity_type for qrel in label.qrels})
            if not entity_types:
                raise ValueError(f"Retrieval case '{case.id}' has no qrels")
            for entity_type in entity_types:
                result = await retrieval_evaluator.aevaluate_project(
                    query=query,
                    qrels=label.qrels,
                    entity_type=entity_type,
                    metadata={"case_id": case.id},
                )
                observation = _retrieval_observation(
                    target=target,
                    case_id=case.id,
                    result=result,
                )
                for name, value in result.metric_scores.items():
                    metric = f"{entity_type}_{name}"
                    scores[metric].append(value)
                    details[metric].append(
                        {
                            "case_id": case.id,
                            "score": value,
                            "expected_ids": result.expected_ids,
                            "retrieved_ids": result.retrieved_ids,
                            "retrieved_scores": [node.score for node in result.retrieved_nodes],
                            "entity_type": entity_type,
                            "observation": observation.model_dump(mode="json"),
                        }
                    )

        if "context_quality" in suites:
            if judges is None:
                raise ValueError("context_quality requires a judge LLM")
            nodes = target.retrieve(query)
            contexts = [node.node.get_content() for node in nodes]
            judged = await judges.evaluate_context(query=query, contexts=contexts)
            scores[judged.metric_name].append(judged.score)
            details[judged.metric_name].append(
                {
                    "case_id": case.id,
                    "score": judged.score,
                    "passing": judged.passing,
                    "feedback": judged.feedback,
                    "retrieved_ids": [node.node.node_id for node in nodes],
                }
            )

        if "generation_quality" in suites:
            if generation_llm is None or judges is None:
                raise ValueError("generation_quality requires generation and judge LLMs")
            response = target.query(
                query,
                llm=generation_llm,
                prompts=DEFAULT_RAG_QUERY_PROMPTS,
            )
            answer = str(response)
            contexts = [node.node.get_content() for node in response.source_nodes]
            for judged in await judges.evaluate_generation(
                query=query,
                answer=answer,
                contexts=contexts,
                reference_answers=label.reference_answers,
            ):
                prompt_identity = DEFAULT_RAG_QUERY_PROMPTS.identity
                observation = AnswerEvalObservation(
                    case_id=case.id,
                    answer=answer,
                    cited_chunk_ids=[node.node.node_id for node in response.source_nodes],
                    generation=GenerationObservation(
                        prompt_id=prompt_identity.prompt_id,
                        prompt_version=prompt_identity.prompt_version,
                        prompt_digest=prompt_identity.prompt_digest,
                        prompt_params=prompt_identity.prompt_params,
                    ),
                )
                scores[judged.metric_name].append(judged.score)
                details[judged.metric_name].append(
                    {
                        "case_id": case.id,
                        "score": judged.score,
                        "passing": judged.passing,
                        "feedback": judged.feedback,
                        "answer": answer,
                        "reference_answers": label.reference_answers,
                        "cited_chunk_ids": [node.node.node_id for node in response.source_nodes],
                        "prompt_identity": DEFAULT_RAG_QUERY_PROMPTS.identity.model_dump(
                            mode="json"
                        ),
                        "observation": observation.model_dump(mode="json"),
                    }
                )
    return scores, details


def run_benchmark(
    *,
    catalog_path: Path | str,
    source_instance_id: str,
    alias: str,
    rag_data_root: Path | str,
    db_url: str,
    runtime,
    base_model: str,
    generation_llm: LLM | None = None,
    judge_llm: LLM | None = None,
    judge_model: str | None = None,
    target_factory: Callable[..., BenchmarkTarget] = materialize_benchmark_target,
    writer: Callable[..., None] = write_evaluation_results,
) -> BenchmarkRunSummary:
    """Execute one prepared benchmark against one mandatory, explicit KB alias."""
    if not alias.strip():
        raise ValueError("benchmark execution requires an explicit alias")
    _, kb_index, source_index = load_catalog_with_source_index(catalog_path)
    instance = source_index.get(source_instance_id)
    if instance.role != "benchmark" or instance.benchmark is None:
        raise ValueError(f"Source instance '{source_instance_id}' is not a benchmark")
    kb = kb_index[instance.knowledge_base]
    if alias not in kb.aliases:
        raise ValueError(f"Unknown alias '{alias}' for KB '{kb.name}'")

    artifacts = read_prepared_benchmark_artifacts(rag_data_root, source_instance_id)
    if not artifacts.cases:
        raise ValueError(f"Benchmark '{source_instance_id}' has no prepared cases")
    metadata = _prepared_metadata(rag_data_root, source_instance_id)
    label_index = {label.case_id: label for label in artifacts.labels}
    if len(label_index) != len(artifacts.labels):
        raise ValueError("benchmark labels contain duplicate case ids")

    target = target_factory(
        runtime=runtime,
        source_instance_id=source_instance_id,
        kb=kb,
        alias=alias,
        artifacts=artifacts,
        rag_data_root=rag_data_root,
    )
    try:
        scores, sample_details = asyncio.run(
            _execute(
                target=target,
                cases=artifacts.cases,
                labels=label_index,
                suites=list(instance.benchmark.suites),
                generation_llm=generation_llm,
                judge_llm=judge_llm,
            )
        )
        now = datetime.now(tz=UTC)
        rows: list[dict[str, Any]] = []
        samples: list[dict[str, Any]] = []
        metric_values: dict[str, float] = {}
        for metric_name, values in scores.items():
            run_id = uuid.uuid4()
            metric_value = sum(values) / len(values)
            metric_values[metric_name] = metric_value
            chunking = target.build_profile.chunking_config
            uses_judge = metric_name in {
                "context_relevancy",
                "faithfulness",
                "answer_relevancy",
                "correctness",
            }
            rows.append(
                {
                    "id": run_id,
                    "created_at": now,
                    "finished_at": now,
                    "status": "completed",
                    "task": "rag",
                    "dataset_name": source_instance_id,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "base_model": base_model,
                    "rag_enabled": True,
                    "rag_alias": alias,
                    "knowledge_base": kb.name,
                    "qdrant_alias": target.state.qdrant_alias,
                    "qdrant_collection": target.state.collection_name,
                    "rag_manifest_id": target.state.attestation.manifest_id,
                    "embedding_model": target.state.attestation.embedding_model,
                    "chunking_strategy": str(chunking.get("strategy") or "") or None,
                    "chunk_size": chunking.get("chunk_size"),
                    "chunk_overlap": chunking.get("chunk_overlap"),
                    "retrieval_top_k": target.alias_config.top_k,
                    "score_threshold": target.alias_config.score_threshold,
                    "reranking_strategy": target.alias_config.reranker,
                    "judge_backend": "llamaindex" if uses_judge else None,
                    "judge_model": (judge_model or base_model) if uses_judge else None,
                    "extra": {
                        "benchmark_source_instance_id": source_instance_id,
                        "benchmark_artifact_digests": metadata.get("artifact_digests", {}),
                        "benchmark_suites": list(instance.benchmark.suites),
                        "benchmark_adapter_id": metadata.get("adapter_id"),
                        "benchmark_adapter_version": metadata.get("adapter_version"),
                        "parameter_source_collection": target.parameter_state.collection_name,
                        "parameter_source_manifest_id": (
                            target.parameter_state.attestation.manifest_id
                        ),
                        "retrieval_strategy": target.alias_config.retrieval_strategy,
                        "retrieval_capability": (
                            target.state.attestation.retrieval_capability.value
                        ),
                        "prompt_identity": (
                            DEFAULT_RAG_QUERY_PROMPTS.identity.model_dump(mode="json")
                            if "generation_quality" in instance.benchmark.suites
                            else None
                        ),
                    },
                }
            )
            case_by_id = {case.id: case for case in artifacts.cases}
            for sample_idx, detail in enumerate(sample_details[metric_name]):
                sample_case = case_by_id[detail["case_id"]]
                samples.append(
                    {
                        "eval_run_id": run_id,
                        "sample_idx": sample_idx,
                        "sample_id": detail["case_id"],
                        "input": _case_query(sample_case),
                        "output": detail.get("answer"),
                        "reference": "\n".join(detail.get("reference_answers", [])) or None,
                        "detail": detail,
                    }
                )
        writer(rows, db_url=db_url, sample_rows=samples)
        return BenchmarkRunSummary(
            source_instance_id=source_instance_id,
            knowledge_base=kb.name,
            alias=alias,
            collection_name=target.state.collection_name,
            case_count=len(artifacts.cases),
            metric_values=metric_values,
        )
    finally:
        target.close()
