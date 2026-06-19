"""LlamaIndex retrieval evaluation with project graded-qrel support."""

from __future__ import annotations

import math
from typing import Any, Sequence

from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.evaluation.retrieval.base import RetrievalEvalMode
from llama_index.core.evaluation.retrieval.metrics_base import RetrievalMetricResult
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import BaseModel, ConfigDict, Field

from rag.evaluation.models import EntityType, Qrel


class ProjectRetrievalEvalResult(BaseModel):
    """One retrieval evaluation, including native nodes and project metrics."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    query: str
    entity_type: EntityType
    expected_ids: list[str] = Field(default_factory=list)
    retrieved_ids: list[str] = Field(default_factory=list)
    retrieved_nodes: list[NodeWithScore] = Field(default_factory=list)
    metric_scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _entity_id(node: NodeWithScore, entity_type: EntityType) -> str:
    metadata = node.node.metadata
    if entity_type == "document":
        return str(
            metadata.get("document_id")
            or metadata.get("source_document_id")
            or node.node.node_id
        )
    return str(metadata.get("chunk_id") or node.node.node_id)


def graded_ndcg(qrels: Sequence[Qrel], retrieved_ids: Sequence[str], *, k: int) -> float:
    """Compute NDCG@k using integer relevance grades."""
    if k <= 0:
        raise ValueError("k must be positive")
    grades = {qrel.entity_id: qrel.relevance_grade for qrel in qrels}
    if not grades:
        return 0.0

    def dcg(values: Sequence[int]) -> float:
        return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(values))

    actual = [grades.get(entity_id, 0) for entity_id in retrieved_ids[:k]]
    ideal = sorted(grades.values(), reverse=True)[:k]
    denominator = dcg(ideal)
    return dcg(actual) / denominator if denominator else 0.0


class ProjectRetrieverEvaluator(RetrieverEvaluator):
    """Retain LlamaIndex retrieval mechanics while evaluating project qrels."""

    async def _aretrieve_nodes(self, query: str) -> list[NodeWithScore]:
        nodes = list(await self.retriever.aretrieve(query))
        if self.node_postprocessors:
            bundle = QueryBundle(query)
            for postprocessor in self.node_postprocessors:
                nodes = postprocessor.postprocess_nodes(nodes, query_bundle=bundle)
        return nodes

    async def aevaluate_project(
        self,
        *,
        query: str,
        qrels: Sequence[Qrel],
        entity_type: EntityType,
        metadata: dict[str, Any] | None = None,
    ) -> ProjectRetrievalEvalResult:
        """Evaluate binary LlamaIndex metrics and graded NDCG from one retrieval."""
        nodes = await self._aretrieve_nodes(query)
        retrieved_ids = [_entity_id(node, entity_type) for node in nodes]
        selected_qrels = [qrel for qrel in qrels if qrel.entity_type == entity_type]
        expected_ids = [qrel.entity_id for qrel in selected_qrels if qrel.relevance_grade > 0]
        retrieved_texts = [node.node.get_content() for node in nodes]

        scores: dict[str, float] = {}
        for metric in self.metrics:
            if expected_ids and retrieved_ids:
                result: RetrievalMetricResult = metric.compute(
                    query=query,
                    expected_ids=expected_ids,
                    retrieved_ids=retrieved_ids,
                    retrieved_texts=retrieved_texts,
                )
                scores[metric.metric_name] = float(result.score)
            else:
                scores[metric.metric_name] = 0.0
        if selected_qrels:
            scores[f"graded_ndcg@{len(retrieved_ids)}"] = graded_ndcg(
                selected_qrels,
                retrieved_ids,
                k=max(1, len(retrieved_ids)),
            )

        return ProjectRetrievalEvalResult(
            query=query,
            entity_type=entity_type,
            expected_ids=expected_ids,
            retrieved_ids=retrieved_ids,
            retrieved_nodes=nodes,
            metric_scores=scores,
            metadata=dict(metadata or {}),
        )

    async def _aget_retrieved_ids_and_texts(
        self,
        query: str,
        mode: RetrievalEvalMode = RetrievalEvalMode.TEXT,
    ) -> tuple[list[str], list[str]]:
        del mode
        nodes = await self._aretrieve_nodes(query)
        return (
            [node.node.node_id for node in nodes],
            [node.node.get_content() for node in nodes],
        )
