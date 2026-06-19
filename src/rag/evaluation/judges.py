"""LlamaIndex judge adapters for context and generation benchmark suites."""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)
from llama_index.core.llms import LLM


@dataclass(frozen=True, slots=True)
class JudgeResult:
    metric_name: str
    score: float
    passing: bool | None
    feedback: str | None


class BenchmarkJudges:
    """Small project wrapper around LlamaIndex's generation evaluators."""

    def __init__(self, llm: LLM) -> None:
        self.context_relevancy = RelevancyEvaluator(llm=llm)
        self.faithfulness = FaithfulnessEvaluator(llm=llm)
        self.answer_relevancy = RelevancyEvaluator(llm=llm)
        self.correctness = CorrectnessEvaluator(llm=llm)

    @staticmethod
    def _result(metric_name: str, result) -> JudgeResult:
        return JudgeResult(
            metric_name=metric_name,
            score=float(result.score or 0.0),
            passing=result.passing,
            feedback=result.feedback,
        )

    async def evaluate_context(self, *, query: str, contexts: list[str]) -> JudgeResult:
        response = "\n\n".join(contexts)
        result = await self.context_relevancy.aevaluate(
            query=query,
            response=response,
            contexts=contexts,
        )
        return self._result("context_relevancy", result)

    async def evaluate_generation(
        self,
        *,
        query: str,
        answer: str,
        contexts: list[str],
        reference_answers: list[str],
    ) -> list[JudgeResult]:
        results = [
            self._result(
                "faithfulness",
                await self.faithfulness.aevaluate(response=answer, contexts=contexts),
            ),
            self._result(
                "answer_relevancy",
                await self.answer_relevancy.aevaluate(
                    query=query,
                    response=answer,
                    contexts=contexts,
                ),
            ),
        ]
        if reference_answers:
            results.append(
                self._result(
                    "correctness",
                    await self.correctness.aevaluate(
                        query=query,
                        response=answer,
                        reference="\n".join(reference_answers),
                    ),
                )
            )
        return results
