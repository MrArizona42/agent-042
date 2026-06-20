"""LlamaIndex judge adapters for context and generation benchmark suites."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

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
    """Small project wrapper around LlamaIndex's generation evaluators.

    Each evaluator is built lazily on first use: a context_quality-only run
    never constructs faithfulness/answer_relevancy/correctness, and a
    generation_quality run without reference answers never constructs
    correctness.
    """

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    @cached_property
    def context_relevancy(self) -> RelevancyEvaluator:
        return RelevancyEvaluator(llm=self._llm)

    @cached_property
    def faithfulness(self) -> FaithfulnessEvaluator:
        return FaithfulnessEvaluator(llm=self._llm)

    @cached_property
    def answer_relevancy(self) -> RelevancyEvaluator:
        return RelevancyEvaluator(llm=self._llm)

    @cached_property
    def correctness(self) -> CorrectnessEvaluator:
        return CorrectnessEvaluator(llm=self._llm)

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
