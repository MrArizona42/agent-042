"""Tests for rag.evaluation.judges.BenchmarkJudges lazy evaluator construction.

Phase 6 acceptance: judge construction stays lazy by suite. A context_quality
run must never construct faithfulness/answer_relevancy/correctness; a
generation_quality run without reference answers must never construct
correctness.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag.evaluation.judges import BenchmarkJudges


def _result(score: float = 0.9) -> MagicMock:
    result = MagicMock()
    result.score = score
    result.passing = True
    result.feedback = None
    return result


class TestLazyConstruction:
    def test_init_constructs_no_evaluators(self):
        with (
            patch("rag.evaluation.judges.RelevancyEvaluator") as relevancy,
            patch("rag.evaluation.judges.FaithfulnessEvaluator") as faithfulness,
            patch("rag.evaluation.judges.CorrectnessEvaluator") as correctness,
        ):
            BenchmarkJudges(llm=object())

            relevancy.assert_not_called()
            faithfulness.assert_not_called()
            correctness.assert_not_called()

    @pytest.mark.parametrize("has_reference_answers", [False, True])
    def test_context_quality_only_constructs_context_relevancy(self, has_reference_answers):
        with (
            patch("rag.evaluation.judges.RelevancyEvaluator") as relevancy,
            patch("rag.evaluation.judges.FaithfulnessEvaluator") as faithfulness,
            patch("rag.evaluation.judges.CorrectnessEvaluator") as correctness,
        ):
            relevancy.return_value.aevaluate = AsyncMock(return_value=_result())
            judges = BenchmarkJudges(llm=object())

            asyncio.run(judges.evaluate_context(query="q", contexts=["c"]))

            relevancy.assert_called_once()
            faithfulness.assert_not_called()
            correctness.assert_not_called()

    def test_generation_quality_without_references_never_constructs_correctness(self):
        with (
            patch("rag.evaluation.judges.RelevancyEvaluator") as relevancy,
            patch("rag.evaluation.judges.FaithfulnessEvaluator") as faithfulness,
            patch("rag.evaluation.judges.CorrectnessEvaluator") as correctness,
        ):
            relevancy.return_value.aevaluate = AsyncMock(return_value=_result())
            faithfulness.return_value.aevaluate = AsyncMock(return_value=_result())
            judges = BenchmarkJudges(llm=object())

            asyncio.run(
                judges.evaluate_generation(
                    query="q", answer="a", contexts=["c"], reference_answers=[]
                )
            )

            faithfulness.assert_called_once()
            relevancy.assert_called_once()
            correctness.assert_not_called()

    def test_generation_quality_with_references_constructs_correctness(self):
        with (
            patch("rag.evaluation.judges.RelevancyEvaluator") as relevancy,
            patch("rag.evaluation.judges.FaithfulnessEvaluator") as faithfulness,
            patch("rag.evaluation.judges.CorrectnessEvaluator") as correctness,
        ):
            relevancy.return_value.aevaluate = AsyncMock(return_value=_result())
            faithfulness.return_value.aevaluate = AsyncMock(return_value=_result())
            correctness.return_value.aevaluate = AsyncMock(return_value=_result())
            judges = BenchmarkJudges(llm=object())

            asyncio.run(
                judges.evaluate_generation(
                    query="q", answer="a", contexts=["c"], reference_answers=["ref"]
                )
            )

            correctness.assert_called_once()

    def test_evaluator_is_cached_across_calls(self):
        with patch("rag.evaluation.judges.RelevancyEvaluator") as relevancy:
            relevancy.return_value.aevaluate = AsyncMock(return_value=_result())
            judges = BenchmarkJudges(llm=object())

            asyncio.run(judges.evaluate_context(query="q1", contexts=["c"]))
            asyncio.run(judges.evaluate_context(query="q2", contexts=["c"]))

            relevancy.assert_called_once()
