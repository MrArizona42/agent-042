"""Tests for the evaluation workflow: DB model, metrics, runner, and settings."""

from __future__ import annotations

import json
import sys
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from qdrant_client.models import SparseVector

from rag.vector_store import Document


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_settings_caches():
    """Clear lru_cache on settings between tests."""
    import shared.config as cfg

    cfg.get_settings.cache_clear()
    cfg.clear_knowledge_base_caches()
    yield
    cfg.get_settings.cache_clear()
    cfg.clear_knowledge_base_caches()


# ---------------------------------------------------------------------------
# EvalRun model tests
# ---------------------------------------------------------------------------


class TestEvalRunModel:
    """Tests for the EvalRun SQLAlchemy ORM model."""

    def test_eval_run_table_name(self):
        from shared.db.models import EvalRun

        assert EvalRun.__tablename__ == "eval_runs"

    def test_eval_run_has_required_columns(self):
        from shared.db.models import EvalRun

        columns = {c.name for c in EvalRun.__table__.columns}
        required = {
            "id",
            "created_at",
            "finished_at",
            "status",
            "task",
            "dataset_name",
            "metric_name",
            "metric_value",
            "base_model",
            "adapter_name",
            "adapter_version",
            "adapter_mlflow_run_id",
            "lora_alias",
            "rag_enabled",
            "rag_alias",
            "knowledge_base",
            "qdrant_alias",
            "qdrant_collection",
            "rag_manifest_id",
            "embedding_model",
            "chunking_strategy",
            "chunk_size",
            "chunk_overlap",
            "retrieval_top_k",
            "score_threshold",
            "qdrant_snapshot_id",
            "dataset_dvc_hash",
            "reranking_strategy",
            "judge_backend",
            "judge_model",
            "bert_score_model",
            "temperature",
            "max_tokens",
            "eval_verdict",
            "extra",
            "error_message",
        }
        assert required.issubset(columns)

    def test_eval_run_default_status(self):
        from shared.db.models import EvalRun

        col = EvalRun.__table__.c.status
        assert col.default.arg == "running"

    def test_eval_run_default_rag_enabled(self):
        from shared.db.models import EvalRun

        col = EvalRun.__table__.c.rag_enabled
        assert col.default.arg is False

    def test_eval_run_extra_is_jsonb(self):
        from sqlalchemy.dialects.postgresql import JSONB

        from shared.db.models import EvalRun

        col = EvalRun.__table__.c.extra
        assert isinstance(col.type, JSONB)


# ---------------------------------------------------------------------------
# EvalSettings tests
# ---------------------------------------------------------------------------


class TestEvalSettings:
    """Tests for the EvalSettings configuration."""

    def test_defaults(self):
        from shared.config import get_settings

        settings = get_settings()
        s = settings.eval
        resolved_judge = s.resolve_judge_settings(settings.platform)

        assert s.judge.backend == "local_vllm"
        assert s.judge.model == "/models/Qwen/Qwen3-0.6B"
        assert resolved_judge.backend == "local_vllm"
        assert resolved_judge.model == "/models/Qwen/Qwen3-0.6B"
        assert resolved_judge.base_url == "http://localhost:8000"
        assert s.metrics.temperature == 0.0
        assert s.metrics.max_completion_tokens == 2048
        assert s.sandbox.code_exec_timeout == 30
        assert s.sandbox.code_exec_mem_limit == "512m"
        assert s.metrics.bert_score_model == "microsoft/deberta-v3-base"
        assert s.judge.backend == "local_vllm"
        assert s.metrics.max_completion_tokens == 2048
        assert s.sandbox.code_exec_timeout == 30

    def test_explicit_override_and_secret_env(self, monkeypatch):
        from shared.config import load_settings

        monkeypatch.setenv("EVAL__JUDGE__API_KEY", "secret")
        settings = load_settings(
            {
                "eval": {
                    "judge": {
                        "model": "judge-model",
                        "backend": "openai_compatible",
                        "base_url": "https://judge.example",
                    },
                    "metrics": {"temperature": 0.7},
                }
            }
        )
        s = settings.eval
        resolved_judge = s.resolve_judge_settings(settings.platform)

        assert s.judge.backend == "openai_compatible"
        assert s.judge.model == "judge-model"
        assert resolved_judge.backend == "openai_compatible"
        assert resolved_judge.model == "judge-model"
        assert resolved_judge.base_url == "https://judge.example"
        assert resolved_judge.api_key == "secret"
        assert s.metrics.temperature == 0.7
        assert s.judge.base_url == "https://judge.example"
        assert s.metrics.temperature == 0.7


# ---------------------------------------------------------------------------
# Automatic metrics tests
# ---------------------------------------------------------------------------


class TestAutomaticMetrics:
    """Tests for automatic evaluation metrics."""

    def test_compute_bertscore_uses_fast_tokenizer(self, monkeypatch):
        from experiments.eval.eval_scripts.metrics import automatic

        calls = {}

        class FakeTensor:
            def __init__(self, value):
                self._value = value

            def mean(self):
                return self

            def item(self):
                return self._value

        class FakeNoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeScorer:
            def __init__(self, *, model_type, use_fast_tokenizer):
                calls["model_type"] = model_type
                calls["use_fast_tokenizer"] = use_fast_tokenizer
                self._model = types.SimpleNamespace(
                    config=types.SimpleNamespace(max_position_embeddings=512)
                )
                self._tokenizer = types.SimpleNamespace(model_max_length=10**30)

            def score(self, predictions, references):
                calls["predictions"] = predictions
                calls["references"] = references
                calls["model_max_length_after_cap"] = self._tokenizer.model_max_length
                return FakeTensor(0.1), FakeTensor(0.2), FakeTensor(0.3)

        fake_bert_score = types.ModuleType("bert_score")
        fake_bert_score.BERTScorer = FakeScorer
        fake_torch = types.ModuleType("torch")
        fake_torch.no_grad = FakeNoGrad
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
        )

        monkeypatch.setitem(sys.modules, "bert_score", fake_bert_score)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        result = automatic.compute_bertscore(
            predictions=["predicted answer"],
            references=["reference answer"],
            model_name="microsoft/deberta-v3-base",
        )

        assert calls["model_type"] == "microsoft/deberta-v3-base"
        assert calls["use_fast_tokenizer"] is True
        assert calls["predictions"] == ["predicted answer"]
        assert calls["references"] == ["reference answer"]
        assert calls["model_max_length_after_cap"] == 512
        assert result == {
            "bertscore_precision": pytest.approx(0.1),
            "bertscore_recall": pytest.approx(0.2),
            "bertscore_f1": pytest.approx(0.3),
        }

    def test_compute_bertscore_treats_empty_pairs_as_zero(self, monkeypatch):
        from experiments.eval.eval_scripts.metrics import automatic

        calls = {}

        class FakeTensor:
            def __init__(self, value):
                self._value = value

            def mean(self):
                return self

            def item(self):
                return self._value

        class FakeNoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeScorer:
            def __init__(self, *, model_type, use_fast_tokenizer):
                calls["model_type"] = model_type
                calls["use_fast_tokenizer"] = use_fast_tokenizer
                self._model = types.SimpleNamespace(
                    config=types.SimpleNamespace(max_position_embeddings=512)
                )
                self._tokenizer = types.SimpleNamespace(model_max_length=512)

            def score(self, predictions, references):
                calls["predictions"] = predictions
                calls["references"] = references
                return FakeTensor(0.1), FakeTensor(0.2), FakeTensor(0.3)

        fake_bert_score = types.ModuleType("bert_score")
        fake_bert_score.BERTScorer = FakeScorer
        fake_torch = types.ModuleType("torch")
        fake_torch.no_grad = FakeNoGrad
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
        )

        monkeypatch.setitem(sys.modules, "bert_score", fake_bert_score)
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        result = automatic.compute_bertscore(
            predictions=["", "predicted answer"],
            references=["reference answer", "target answer"],
            model_name="microsoft/deberta-v3-base",
        )

        assert calls["model_type"] == "microsoft/deberta-v3-base"
        assert calls["use_fast_tokenizer"] is True
        assert calls["predictions"] == ["predicted answer"]
        assert calls["references"] == ["target answer"]
        assert result == {
            "bertscore_precision": pytest.approx(0.05),
            "bertscore_recall": pytest.approx(0.1),
            "bertscore_f1": pytest.approx(0.15),
        }

    def test_rouge_l_identical(self):
        from experiments.eval.eval_scripts.metrics.automatic import compute_rouge_l

        assert compute_rouge_l("hello world", "hello world") == 1.0

    def test_rouge_l_empty(self):
        from experiments.eval.eval_scripts.metrics.automatic import compute_rouge_l

        assert compute_rouge_l("", "hello") == 0.0
        assert compute_rouge_l("hello", "") == 0.0

    def test_rouge_l_partial(self):
        from experiments.eval.eval_scripts.metrics.automatic import compute_rouge_l

        score = compute_rouge_l("the cat sat on the mat", "the cat on the mat")
        assert 0.0 < score < 1.0

    def test_recall_at_k(self):
        from experiments.eval.eval_scripts.metrics.automatic import compute_recall_at_k

        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "f"}
        recall = compute_recall_at_k(retrieved, relevant, k=5)
        assert recall == pytest.approx(2 / 3)

    def test_recall_at_k_empty(self):
        from experiments.eval.eval_scripts.metrics.automatic import compute_recall_at_k

        assert compute_recall_at_k(["a", "b"], set(), k=5) == 0.0

    def test_ndcg_at_k_perfect(self):
        from experiments.eval.eval_scripts.metrics.automatic import compute_ndcg_at_k

        retrieved = ["a", "b", "c"]
        labels = {"a": 3.0, "b": 2.0, "c": 1.0}
        ndcg = compute_ndcg_at_k(retrieved, labels, k=3)
        assert ndcg == pytest.approx(1.0)

    def test_ndcg_at_k_reversed(self):
        from experiments.eval.eval_scripts.metrics.automatic import compute_ndcg_at_k

        retrieved = ["c", "b", "a"]
        labels = {"a": 3.0, "b": 2.0, "c": 1.0}
        ndcg = compute_ndcg_at_k(retrieved, labels, k=3)
        assert 0.0 < ndcg < 1.0

    def test_ndcg_at_k_deduplicates_chunk_ids(self):
        from experiments.eval.eval_scripts.metrics.automatic import compute_ndcg_at_k

        # Simulate a relevant doc split into 3 chunks all appearing in results.
        # Duplicate source ids must be counted only once so the score equals
        # the non-duplicated case.
        labels = {"doc1": 1.0}
        retrieved_deduped = ["doc1", "doc2", "doc3"]
        retrieved_with_dupes = ["doc1", "doc1", "doc1", "doc2", "doc3"]

        score_deduped = compute_ndcg_at_k(retrieved_deduped, labels, k=3)
        score_duped = compute_ndcg_at_k(retrieved_with_dupes, labels, k=3)

        assert score_deduped == pytest.approx(score_duped)
        assert score_deduped == pytest.approx(1.0)  # doc1 is first, perfect ranking

    def test_mrr_at_k_first_hit(self):
        from experiments.eval.eval_scripts.metrics.automatic import compute_mrr_at_k

        relevant = {"doc2"}
        retrieved = ["doc1", "doc2", "doc3"]
        assert compute_mrr_at_k(retrieved, relevant, k=10) == pytest.approx(1 / 2)

    def test_mrr_at_k_no_hit(self):
        from experiments.eval.eval_scripts.metrics.automatic import compute_mrr_at_k

        relevant = {"doc99"}
        retrieved = ["doc1", "doc2", "doc3"]
        assert compute_mrr_at_k(retrieved, relevant, k=10) == pytest.approx(0.0)


class TestNqDatasetParsing:
    """Tests for Natural Questions sample parsing."""

    def test_extract_nq_answer_from_list_of_dict_short_answers(self):
        from experiments.eval.eval_scripts.datasets import _extract_nq_answer

        item = {
            "annotations": {
                "short_answers": [
                    {"text": []},
                    {"text": ["enabled European empire expansion"]},
                ]
            }
        }

        assert _extract_nq_answer(item) == "enabled European empire expansion"

    def test_extract_nq_answer_from_nested_short_answers(self):
        from experiments.eval.eval_scripts.datasets import _extract_nq_answer

        item = {
            "annotations": {
                "short_answers": [
                    [
                        {"text": []},
                        {"text": ["nested answer"]},
                    ]
                ]
            }
        }

        assert _extract_nq_answer(item) == "nested answer"

    def test_extract_nq_answer_from_token_span_fallback(self):
        from experiments.eval.eval_scripts.datasets import _extract_nq_answer

        item = {
            "document": {"tokens": {"token": ["a", "b", "enabled", "trade", "routes", "x"]}},
            "annotations": {
                "short_answers": [
                    {
                        "text": [],
                        "start_token": [2],
                        "end_token": [5],
                    }
                ]
            },
        }

        assert _extract_nq_answer(item) == "enabled trade routes"

    def test_mrr_at_k_deduplicates_chunk_ids(self):
        from experiments.eval.eval_scripts.metrics.automatic import compute_mrr_at_k

        # Duplicate chunks for the same source doc should not inflate rank.
        # doc2 is relevant; after dedup it sits at rank 2.
        relevant = {"doc2"}
        retrieved_duped = ["doc1", "doc1", "doc2", "doc3"]
        retrieved_clean = ["doc1", "doc2", "doc3"]
        assert compute_mrr_at_k(retrieved_duped, relevant, k=10) == pytest.approx(
            compute_mrr_at_k(retrieved_clean, relevant, k=10)
        )

    def test_mrr_at_k_respects_cutoff(self):
        from experiments.eval.eval_scripts.metrics.automatic import compute_mrr_at_k

        relevant = {"doc3"}
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        assert compute_mrr_at_k(retrieved, relevant, k=2) == pytest.approx(0.0)
        assert compute_mrr_at_k(retrieved, relevant, k=3) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# LLM judge tests (mocked)
# ---------------------------------------------------------------------------


class TestLLMJudge:
    """Tests for LLM-as-Judge with mocked judge model calls."""

    @patch("experiments.eval.eval_scripts.metrics.llm_judge._call_judge_model")
    def test_judge_single_relevance(self, mock_judge_model):
        from experiments.eval.eval_scripts.metrics.llm_judge import judge_single
        from shared.config import JudgeSettings

        mock_judge_model.return_value = {"score": 4, "reason": "mostly relevant"}

        result = judge_single(
            "relevance",
            question="What is ML?",
            answer="Machine learning is...",
            reference="ML is a subset of AI",
            judge_settings=JudgeSettings(
                backend="local_vllm",
                model="judge-model",
                base_url="http://localhost:8000",
            ),
        )
        assert result["score"] == 4
        assert "relevant" in result["reason"]
        mock_judge_model.assert_called_once()

    @patch("experiments.eval.eval_scripts.metrics.llm_judge._call_judge_model")
    def test_judge_batch(self, mock_judge_model):
        from experiments.eval.eval_scripts.metrics.llm_judge import judge_batch
        from shared.config import JudgeSettings

        mock_judge_model.return_value = {"score": 3, "reason": "ok"}

        result = judge_batch(
            "correctness",
            samples=[
                {"question": "q1", "answer": "a1", "reference": "r1"},
                {"question": "q2", "answer": "a2", "reference": "r2"},
            ],
            judge_settings=JudgeSettings(
                backend="openai_compatible",
                model="judge-model",
                base_url="https://judge.example",
                api_key="secret",
            ),
        )
        assert "correctness" in result
        assert result["correctness"] == 3.0

    def test_judge_unknown_metric(self):
        from experiments.eval.eval_scripts.metrics.llm_judge import judge_single
        from shared.config import JudgeSettings

        with pytest.raises(ValueError, match="Unknown judge metric"):
            judge_single(
                "nonexistent_metric",
                answer="test",
                judge_settings=JudgeSettings(
                    backend="local_vllm",
                    model="judge-model",
                    base_url="http://localhost:8000",
                ),
            )


# ---------------------------------------------------------------------------
# Code execution tests (mocked)
# ---------------------------------------------------------------------------


class TestCodeExec:
    """Tests for sandboxed code execution metrics."""

    def test_evaluate_humaneval_sample_invokes_entry_point_check(self):
        from experiments.eval.eval_scripts.metrics.code_exec import evaluate_humaneval_sample

        with patch(
            "experiments.eval.eval_scripts.metrics.code_exec._run_in_sandbox",
            return_value={
                "passed": True,
                "executable": True,
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
            },
        ) as mock_run:
            evaluate_humaneval_sample(
                prompt='def add(a, b):\n    """Return the sum."""',
                generated_code="    return a + b",
                test_code="def check(candidate):\n    assert candidate(1, 2) == 3",
                entry_point="add",
                timeout=30,
                mem_limit="512m",
                cpus=1.0,
            )

        full_code = mock_run.call_args.args[0]
        assert "check(add)" in full_code
        assert "sys.exit(100)" in full_code

    def test_pass_at_1_all_pass(self):
        from experiments.eval.eval_scripts.metrics.code_exec import compute_pass_at_1

        results = [
            {"passed": True, "executable": True, "exit_code": 0},
            {"passed": True, "executable": True, "exit_code": 0},
        ]
        metrics = compute_pass_at_1(results)
        assert metrics["pass_at_1"] == 1.0
        assert metrics["executable_rate"] == 1.0

    def test_pass_at_1_none_pass(self):
        from experiments.eval.eval_scripts.metrics.code_exec import compute_pass_at_1

        results = [
            {"passed": False, "executable": False, "exit_code": 1},
            {"passed": False, "executable": False, "exit_code": -1},
        ]
        metrics = compute_pass_at_1(results)
        assert metrics["pass_at_1"] == 0.0
        assert metrics["executable_rate"] == 0.0

    def test_pass_at_1_empty(self):
        from experiments.eval.eval_scripts.metrics.code_exec import compute_pass_at_1

        metrics = compute_pass_at_1([])
        assert metrics["pass_at_1"] == 0.0
        assert metrics["executable_rate"] == 0.0

    def test_pass_at_1_partial(self):
        from experiments.eval.eval_scripts.metrics.code_exec import compute_pass_at_1

        results = [
            {"passed": True, "executable": True, "exit_code": 0},
            {"passed": False, "executable": True, "exit_code": 100},
            {"passed": True, "executable": True, "exit_code": 0},
            {"passed": False, "executable": False, "exit_code": -1},
        ]
        metrics = compute_pass_at_1(results)
        assert metrics["pass_at_1"] == 0.5
        assert metrics["executable_rate"] == 0.75


# ---------------------------------------------------------------------------
# Runner configuration tests
# ---------------------------------------------------------------------------


class TestRunnerConfig:
    """Tests for eval runner configuration and CLI parsing."""

    def test_metric_category_sets_are_disjoint(self):
        """The three metric routing sets must not overlap — a metric can only be
        dispatched to one handler."""
        from experiments.eval.eval_scripts.runner import (
            _AUTOMATIC_METRICS,
            _CODE_EXEC_METRICS,
            _JUDGE_METRICS,
        )

        assert _AUTOMATIC_METRICS.isdisjoint(_JUDGE_METRICS)
        assert _AUTOMATIC_METRICS.isdisjoint(_CODE_EXEC_METRICS)
        assert _JUDGE_METRICS.isdisjoint(_CODE_EXEC_METRICS)

    def test_run_eval_validates_metric(self):
        """run_eval raises ValueError for invalid task/metric combination."""
        from experiments.eval.eval_scripts.runner import run_eval

        with pytest.raises(ValueError, match="not valid for task"):
            run_eval(
                task="chat",
                dataset_name="hotpotqa",
                metric="pass_at_1",  # not valid for chat
                rag_aliases=["none"],
                lora_aliases=["none"],
            )

    def test_dataset_local_mapping_covers_all_suites(self):
        """All datasets used in _SUITE_KB have a local mapping."""
        from experiments.eval.eval_scripts.runner import _DATASET_LOCAL, _SUITE_KB

        for (_task, dataset_name), _kb in _SUITE_KB.items():
            assert dataset_name in _DATASET_LOCAL, (
                f"Dataset '{dataset_name}' in _SUITE_KB but not in _DATASET_LOCAL"
            )

    def test_load_dataset_samples_unknown_returns_empty(self):
        """_load_dataset_samples returns [] for an unknown dataset."""
        from experiments.eval.eval_scripts.runner import _load_dataset_samples

        assert _load_dataset_samples("chat", "nonexistent_dataset") == []

    def test_load_dataset_samples_missing_dir_returns_empty(self):
        """_load_dataset_samples returns [] when dataset dir does not exist."""
        from experiments.eval.eval_scripts.runner import _load_dataset_samples

        # hotpotqa is valid but its directory won't exist in test env
        result = _load_dataset_samples("chat", "hotpotqa")
        assert result == []

    def test_build_common_fields(self):
        from experiments.eval.eval_scripts.runner import _build_common_fields

        settings = types.SimpleNamespace(
            judge_backend="local_vllm",
            judge_model="judge-model",
            bert_score_model="deberta",
            temperature=0.0,
        )

        fields = _build_common_fields(
            task="chat",
            dataset_name="hotpotqa",
            base_model="Qwen/Qwen3-0.6B",
            lora_alias="none",
            lora_info={
                "adapter_name": None,
                "adapter_version": None,
                "adapter_mlflow_run_id": None,
            },
            rag_alias="none",
            rag_enabled=False,
            kb_name=None,
            eval_settings=settings,
            eval_context=None,
            now=datetime.now(timezone.utc),
        )

        assert fields["task"] == "chat"
        assert fields["dataset_name"] == "hotpotqa"
        assert fields["base_model"] == "Qwen/Qwen3-0.6B"
        assert fields["lora_alias"] == "none"
        assert fields["rag_enabled"] is False
        assert fields["rag_alias"] is None
        assert fields["knowledge_base"] is None
        assert fields["judge_backend"] == "local_vllm"
        assert fields["judge_model"] == "judge-model"
        assert fields["status"] == "running"
        assert isinstance(fields["id"], uuid.UUID)

    def test_build_common_fields_with_rag(self):
        from experiments.eval.eval_scripts.runner import _build_common_fields

        settings = types.SimpleNamespace(
            judge_backend="openai_compatible",
            judge_model="judge-model",
            judge_base_url="https://judge.example",
            bert_score_model="deberta",
            temperature=0.0,
        )

        fields = _build_common_fields(
            task="chat",
            dataset_name="hotpotqa",
            base_model="Qwen/Qwen3-0.6B",
            lora_alias="champion",
            lora_info={
                "adapter_name": "lora-chat",
                "adapter_version": 3,
                "adapter_mlflow_run_id": "run123",
            },
            rag_alias="champion",
            rag_enabled=True,
            kb_name="arxiv",
            eval_settings=settings,
            eval_context=None,
            now=datetime.now(timezone.utc),
        )

        assert fields["rag_enabled"] is True
        assert fields["rag_alias"] == "champion"
        assert fields["knowledge_base"] == "arxiv"
        assert fields["adapter_name"] == "lora-chat"
        assert fields["adapter_version"] == 3
        assert fields["lora_alias"] == "champion"
        assert fields["judge_backend"] == "openai_compatible"
        assert fields["judge_model"] == "judge-model"

    def test_calculate_metrics_applies_eval_context_overrides(self):
        from experiments.eval.eval_scripts.runner import calculate_metrics

        prediction_data = {
            "task": "summarize",
            "dataset_name": "arxiv_summarization",
            "kb_name": None,
            "base_model": "base-model",
            "eval_context": {
                "temperature": 0.25,
                "max_tokens": 128,
                "extra": {"evaluation_backend": "local_peft_generation"},
            },
            "bundles": [
                {
                    "rag_alias": "none",
                    "lora_alias": "local",
                    "lora_info": {
                        "adapter_name": "lora-summarize-local",
                        "adapter_version": None,
                        "adapter_mlflow_run_id": "run-123",
                    },
                    "rag_enabled": False,
                    "predictions": ["summary"],
                    "references": ["summary"],
                    "judge_samples": [
                        {
                            "question": "article",
                            "answer": "summary",
                            "reference": "summary",
                            "context": "",
                        }
                    ],
                    "sample_details": [
                        {
                            "sample_idx": 0,
                            "input": "article",
                            "output": "summary",
                            "reference": "summary",
                            "detail": {},
                        }
                    ],
                }
            ],
        }

        with patch("experiments.eval.eval_scripts.runner._log_to_db") as mock_log_to_db:
            rows = calculate_metrics(metric="rouge_l", prediction_data=prediction_data)

        assert len(rows) == 1
        assert rows[0]["adapter_mlflow_run_id"] == "run-123"
        assert rows[0]["temperature"] == 0.25
        assert rows[0]["max_tokens"] == 128
        assert rows[0]["extra"] == {"evaluation_backend": "local_peft_generation"}
        mock_log_to_db.assert_called_once()


class TestRunnerProgress:
    def test_log_fetch_progress_logs_only_boundaries_and_stride(self):
        from experiments.eval.eval_scripts.runner import _log_fetch_progress

        with patch("experiments.eval.eval_scripts.runner.logger.info") as mock_info:
            _log_fetch_progress(
                phase="generation",
                task="chat",
                dataset_name="hotpotqa",
                rag_alias="none",
                lora_alias="none",
                completed=0,
                total=10,
                every=5,
            )
            _log_fetch_progress(
                phase="generation",
                task="chat",
                dataset_name="hotpotqa",
                rag_alias="none",
                lora_alias="none",
                completed=1,
                total=10,
                every=5,
            )
            _log_fetch_progress(
                phase="generation",
                task="chat",
                dataset_name="hotpotqa",
                rag_alias="none",
                lora_alias="none",
                completed=5,
                total=10,
                every=5,
            )
            _log_fetch_progress(
                phase="generation",
                task="chat",
                dataset_name="hotpotqa",
                rag_alias="none",
                lora_alias="none",
                completed=10,
                total=10,
                every=5,
                gateway_failures=2,
            )

        assert mock_info.call_count == 3
        assert mock_info.call_args_list[-1].args[-1] == " gateway_failures=2"

    def test_fetch_generation_predictions_emits_progress_updates(self):
        from experiments.eval.eval_scripts.runner import _fetch_generation_predictions

        eval_settings = types.SimpleNamespace(
            gateway_url="http://gateway:9001",
            temperature=0.0,
            internal_api_key="",
            max_completion_tokens=256,
        )

        with (
            patch(
                "experiments.eval.eval_scripts.runner._resolve_lora_alias",
                return_value={
                    "adapter_name": None,
                    "adapter_version": None,
                    "adapter_mlflow_run_id": None,
                },
            ),
            patch(
                "experiments.eval.eval_scripts.runner._load_dataset_samples",
                return_value=[
                    {"question": "q1", "answer": "a1", "id": "s1"},
                    {"question": "q2", "answer": "a2", "id": "s2"},
                ],
            ),
            patch(
                "experiments.eval.eval_scripts.runner._call_gateway",
                side_effect=[
                    {"choices": [{"message": {"content": "pred-1"}}]},
                    RuntimeError("timed out"),
                ],
            ) as mock_call_gateway,
            patch("experiments.eval.eval_scripts.runner._log_fetch_progress") as mock_progress,
        ):
            bundle = _fetch_generation_predictions(
                task="chat",
                dataset_name="hotpotqa",
                rag_alias="none",
                lora_alias="none",
                kb_name=None,
                eval_settings=eval_settings,
            )

        assert bundle["predictions"] == ["pred-1", ""]
        assert [call.kwargs["completed"] for call in mock_progress.call_args_list] == [0, 1, 2]
        assert mock_progress.call_args_list[-1].kwargs["gateway_failures"] == 1
        assert all(
            call.kwargs["max_completion_tokens"] == 256 for call in mock_call_gateway.call_args_list
        )


class _FakeStreamResponse:
    def __init__(self, *, lines, headers=None):
        self._lines = list(lines)
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        yield from self._lines


class _FailingStreamResponse(_FakeStreamResponse):
    def __init__(self, *, status_code: int, url: str):
        super().__init__(lines=[], headers={})
        self._request = httpx.Request("POST", url)
        self._response = httpx.Response(status_code, request=self._request)

    def raise_for_status(self) -> None:
        raise httpx.HTTPStatusError(
            f"Server error '{self._response.status_code}' for url '{self._request.url}'",
            request=self._request,
            response=self._response,
        )


class _FakeStreamContext:
    def __init__(self, response: _FakeStreamResponse):
        self._response = response

    def __enter__(self) -> _FakeStreamResponse:
        return self._response

    def __exit__(self, exc_type, exc, tb) -> bool:
        del exc_type, exc, tb
        return False


class _FakeJSONResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def _sse_data(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}"


class TestRunnerGatewayTransport:
    def test_call_gateway_uses_unbounded_read_timeout(self):
        from experiments.eval.eval_scripts.runner import _call_gateway

        stream_response = _FakeStreamResponse(headers={}, lines=["data: [DONE]", ""])

        with (
            patch(
                "experiments.eval.eval_scripts.runner.httpx.stream",
                return_value=_FakeStreamContext(stream_response),
            ) as mock_stream,
            patch(
                "experiments.eval.eval_scripts.runner.httpx.get",
                return_value=_FakeJSONResponse({}),
            ),
        ):
            _call_gateway(
                messages=[{"role": "user", "content": "hello"}],
                gateway_url="http://gateway:9000",
                temperature=0.0,
                internal_api_key="",
                max_completion_tokens=512,
            )

        timeout = mock_stream.call_args.kwargs["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.read is None
        assert timeout.connect == 30.0
        assert mock_stream.call_args.kwargs["json"]["max_completion_tokens"] == 512

    def test_call_gateway_propagates_gateway_http_failure_for_rag_requests(self):
        from experiments.eval.eval_scripts.runner import _call_gateway

        stream_response = _FailingStreamResponse(
            status_code=500,
            url="http://gateway:9000/v1/chat/completions",
        )

        with patch(
            "experiments.eval.eval_scripts.runner.httpx.stream",
            return_value=_FakeStreamContext(stream_response),
        ):
            with pytest.raises(httpx.HTTPStatusError, match="500"):
                _call_gateway(
                    messages=[{"role": "user", "content": "hello"}],
                    gateway_url="http://gateway:9000",
                    rag_sources=[{"knowledge_base": "arxiv", "alias": "champion"}],
                    temperature=0.0,
                    internal_api_key="secret",
                    expect_rag_context=True,
                )

    def test_call_gateway_reconstructs_chat_response_from_standard_sse(self):
        from experiments.eval.eval_scripts.runner import _call_gateway

        stream_response = _FakeStreamResponse(
            headers={"X-Request-Id": "req-123"},
            lines=[
                _sse_data(
                    {
                        "id": "chatcmpl-req-123",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {"index": 0, "delta": {"content": "hello"}, "finish_reason": None}
                        ],
                    }
                ),
                "",
                _sse_data(
                    {
                        "id": "chatcmpl-req-123",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {"index": 0, "delta": {"content": " world"}, "finish_reason": None}
                        ],
                    }
                ),
                "",
                _sse_data(
                    {
                        "id": "chatcmpl-req-123",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                ),
                "",
                _sse_data(
                    {
                        "id": "chatcmpl-req-123",
                        "object": "chat.completion.chunk",
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 11,
                            "completion_tokens": 5,
                            "total_tokens": 16,
                        },
                    }
                ),
                "",
                "data: [DONE]",
                "",
            ],
        )
        prompt_preview = _FakeJSONResponse(
            {
                "request_id": "req-123",
                "prompt_messages": [{"role": "system", "content": "prompt"}],
                "rag_context": [{"content": "doc-1"}],
            }
        )

        with (
            patch(
                "experiments.eval.eval_scripts.runner.httpx.stream",
                return_value=_FakeStreamContext(stream_response),
            ) as mock_stream,
            patch(
                "experiments.eval.eval_scripts.runner.httpx.get",
                return_value=prompt_preview,
            ) as mock_get,
        ):
            response = _call_gateway(
                messages=[{"role": "user", "content": "hello"}],
                gateway_url="http://gateway:9000",
                rag_sources=[{"knowledge_base": "arxiv", "alias": "champion"}],
                temperature=0.0,
                internal_api_key="secret",
                expect_rag_context=True,
            )

        assert response["id"] == "chatcmpl-req-123"
        assert response["choices"][0]["message"]["content"] == "hello world"
        assert response["choices"][0]["finish_reason"] == "stop"
        assert response["usage"] == {
            "prompt_tokens": 11,
            "completion_tokens": 5,
            "total_tokens": 16,
        }
        assert response["rag_context"] == [{"content": "doc-1"}]
        assert response["_prompt_messages"] == [{"role": "system", "content": "prompt"}]
        mock_stream.assert_called_once()
        timeout = mock_stream.call_args.kwargs["timeout"]
        assert isinstance(timeout, httpx.Timeout)
        assert timeout.read is None
        mock_get.assert_called_once_with(
            "http://gateway:9000/v1/chat/prompt-preview/req-123",
            headers={"X-API-Key": "secret"},
            timeout=30,
        )

    def test_call_gateway_requires_prompt_preview_for_rag_requests(self):
        from experiments.eval.eval_scripts.runner import _call_gateway

        stream_response = _FakeStreamResponse(
            headers={"X-Request-Id": "req-123"},
            lines=[
                _sse_data(
                    {
                        "id": "chatcmpl-req-123",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {"index": 0, "delta": {"content": "hello"}, "finish_reason": None}
                        ],
                    }
                ),
                "",
                _sse_data(
                    {
                        "id": "chatcmpl-req-123",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                ),
                "",
                "data: [DONE]",
                "",
            ],
        )

        with (
            patch(
                "experiments.eval.eval_scripts.runner.httpx.stream",
                return_value=_FakeStreamContext(stream_response),
            ),
            patch(
                "experiments.eval.eval_scripts.runner.httpx.get",
                return_value=_FakeJSONResponse({"request_id": "req-123"}),
            ),
        ):
            with pytest.raises(RuntimeError, match="rag_context"):
                _call_gateway(
                    messages=[{"role": "user", "content": "hello"}],
                    gateway_url="http://gateway:9000",
                    rag_sources=[{"knowledge_base": "arxiv", "alias": "champion"}],
                    temperature=0.0,
                    internal_api_key="secret",
                    expect_rag_context=True,
                )


class TestFetchPredictionsRagSelection:
    def test_fetch_predictions_preserves_auto_mode_for_generation_tasks(self):
        from experiments.eval.eval_scripts.runner import fetch_predictions

        eval_settings = types.SimpleNamespace(
            metrics=types.SimpleNamespace(
                temperature=0.0,
                bert_score_model="bert-model",
                max_completion_tokens=256,
            ),
            judge_backend="local_vllm",
            judge_model="judge-model",
            vllm_base_url="http://localhost:8000",
        )
        settings = types.SimpleNamespace(
            platform=types.SimpleNamespace(vllm_base_url="http://localhost:8000"),
            vllm=types.SimpleNamespace(model="base-model"),
            eval=eval_settings,
        )

        with (
            patch("experiments.eval.eval_scripts.runner.get_settings", return_value=settings),
            patch(
                "experiments.eval.eval_scripts.runner._fetch_generation_predictions",
                return_value={
                    "rag_alias": "champion",
                    "lora_alias": "none",
                    "lora_info": {
                        "adapter_name": None,
                        "adapter_version": None,
                        "adapter_mlflow_run_id": None,
                    },
                    "rag_enabled": False,
                    "predictions": ["answer"],
                    "references": ["reference"],
                    "judge_samples": [],
                    "sample_details": [],
                },
            ) as mock_fetch_generation,
        ):
            prediction_data = fetch_predictions(
                task="chat",
                dataset_name="hotpotqa",
                kb_name="arxiv",
                use_auto_rag=True,
                rag_aliases=["champion"],
                lora_aliases=["none"],
            )

        assert prediction_data["kb_name"] is None
        assert prediction_data["judge_backend"] == "local_vllm"
        assert prediction_data["judge_model"] == "judge-model"
        assert mock_fetch_generation.call_args.kwargs["kb_name"] is None

    def test_fetch_predictions_keeps_suite_default_when_auto_mode_is_off(self):
        from experiments.eval.eval_scripts.runner import fetch_predictions

        eval_settings = types.SimpleNamespace(
            metrics=types.SimpleNamespace(
                temperature=0.0,
                bert_score_model="bert-model",
                max_completion_tokens=256,
            ),
            judge_backend="local_vllm",
            judge_model="judge-model",
            vllm_base_url="http://localhost:8000",
        )
        settings = types.SimpleNamespace(
            platform=types.SimpleNamespace(vllm_base_url="http://localhost:8000"),
            vllm=types.SimpleNamespace(model="base-model"),
            eval=eval_settings,
        )

        with (
            patch("experiments.eval.eval_scripts.runner.get_settings", return_value=settings),
            patch(
                "experiments.eval.eval_scripts.runner._fetch_generation_predictions",
                return_value={
                    "rag_alias": "champion",
                    "lora_alias": "none",
                    "lora_info": {
                        "adapter_name": None,
                        "adapter_version": None,
                        "adapter_mlflow_run_id": None,
                    },
                    "rag_enabled": True,
                    "predictions": ["answer"],
                    "references": ["reference"],
                    "judge_samples": [],
                    "sample_details": [],
                },
            ) as mock_fetch_generation,
        ):
            prediction_data = fetch_predictions(
                task="chat",
                dataset_name="hotpotqa",
                rag_aliases=["champion"],
                lora_aliases=["none"],
            )

        assert prediction_data["kb_name"] is None
        assert prediction_data["judge_backend"] == "local_vllm"
        assert prediction_data["judge_model"] == "judge-model"
        assert mock_fetch_generation.call_args.kwargs["kb_name"] is None

    def test_fetch_predictions_rejects_auto_mode_for_retrieval(self):
        from experiments.eval.eval_scripts.runner import fetch_predictions

        with pytest.raises(ValueError, match="does not support use_auto_rag"):
            fetch_predictions(
                task="retrieval",
                dataset_name="beir_scifact",
                use_auto_rag=True,
                kb_name="arxiv",
                rag_aliases=["champion"],
                lora_aliases=["none"],
            )


# ---------------------------------------------------------------------------
# Migration SQL file tests
# ---------------------------------------------------------------------------


class TestMigrationSQL:
    """Tests that the migration SQL file exists and is well-formed."""

    def test_migration_file_exists(self):
        path = (
            Path(__file__).resolve().parent.parent.parent
            / "src"
            / "shared"
            / "db"
            / "eval_runs.sql"
        )
        assert path.exists()

    def test_migration_creates_table(self):
        path = (
            Path(__file__).resolve().parent.parent.parent
            / "src"
            / "shared"
            / "db"
            / "eval_runs.sql"
        )
        sql = path.read_text()
        assert "CREATE TABLE" in sql
        assert "eval_runs" in sql
        assert "gen_random_uuid()" in sql

    def test_migration_creates_indexes(self):
        path = (
            Path(__file__).resolve().parent.parent.parent
            / "src"
            / "shared"
            / "db"
            / "eval_runs.sql"
        )
        sql = path.read_text()
        assert "idx_eval_runs_task" in sql
        assert "idx_eval_runs_dataset" in sql
        assert "idx_eval_runs_rag_alias" in sql
        assert "idx_eval_runs_qdrant_alias" in sql
        assert "idx_eval_runs_rag_manifest" in sql
        assert "idx_eval_runs_verdict" in sql
        assert "idx_eval_runs_lora_alias" in sql
        assert "idx_eval_runs_extra" in sql

    def test_rag_observability_migration_adds_columns(self):
        path = (
            Path(__file__).resolve().parent.parent.parent
            / "src"
            / "shared"
            / "db"
            / "eval_runs_add_rag_observability_columns.sql"
        )
        sql = path.read_text()
        assert "ALTER TABLE eval_runs" in sql
        assert "qdrant_alias" in sql
        assert "rag_manifest_id" in sql
        assert "eval_verdict" in sql

    def test_chat_messages_usage_migration_exists(self):
        path = (
            Path(__file__).resolve().parent.parent.parent
            / "src"
            / "shared"
            / "db"
            / "chat_messages_add_usage_columns.sql"
        )
        assert path.exists()

    def test_chat_messages_usage_migration_adds_columns(self):
        path = (
            Path(__file__).resolve().parent.parent.parent
            / "src"
            / "shared"
            / "db"
            / "chat_messages_add_usage_columns.sql"
        )
        sql = path.read_text()
        assert "ALTER TABLE chat_messages" in sql
        assert "prompt_tokens" in sql
        assert "completion_tokens" in sql


class TestEvalDvcTraceability:
    def test_dvc_pointer_hash_reads_md5(self, tmp_path: Path):
        from experiments.eval.eval_scripts.runner import _dvc_pointer_hash

        pointer = tmp_path / "dataset.dvc"
        pointer.write_text(
            "outs:\n- md5: abc123.dir\n  size: 10\n  path: dataset\n",
            encoding="utf-8",
        )

        assert _dvc_pointer_hash(pointer) == "abc123.dir"


class TestRetrievalEvalParity:
    def test_build_temp_collection_preserves_hybrid_sparse_leg(self):
        from experiments.eval.eval_scripts.retrieval_bench import (
            EvalBuildConfig,
            build_temp_collection,
        )
        from rag.domain import RetrievalCapability

        build_config = EvalBuildConfig(
            qdrant_alias="rag__arxiv__challenger",
            collection_name="rag__arxiv__20260609_120000",
            manifest_id="sha256:test",
            chunking_strategy="recursive",
            chunk_size=128,
            chunk_overlap=16,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            sparse_encoder="Qdrant/bm25",
            retrieval_capability=RetrievalCapability.HYBRID,
        )
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = ["chunk-1"]
        mock_emb = MagicMock()
        mock_emb.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        sparse_vectors = [SparseVector(indices=[0, 1], values=[0.4, 0.6])]
        mock_sparse = MagicMock()
        mock_sparse.encode_documents.return_value = sparse_vectors
        mock_vs = MagicMock()

        with (
            patch("rag.chunking.get_chunker", return_value=mock_chunker),
            patch("rag.embeddings.EmbeddingService", return_value=mock_emb),
            patch("rag.sparse_encoder.SparseEncoderService", return_value=mock_sparse),
            patch("rag.vector_store.QdrantVectorStore", return_value=mock_vs),
        ):
            build_temp_collection(
                kb_name="arxiv",
                dataset_name="beir_scifact",
                rag_alias="challenger",
                corpus=[{"doc_id": "doc-1", "text": "chunk me"}],
                build_config=build_config,
                qdrant_host="localhost",
                qdrant_port=6333,
                embeddings_url="http://embeddings:8100",
            )

        mock_vs.create_collection.assert_called_once_with(
            dimension=3,
            retrieval_capability="hybrid",
        )
        add_kwargs = mock_vs.add_documents.call_args.kwargs
        assert add_kwargs["embeddings"] == [[0.1, 0.2, 0.3]]
        assert add_kwargs["sparse_vectors"] == sparse_vectors
        mock_emb.close.assert_called_once()
        mock_sparse.close.assert_called_once()

    def test_fetch_retrieval_predictions_uses_alias_configured_retriever(self):
        from experiments.eval.eval_scripts.retrieval_bench import EvalBuildConfig
        from experiments.eval.eval_scripts.runner import _fetch_retrieval_predictions
        from rag.domain import RetrievalCapability

        build_config = EvalBuildConfig(
            qdrant_alias="rag__arxiv__challenger",
            collection_name="rag__arxiv__20260609_120000",
            manifest_id="sha256:test",
            chunking_strategy="recursive",
            chunk_size=128,
            chunk_overlap=16,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            sparse_encoder="Qdrant/bm25",
            retrieval_capability=RetrievalCapability.HYBRID,
        )
        alias_config = types.SimpleNamespace(
            top_k=5,
            score_threshold=0.01,
            reranker="cross-encoder/ms-marco-MiniLM-L-6-v2",
            retrieval_strategy="hybrid",
            reranker_multiplier=4,
        )
        mock_settings = types.SimpleNamespace(
            platform=types.SimpleNamespace(
                qdrant_host="localhost",
                qdrant_port=6333,
                embeddings_url="http://embeddings:8100",
            ),
            rag=types.SimpleNamespace(sparse_encoder_model="Qdrant/bm25"),
        )
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            Document(content="chunk-1", metadata={"source": "doc-1"}, score=0.9)
        ]
        mock_reranker = MagicMock()
        mock_emb = MagicMock()
        mock_sparse = MagicMock()

        with (
            patch("experiments.eval.eval_scripts.runner.get_settings", return_value=mock_settings),
            patch(
                "experiments.eval.eval_scripts.runner.get_kb_config",
                return_value=types.SimpleNamespace(aliases={"challenger": alias_config}),
            ),
            patch(
                "experiments.eval.eval_scripts.runner.read_build_config",
                return_value=build_config,
            ),
            patch(
                "experiments.eval.eval_scripts.runner._load_dataset_samples",
                return_value=[
                    {
                        "query": "what is hybrid retrieval",
                        "query_id": "q-1",
                        "relevance": {"doc-1": 1.0},
                        "relevant_docs": [{"doc_id": "doc-1", "text": "chunk-1"}],
                    }
                ],
            ),
            patch(
                "experiments.eval.eval_scripts.runner.build_temp_collection",
                return_value="eval_arxiv_tmp",
            ),
            patch("experiments.eval.eval_scripts.runner.EmbeddingService", return_value=mock_emb),
            patch(
                "experiments.eval.eval_scripts.runner.SparseEncoderService",
                return_value=mock_sparse,
            ),
            patch(
                "experiments.eval.eval_scripts.runner.get_reranker",
                return_value=mock_reranker,
            ),
            patch(
                "experiments.eval.eval_scripts.runner.QdrantVectorStore",
                return_value=MagicMock(),
            ),
            patch(
                "experiments.eval.eval_scripts.runner.Retriever",
                return_value=mock_retriever,
            ) as mock_retriever_cls,
            patch("experiments.eval.eval_scripts.runner.delete_temp_collection") as mock_delete,
            patch("experiments.eval.eval_scripts.runner._log_fetch_progress"),
        ):
            result = _fetch_retrieval_predictions(
                dataset_name="beir_scifact",
                rag_alias="challenger",
                kb_name="arxiv",
                eval_settings=types.SimpleNamespace(),
            )

        mock_retriever_cls.assert_called_once()
        retriever_kwargs = mock_retriever_cls.call_args.kwargs
        assert retriever_kwargs["reranker"] is mock_reranker
        assert retriever_kwargs["sparse_encoder_service"] is mock_sparse
        assert retriever_kwargs["reranker_multiplier"] == 4
        mock_retriever.retrieve.assert_called_once_with(
            query="what is hybrid retrieval",
            top_k=5,
            score_threshold=0.01,
            strategy="hybrid",
        )
        assert result["query_results"] == [
            {
                "retrieved_ids": ["doc-1"],
                "relevance": {"doc-1": 1.0},
            }
        ]
        assert result["retrieval_top_k"] == 5
        assert result["score_threshold"] == 0.01
        assert result["build_config"]["qdrant_alias"] == "rag__arxiv__challenger"
        assert result["build_config"]["collection_name"] == "rag__arxiv__20260609_120000"
        assert result["build_config"]["manifest_id"] == "sha256:test"
        assert result["rag_observability"]["qdrant_alias"] == "rag__arxiv__challenger"
        assert result["rag_observability"]["qdrant_collection"] == "rag__arxiv__20260609_120000"
        assert result["rag_observability"]["rag_manifest_id"] == "sha256:test"
        mock_delete.assert_called_once_with(
            "eval_arxiv_tmp",
            qdrant_host="localhost",
            qdrant_port=6333,
        )

    def test_calculate_metrics_uses_bundle_retrieval_top_k(self):
        from experiments.eval.eval_scripts.runner import calculate_metrics

        prediction_data = {
            "task": "retrieval",
            "dataset_name": "beir_scifact",
            "kb_name": "arxiv",
            "base_model": "base-model",
            "eval_context": {},
            "bundles": [
                {
                    "rag_alias": "challenger",
                    "lora_alias": "none",
                    "lora_info": {
                        "adapter_name": None,
                        "adapter_version": None,
                        "adapter_mlflow_run_id": None,
                    },
                    "rag_enabled": True,
                    "query_results": [
                        {
                            "retrieved_ids": ["doc-1", "doc-2", "doc-3"],
                            "relevance": {"doc-1": 1.0, "doc-2": 0.5},
                        }
                    ],
                    "sample_details": [],
                    "retrieval_top_k": 2,
                    "score_threshold": 0.01,
                    "build_config": {
                        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                        "chunking_strategy": "recursive",
                        "chunk_size": 128,
                        "chunk_overlap": 16,
                        "qdrant_alias": "rag__arxiv__challenger",
                        "collection_name": "rag__arxiv__20260609_120000",
                        "manifest_id": "sha256:test",
                    },
                    "rag_observability": {
                        "qdrant_alias": "rag__arxiv__challenger",
                        "qdrant_collection": "rag__arxiv__20260609_120000",
                        "rag_manifest_id": "sha256:test",
                    },
                    "temp_collection": "eval_arxiv_tmp",
                }
            ],
        }

        with patch("experiments.eval.eval_scripts.runner._log_to_db"):
            rows = calculate_metrics(metric="recall_at_k", prediction_data=prediction_data)

        assert len(rows) == 1
        assert rows[0]["metric_name"] == "recall_at_2"
        assert rows[0]["retrieval_top_k"] == 2
        assert rows[0]["score_threshold"] == 0.01
        assert rows[0]["qdrant_alias"] == "rag__arxiv__challenger"
        assert rows[0]["qdrant_collection"] == "rag__arxiv__20260609_120000"
        assert rows[0]["rag_manifest_id"] == "sha256:test"
        assert rows[0]["eval_verdict"] == "unscored"
        assert rows[0]["metric_value"] == pytest.approx(1.0)
