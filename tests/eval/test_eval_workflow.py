"""Tests for the evaluation workflow: DB model, metrics, runner, and settings."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure settings don't require live services
os.environ.setdefault("GATEWAY_RAG_ENABLED", "false")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_settings_caches():
    """Clear lru_cache on settings between tests."""
    import shared.config as cfg

    cfg.get_settings.cache_clear()
    cfg.get_eval_settings.cache_clear()
    cfg._KB_REGISTRY = None
    cfg.KNOWLEDGE_BASES._loaded = False
    cfg.KNOWLEDGE_BASES.clear()
    yield
    cfg.get_settings.cache_clear()
    cfg.get_eval_settings.cache_clear()
    cfg._KB_REGISTRY = None
    cfg.KNOWLEDGE_BASES._loaded = False
    cfg.KNOWLEDGE_BASES.clear()


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
            "qdrant_collection",
            "embedding_model",
            "chunking_strategy",
            "chunk_size",
            "chunk_overlap",
            "retrieval_top_k",
            "score_threshold",
            "qdrant_snapshot_id",
            "dataset_dvc_hash",
            "reranking_strategy",
            "judge_model",
            "bert_score_model",
            "temperature",
            "max_tokens",
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
        from shared.config import get_eval_settings

        s = get_eval_settings()
        assert s.judge_model == "gemini-2.0-flash"
        assert s.temperature == 0.0
        assert s.max_tokens == 512
        assert s.code_exec_timeout == 30
        assert s.code_exec_mem_limit == "512m"
        assert s.bert_score_model == "microsoft/deberta-base-mnli"

    def test_env_override(self, monkeypatch):
        from shared.config import EvalSettings

        monkeypatch.setenv("EVAL_JUDGE_MODEL", "gemini-1.5-pro")
        monkeypatch.setenv("EVAL_TEMPERATURE", "0.7")
        s = EvalSettings()
        assert s.judge_model == "gemini-1.5-pro"
        assert s.temperature == 0.7


# ---------------------------------------------------------------------------
# Automatic metrics tests
# ---------------------------------------------------------------------------


class TestAutomaticMetrics:
    """Tests for automatic evaluation metrics."""

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
    """Tests for LLM-as-Judge with mocked Gemini calls."""

    @patch("experiments.eval.eval_scripts.metrics.llm_judge._call_gemini")
    def test_judge_single_relevance(self, mock_gemini):
        from experiments.eval.eval_scripts.metrics.llm_judge import judge_single

        mock_gemini.return_value = {"score": 4, "reason": "mostly relevant"}

        result = judge_single(
            "relevance",
            question="What is ML?",
            answer="Machine learning is...",
            reference="ML is a subset of AI",
            api_key="test-key",
            model="gemini-2.0-flash",
        )
        assert result["score"] == 4
        assert "relevant" in result["reason"]
        mock_gemini.assert_called_once()

    @patch("experiments.eval.eval_scripts.metrics.llm_judge._call_gemini")
    def test_judge_batch(self, mock_gemini):
        from experiments.eval.eval_scripts.metrics.llm_judge import judge_batch

        mock_gemini.return_value = {"score": 3, "reason": "ok"}

        result = judge_batch(
            "correctness",
            samples=[
                {"question": "q1", "answer": "a1", "reference": "r1"},
                {"question": "q2", "answer": "a2", "reference": "r2"},
            ],
            api_key="test-key",
            model="gemini-2.0-flash",
        )
        assert "correctness" in result
        assert result["correctness"] == 3.0

    def test_judge_unknown_metric(self):
        from experiments.eval.eval_scripts.metrics.llm_judge import judge_single

        with pytest.raises(ValueError, match="Unknown judge metric"):
            judge_single(
                "nonexistent_metric",
                answer="test",
                api_key="key",
                model="gemini-2.0-flash",
            )


# ---------------------------------------------------------------------------
# Code execution tests (mocked)
# ---------------------------------------------------------------------------


class TestCodeExec:
    """Tests for sandboxed code execution metrics."""

    def test_pass_at_1_all_pass(self):
        from experiments.eval.eval_scripts.metrics.code_exec import compute_pass_at_1

        results = [
            {"passed": True, "exit_code": 0},
            {"passed": True, "exit_code": 0},
        ]
        metrics = compute_pass_at_1(results)
        assert metrics["pass_at_1"] == 1.0
        assert metrics["executable_rate"] == 1.0

    def test_pass_at_1_none_pass(self):
        from experiments.eval.eval_scripts.metrics.code_exec import compute_pass_at_1

        results = [
            {"passed": False, "exit_code": 1},
            {"passed": False, "exit_code": -1},
        ]
        metrics = compute_pass_at_1(results)
        assert metrics["pass_at_1"] == 0.0

    def test_pass_at_1_empty(self):
        from experiments.eval.eval_scripts.metrics.code_exec import compute_pass_at_1

        metrics = compute_pass_at_1([])
        assert metrics["pass_at_1"] == 0.0
        assert metrics["executable_rate"] == 0.0

    def test_pass_at_1_partial(self):
        from experiments.eval.eval_scripts.metrics.code_exec import compute_pass_at_1

        results = [
            {"passed": True, "exit_code": 0},
            {"passed": False, "exit_code": 1},
            {"passed": True, "exit_code": 0},
            {"passed": False, "exit_code": -1},
        ]
        metrics = compute_pass_at_1(results)
        assert metrics["pass_at_1"] == 0.5
        assert metrics["executable_rate"] == 0.5


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

        assert _load_dataset_samples("chat", "nonexistent_dataset", limit=10) == []

    def test_load_dataset_samples_missing_dir_returns_empty(self):
        """_load_dataset_samples returns [] when dataset dir does not exist."""
        from experiments.eval.eval_scripts.runner import _load_dataset_samples

        # hotpotqa is valid but its directory won't exist in test env
        result = _load_dataset_samples("chat", "hotpotqa", limit=10)
        assert result == []

    def test_build_common_fields(self):
        from experiments.eval.eval_scripts.runner import _build_common_fields

        settings = MagicMock()
        settings.judge_model = "gemini-2.0-flash"
        settings.bert_score_model = "deberta"
        settings.temperature = 0.0
        settings.max_tokens = 512

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
        assert fields["status"] == "running"
        assert isinstance(fields["id"], uuid.UUID)

    def test_build_common_fields_with_rag(self):
        from experiments.eval.eval_scripts.runner import _build_common_fields

        settings = MagicMock()
        settings.judge_model = "gemini-2.0-flash"
        settings.bert_score_model = "deberta"
        settings.temperature = 0.0
        settings.max_tokens = 512

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
        assert "idx_eval_runs_lora_alias" in sql
        assert "idx_eval_runs_extra" in sql
