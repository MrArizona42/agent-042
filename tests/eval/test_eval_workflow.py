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
            "id", "created_at", "finished_at", "status",
            "task", "dataset_name", "metric_name", "metric_value",
            "base_model", "adapter_name", "adapter_version",
            "adapter_mlflow_run_id", "lora_alias",
            "rag_enabled", "rag_alias", "knowledge_base",
            "qdrant_collection", "embedding_model",
            "chunking_strategy", "chunk_size", "chunk_overlap",
            "retrieval_top_k", "score_threshold",
            "qdrant_snapshot_id", "dataset_dvc_hash", "reranking_strategy",
            "judge_model", "bert_score_model",
            "temperature", "max_tokens",
            "extra", "error_message",
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
        assert s.bert_score_model == "microsoft/deberta-xlarge-mnli"
        assert s.temperature == 0.0
        assert s.max_tokens == 512
        assert s.sample_limit == 0
        assert s.code_exec_timeout == 30
        assert s.code_exec_image == "python:3.11-slim"

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
        from experiments.scripts.eval.metrics.automatic import compute_rouge_l

        assert compute_rouge_l("hello world", "hello world") == 1.0

    def test_rouge_l_empty(self):
        from experiments.scripts.eval.metrics.automatic import compute_rouge_l

        assert compute_rouge_l("", "hello") == 0.0
        assert compute_rouge_l("hello", "") == 0.0

    def test_rouge_l_partial(self):
        from experiments.scripts.eval.metrics.automatic import compute_rouge_l

        score = compute_rouge_l("the cat sat on the mat", "the cat on the mat")
        assert 0.0 < score < 1.0

    def test_recall_at_k(self):
        from experiments.scripts.eval.metrics.automatic import compute_recall_at_k

        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "f"}
        recall = compute_recall_at_k(retrieved, relevant, k=5)
        assert recall == pytest.approx(2 / 3)

    def test_recall_at_k_empty(self):
        from experiments.scripts.eval.metrics.automatic import compute_recall_at_k

        assert compute_recall_at_k(["a", "b"], set(), k=5) == 0.0

    def test_ndcg_at_k_perfect(self):
        from experiments.scripts.eval.metrics.automatic import compute_ndcg_at_k

        retrieved = ["a", "b", "c"]
        labels = {"a": 3.0, "b": 2.0, "c": 1.0}
        ndcg = compute_ndcg_at_k(retrieved, labels, k=3)
        assert ndcg == pytest.approx(1.0)

    def test_ndcg_at_k_reversed(self):
        from experiments.scripts.eval.metrics.automatic import compute_ndcg_at_k

        retrieved = ["c", "b", "a"]
        labels = {"a": 3.0, "b": 2.0, "c": 1.0}
        ndcg = compute_ndcg_at_k(retrieved, labels, k=3)
        assert 0.0 < ndcg < 1.0


# ---------------------------------------------------------------------------
# LLM judge tests (mocked)
# ---------------------------------------------------------------------------


class TestLLMJudge:
    """Tests for LLM-as-Judge with mocked Gemini calls."""

    @patch("experiments.scripts.eval.metrics.llm_judge._call_gemini")
    def test_judge_single_relevance(self, mock_gemini):
        from experiments.scripts.eval.metrics.llm_judge import judge_single

        mock_gemini.return_value = {"score": 4, "reason": "mostly relevant"}

        result = judge_single(
            "relevance",
            question="What is ML?",
            answer="Machine learning is...",
            reference="ML is a subset of AI",
            api_key="test-key",
        )
        assert result["score"] == 4
        assert "relevant" in result["reason"]
        mock_gemini.assert_called_once()

    @patch("experiments.scripts.eval.metrics.llm_judge._call_gemini")
    def test_judge_batch(self, mock_gemini):
        from experiments.scripts.eval.metrics.llm_judge import judge_batch

        mock_gemini.return_value = {"score": 3, "reason": "ok"}

        result = judge_batch(
            "correctness",
            samples=[
                {"question": "q1", "answer": "a1", "reference": "r1"},
                {"question": "q2", "answer": "a2", "reference": "r2"},
            ],
            api_key="test-key",
        )
        assert "correctness" in result
        assert result["correctness"] == 3.0

    def test_judge_unknown_metric(self):
        from experiments.scripts.eval.metrics.llm_judge import judge_single

        with pytest.raises(ValueError, match="Unknown judge metric"):
            judge_single("nonexistent_metric", answer="test", api_key="key")


# ---------------------------------------------------------------------------
# Code execution tests (mocked)
# ---------------------------------------------------------------------------


class TestCodeExec:
    """Tests for sandboxed code execution metrics."""

    def test_pass_at_1_all_pass(self):
        from experiments.scripts.eval.metrics.code_exec import compute_pass_at_1

        results = [
            {"passed": True, "exit_code": 0},
            {"passed": True, "exit_code": 0},
        ]
        metrics = compute_pass_at_1(results)
        assert metrics["pass_at_1"] == 1.0
        assert metrics["executable_rate"] == 1.0

    def test_pass_at_1_none_pass(self):
        from experiments.scripts.eval.metrics.code_exec import compute_pass_at_1

        results = [
            {"passed": False, "exit_code": 1},
            {"passed": False, "exit_code": -1},
        ]
        metrics = compute_pass_at_1(results)
        assert metrics["pass_at_1"] == 0.0

    def test_pass_at_1_empty(self):
        from experiments.scripts.eval.metrics.code_exec import compute_pass_at_1

        metrics = compute_pass_at_1([])
        assert metrics["pass_at_1"] == 0.0
        assert metrics["executable_rate"] == 0.0

    def test_pass_at_1_partial(self):
        from experiments.scripts.eval.metrics.code_exec import compute_pass_at_1

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

    def test_suite_kb_mapping(self):
        from experiments.scripts.eval.runner import _SUITE_KB

        assert _SUITE_KB[("chat", "hotpotqa")] == "arxiv"
        assert _SUITE_KB[("chat", "nq")] == "arxiv"
        assert _SUITE_KB[("code", "humaneval")] == "pytorch_docs"
        assert _SUITE_KB[("summarize", "arxiv_summarization")] is None

    def test_suite_kb_missing_returns_none(self):
        from experiments.scripts.eval.runner import _SUITE_KB

        assert _SUITE_KB.get(("unknown_task", "unknown_dataset")) is None

    def test_task_metrics(self):
        from experiments.scripts.eval.runner import _TASK_METRICS

        assert "relevance" in _TASK_METRICS["chat"]
        assert "correctness" in _TASK_METRICS["chat"]
        assert "rouge_l" in _TASK_METRICS["chat"]
        assert "bertscore_f1" in _TASK_METRICS["chat"]
        assert "pass_at_1" in _TASK_METRICS["code"]
        assert "recall_at_10" in _TASK_METRICS["retrieval"]
        assert "ndcg_at_10" in _TASK_METRICS["retrieval"]

    def test_metric_category_sets(self):
        """Each metric belongs to exactly one category set."""
        from experiments.scripts.eval.runner import (
            _AUTOMATIC_METRICS,
            _CODE_EXEC_METRICS,
            _JUDGE_METRICS,
        )

        assert "rouge_l" in _AUTOMATIC_METRICS
        assert "bertscore_f1" in _AUTOMATIC_METRICS
        assert "recall_at_10" in _AUTOMATIC_METRICS
        assert "ndcg_at_10" in _AUTOMATIC_METRICS
        assert "relevance" in _JUDGE_METRICS
        assert "correctness" in _JUDGE_METRICS
        assert "groundedness" in _JUDGE_METRICS
        assert "pass_at_1" in _CODE_EXEC_METRICS
        assert "executable_rate" in _CODE_EXEC_METRICS
        # No overlap between automatic and judge
        assert _AUTOMATIC_METRICS.isdisjoint(_JUDGE_METRICS)
        assert _AUTOMATIC_METRICS.isdisjoint(_CODE_EXEC_METRICS)
        assert _JUDGE_METRICS.isdisjoint(_CODE_EXEC_METRICS)

    def test_run_eval_validates_metric(self):
        """run_eval raises ValueError for invalid task/metric combination."""
        from experiments.scripts.eval.runner import run_eval

        with pytest.raises(ValueError, match="not valid for task"):
            run_eval(
                task="chat",
                dataset_name="hotpotqa",
                metric="pass_at_1",  # not valid for chat
                rag_aliases=["none"],
                lora_aliases=["none"],
            )

    def test_build_common_fields(self):
        from experiments.scripts.eval.runner import _build_common_fields

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
        from experiments.scripts.eval.runner import _build_common_fields

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
            now=datetime.now(timezone.utc),
        )

        assert fields["rag_enabled"] is True
        assert fields["rag_alias"] == "champion"
        assert fields["knowledge_base"] == "arxiv"
        assert fields["adapter_name"] == "lora-chat"
        assert fields["adapter_version"] == 3
        assert fields["lora_alias"] == "champion"


# ---------------------------------------------------------------------------
# Gateway rag_context tests
# ---------------------------------------------------------------------------


class TestRagContextInResponse:
    """Test that RAGService exposes retrieve_documents and format_documents."""

    def test_rag_service_has_retrieve_documents(self):
        from gateway.services.rag_service import RAGService

        assert hasattr(RAGService, "retrieve_documents")
        assert callable(RAGService.retrieve_documents)

    def test_rag_service_has_format_documents(self):
        from gateway.services.rag_service import RAGService

        assert hasattr(RAGService, "format_documents")
        assert callable(RAGService.format_documents)

    def test_retrieve_context_delegates(self):
        """retrieve_context should still work (backwards compat)."""
        from gateway.services.rag_service import RAGService

        assert hasattr(RAGService, "retrieve_context")
        assert callable(RAGService.retrieve_context)


# ---------------------------------------------------------------------------
# Migration SQL file tests
# ---------------------------------------------------------------------------


class TestMigrationSQL:
    """Tests that the migration SQL file exists and is well-formed."""

    def test_migration_file_exists(self):
        path = Path(__file__).resolve().parent.parent.parent / "migrations" / "eval_runs.sql"
        assert path.exists()

    def test_migration_creates_table(self):
        path = Path(__file__).resolve().parent.parent.parent / "migrations" / "eval_runs.sql"
        sql = path.read_text()
        assert "CREATE TABLE" in sql
        assert "eval_runs" in sql
        assert "gen_random_uuid()" in sql

    def test_migration_creates_indexes(self):
        path = Path(__file__).resolve().parent.parent.parent / "migrations" / "eval_runs.sql"
        sql = path.read_text()
        assert "idx_eval_runs_task" in sql
        assert "idx_eval_runs_dataset" in sql
        assert "idx_eval_runs_rag_alias" in sql
        assert "idx_eval_runs_lora_alias" in sql
        assert "idx_eval_runs_extra" in sql
