"""Tests for the eval database models (EvalRun, EvalMetric, EvalExample).

These tests verify the SQLAlchemy ORM models can be instantiated and
their table metadata is correct.  They do NOT require a live database.
"""

from __future__ import annotations

import uuid

from shared.db.models import (
    Base,
    EvalExample,
    EvalMetric,
    EvalRun,
)


class TestEvalRunModel:
    def test_table_name(self):
        assert EvalRun.__tablename__ == "eval_runs"

    def test_instantiation(self):
        run_id = uuid.uuid4()
        run = EvalRun(
            id=run_id,
            status="running",
            tier="regression",
            task="chat",
            config={"base_model": "Qwen/Qwen3-0.6B"},
            base_model="Qwen/Qwen3-0.6B",
            dataset_name="hotpotqa",
            dataset_split="validation",
        )
        assert run.id == run_id
        assert run.status == "running"
        assert run.tier == "regression"
        assert run.task == "chat"
        assert run.config["base_model"] == "Qwen/Qwen3-0.6B"

    def test_nullable_fields(self):
        run = EvalRun(
            status="running",
            tier="full",
            task="summarize",
            config={},
            base_model="Qwen/Qwen3-0.6B",
            dataset_name="arxiv-summarization",
            dataset_split="validation",
        )
        assert run.adapter_name is None
        assert run.adapter_version is None
        assert run.knowledge_base is None
        assert run.error_message is None
        assert run.finished_at is None


class TestEvalMetricModel:
    def test_table_name(self):
        assert EvalMetric.__tablename__ == "eval_metrics"

    def test_instantiation(self):
        run_id = uuid.uuid4()
        metric = EvalMetric(
            run_id=run_id,
            metric_name="rouge_l_mean",
            value=0.42,
        )
        assert metric.run_id == run_id
        assert metric.metric_name == "rouge_l_mean"
        assert metric.value == 0.42


class TestEvalExampleModel:
    def test_table_name(self):
        assert EvalExample.__tablename__ == "eval_examples"

    def test_instantiation(self):
        run_id = uuid.uuid4()
        ex = EvalExample(
            run_id=run_id,
            example_index=0,
            input_text="What is ML?",
            generated_text="Machine learning is ...",
            rouge_l=0.55,
        )
        assert ex.run_id == run_id
        assert ex.example_index == 0
        assert ex.rouge_l == 0.55
        assert ex.reference_text is None
        assert ex.executable is None

    def test_code_fields(self):
        ex = EvalExample(
            run_id=uuid.uuid4(),
            example_index=1,
            input_text="def add(a, b):",
            generated_text="def add(a, b): return a + b",
            executable=True,
            tests_passed=True,
        )
        assert ex.executable is True
        assert ex.tests_passed is True


class TestBaseMetadata:
    """Verify that eval tables are registered in Base.metadata."""

    def test_eval_tables_in_metadata(self):
        table_names = set(Base.metadata.tables.keys())
        assert "eval_runs" in table_names
        assert "eval_metrics" in table_names
        assert "eval_examples" in table_names

    def test_existing_tables_still_present(self):
        table_names = set(Base.metadata.tables.keys())
        assert "users" in table_names
        assert "chat_sessions" in table_names
        assert "chat_messages" in table_names
