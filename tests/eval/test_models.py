"""Tests for the evaluation database models (EvalRun, EvalMetric, EvalExample).

These tests verify the ORM model definitions without requiring a live
PostgreSQL instance — they inspect table metadata, columns, constraints,
and relationships only.
"""

from __future__ import annotations

from shared.db.models import Base, EvalExample, EvalMetric, EvalRun


class TestEvalRunModel:
    """Verify EvalRun table metadata and column definitions."""

    def test_tablename(self):
        assert EvalRun.__tablename__ == "eval_runs"

    def test_primary_key(self):
        pk_cols = [c.name for c in EvalRun.__table__.primary_key.columns]
        assert pk_cols == ["id"]

    def test_required_columns(self):
        """Non-nullable columns that must always be populated."""
        table = EvalRun.__table__
        required = {
            "id", "status", "tier", "task", "config",
            "base_model", "dataset_name", "dataset_split",
        }
        non_nullable = {
            c.name for c in table.columns
            if not c.nullable and c.name != "created_at"
        }
        assert required.issubset(non_nullable)

    def test_nullable_columns(self):
        """Optional columns that may be NULL."""
        table = EvalRun.__table__
        nullable = {c.name for c in table.columns if c.nullable}
        expected_nullable = {
            "finished_at", "adapter_name", "adapter_version",
            "knowledge_base", "error_message",
        }
        assert expected_nullable.issubset(nullable)

    def test_config_column_is_jsonb(self):
        from sqlalchemy.dialects.postgresql import JSONB

        col = EvalRun.__table__.c.config
        assert isinstance(col.type, JSONB)

    def test_relationships_defined(self):
        rels = {r.key for r in EvalRun.__mapper__.relationships}
        assert "metrics" in rels
        assert "examples" in rels

    def test_indexes_defined(self):
        index_names = {idx.name for idx in EvalRun.__table__.indexes}
        expected = {
            "idx_eval_runs_task",
            "idx_eval_runs_adapter",
            "idx_eval_runs_created",
            "idx_eval_runs_config",
        }
        assert expected.issubset(index_names)

    def test_default_status_column(self):
        """Status column has a default of 'running'."""
        col = EvalRun.__table__.c.status
        assert col.default is not None
        assert col.default.arg == "running"

    def test_id_column_has_uuid_default(self):
        """ID column has a callable default for UUID generation."""
        col = EvalRun.__table__.c.id
        assert col.default is not None
        assert callable(col.default.arg)


class TestEvalMetricModel:
    """Verify EvalMetric table metadata."""

    def test_tablename(self):
        assert EvalMetric.__tablename__ == "eval_metrics"

    def test_unique_constraint(self):
        constraints = [
            c
            for c in EvalMetric.__table__.constraints
            if hasattr(c, "columns")
            and {col.name for col in c.columns} == {"run_id", "metric_name"}
        ]
        assert len(constraints) == 1

    def test_foreign_key_to_eval_runs(self):
        fk_cols = [fk.target_fullname for fk in EvalMetric.__table__.foreign_keys]
        assert "eval_runs.id" in fk_cols

    def test_cascade_delete(self):
        fk = list(EvalMetric.__table__.foreign_keys)[0]
        assert fk.ondelete == "CASCADE"

    def test_index_on_run_id(self):
        index_names = {idx.name for idx in EvalMetric.__table__.indexes}
        assert "idx_eval_metrics_run" in index_names


class TestEvalExampleModel:
    """Verify EvalExample table metadata."""

    def test_tablename(self):
        assert EvalExample.__tablename__ == "eval_examples"

    def test_required_columns(self):
        table = EvalExample.__table__
        required = {"id", "run_id", "example_index", "input_text", "generated_text"}
        non_nullable = {c.name for c in table.columns if not c.nullable}
        assert required.issubset(non_nullable)

    def test_score_columns_are_nullable(self):
        table = EvalExample.__table__
        score_cols = {
            "relevance", "correctness", "faithfulness", "coverage",
            "rouge_l", "bert_score", "executable", "tests_passed",
            "execution_error", "retrieved_docs", "groundedness",
        }
        nullable = {c.name for c in table.columns if c.nullable}
        assert score_cols.issubset(nullable)

    def test_foreign_key_to_eval_runs(self):
        fk_cols = [fk.target_fullname for fk in EvalExample.__table__.foreign_keys]
        assert "eval_runs.id" in fk_cols

    def test_index_on_run_id(self):
        index_names = {idx.name for idx in EvalExample.__table__.indexes}
        assert "idx_eval_examples_run" in index_names


class TestBaseMetadata:
    """Verify that eval tables are registered with the shared Base."""

    def test_eval_tables_in_metadata(self):
        table_names = set(Base.metadata.tables.keys())
        assert "eval_runs" in table_names
        assert "eval_metrics" in table_names
        assert "eval_examples" in table_names

    def test_existing_tables_still_present(self):
        """Eval models must not break existing table definitions."""
        table_names = set(Base.metadata.tables.keys())
        assert "users" in table_names
        assert "chat_sessions" in table_names
        assert "chat_messages" in table_names
