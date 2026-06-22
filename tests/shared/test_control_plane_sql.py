"""Tests that the control-plane migration SQL files exist and are well-formed.

Mirrors `tests/eval/test_eval_workflow.py::TestMigrationSQL` -- this repo
tests migration SQL by asserting on file content, not by executing it
against a live database.
"""

from __future__ import annotations

from pathlib import Path

_DB_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "shared" / "db"


def _read(name: str) -> str:
    path = _DB_DIR / name
    assert path.exists(), f"missing migration file: {path}"
    return path.read_text()


class TestReleaseBuildsMigration:
    def test_creates_table(self):
        sql = _read("rag_release_builds.sql")
        assert "CREATE TABLE IF NOT EXISTS rag_release_builds" in sql
        assert "gen_random_uuid()" in sql

    def test_has_expected_columns(self):
        sql = _read("rag_release_builds.sql")
        for column in (
            "kb_id",
            "requested_alias",
            "status",
            "catalog_digest",
            "build_config_digest",
            "retrieval_config_digest",
            "source_declaration_digest",
            "source_snapshot_id",
            "release_id",
            "collection_name",
            "started_at",
            "finished_at",
            "error",
            "details",
        ):
            assert column in sql

    def test_has_expected_indexes(self):
        sql = _read("rag_release_builds.sql")
        assert "idx_rag_release_builds_kb_started" in sql
        assert "idx_rag_release_builds_status" in sql
        assert "idx_rag_release_builds_release_id" in sql


class TestReleasesMigration:
    def test_creates_table(self):
        sql = _read("rag_releases.sql")
        assert "CREATE TABLE IF NOT EXISTS rag_releases" in sql

    def test_release_table_has_no_alias_column(self):
        # A release carries no alias field -- which alias serves it is
        # recorded in rag_alias_deployments, not here.
        sql = _read("rag_releases.sql")
        table_body = sql.split("CREATE TABLE IF NOT EXISTS rag_releases (")[1].split(");")[0]
        column_names = {line.strip().split()[0] for line in table_body.splitlines() if line.strip()}
        assert "alias" not in column_names

    def test_has_unique_constraints(self):
        sql = _read("rag_releases.sql")
        assert "collection_name" in sql and "UNIQUE" in sql
        assert "manifest_id" in sql and "NOT NULL UNIQUE" in sql
        assert "release_fingerprint" in sql

    def test_has_expected_indexes(self):
        sql = _read("rag_releases.sql")
        assert "idx_rag_releases_kb_created" in sql
        assert "idx_rag_releases_build_config_digest" in sql
        assert "idx_rag_releases_source_declaration_digest" in sql
        assert "idx_rag_releases_source_snapshot_id" in sql


class TestAliasDeploymentsMigration:
    def test_creates_table(self):
        sql = _read("rag_alias_deployments.sql")
        assert "CREATE TABLE IF NOT EXISTS rag_alias_deployments" in sql
        assert "REFERENCES rag_releases(id)" in sql

    def test_has_partial_unique_active_index(self):
        sql = _read("rag_alias_deployments.sql")
        assert "CREATE UNIQUE INDEX IF NOT EXISTS uq_rag_alias_deployments_active" in sql
        assert "WHERE status = 'active'" in sql

    def test_has_expected_indexes(self):
        sql = _read("rag_alias_deployments.sql")
        assert "idx_rag_alias_deployments_release_id" in sql
        assert "idx_rag_alias_deployments_kb_alias_created" in sql


class TestEvalRunsReleaseColumnsMigration:
    def test_is_idempotent_alter(self):
        sql = _read("eval_runs_add_release_columns.sql")
        assert "ALTER TABLE eval_runs" in sql
        assert "ADD COLUMN IF NOT EXISTS" in sql

    def test_has_expected_columns(self):
        sql = _read("eval_runs_add_release_columns.sql")
        for column in (
            "benchmark_execution_id",
            "rag_release_id",
            "alias_deployment_id",
            "build_config_digest",
            "retrieval_config_digest",
        ):
            assert column in sql

    def test_references_release_and_deployment_tables(self):
        sql = _read("eval_runs_add_release_columns.sql")
        assert "REFERENCES rag_releases(id)" in sql
        assert "REFERENCES rag_alias_deployments(id)" in sql
