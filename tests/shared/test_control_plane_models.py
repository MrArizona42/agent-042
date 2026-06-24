"""Tests for the control-plane ORM models in `shared.db.models`.

These compile DDL against the postgresql dialect without a live database
connection (this repo has no Postgres available in its test environment;
migration correctness is otherwise covered by the static SQL content tests
and, for deployment, `bootstrap/apply_agent042_db_migrations.sh`).
"""

from __future__ import annotations

from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable


def _compiled(element) -> str:
    return str(element.compile(dialect=postgresql.dialect()))


class TestRagReleaseBuildRow:
    def test_table_compiles(self):
        from shared.db.models import RagReleaseBuildRow

        ddl = _compiled(CreateTable(RagReleaseBuildRow.__table__))
        assert "CREATE TABLE rag_release_builds" in ddl
        assert "details JSONB NOT NULL" in ddl


class TestRagReleaseRow:
    def test_table_compiles_with_unique_constraints(self):
        from shared.db.models import RagReleaseRow

        ddl = _compiled(CreateTable(RagReleaseRow.__table__))
        assert "CREATE TABLE rag_releases" in ddl
        assert "UNIQUE (collection_name)" in ddl
        assert "UNIQUE (manifest_id)" in ddl
        assert "UNIQUE (release_fingerprint)" in ddl

    def test_table_has_no_alias_column(self):
        from shared.db.models import RagReleaseRow

        assert "alias" not in RagReleaseRow.__table__.columns.keys()


class TestRagAliasDeploymentRow:
    def test_table_compiles_with_release_foreign_key(self):
        from shared.db.models import RagAliasDeploymentRow

        ddl = _compiled(CreateTable(RagAliasDeploymentRow.__table__))
        assert "FOREIGN KEY(release_id) REFERENCES rag_releases (id)" in ddl

    def test_partial_unique_index_on_active_status(self):
        from shared.db.models import RagAliasDeploymentRow

        indexes = {idx.name: idx for idx in RagAliasDeploymentRow.__table__.indexes}
        active_index = indexes["uq_rag_alias_deployments_active"]
        assert active_index.unique is True
        ddl = _compiled(CreateIndex(active_index))
        assert "WHERE status = 'active'" in ddl
        assert "(kb_id, alias)" in ddl


class TestEvalRunReleaseColumns:
    def test_eval_run_has_release_identity_columns(self):
        from shared.db.models import EvalRun

        columns = EvalRun.__table__.columns.keys()
        for column in (
            "benchmark_execution_id",
            "rag_release_id",
            "alias_deployment_id",
            "build_config_digest",
            "retrieval_config_digest",
        ):
            assert column in columns

    def test_eval_run_compiles_with_release_foreign_keys(self):
        from shared.db.models import EvalRun

        ddl = _compiled(CreateTable(EvalRun.__table__))
        assert "FOREIGN KEY(rag_release_id) REFERENCES rag_releases (id)" in ddl
        assert "FOREIGN KEY(alias_deployment_id) REFERENCES rag_alias_deployments (id)" in ddl
