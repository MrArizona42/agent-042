from __future__ import annotations

import importlib
import sys
import types
from datetime import datetime, timezone

import pytest


def _install_airflow_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    airflow_module = types.ModuleType("airflow")
    operators_module = types.ModuleType("airflow.operators")
    python_module = types.ModuleType("airflow.operators.python")

    class DummyDAG:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyPythonOperator:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    setattr(airflow_module, "DAG", DummyDAG)
    setattr(python_module, "PythonOperator", DummyPythonOperator)

    monkeypatch.setitem(sys.modules, "airflow", airflow_module)
    monkeypatch.setitem(sys.modules, "airflow.operators", operators_module)
    monkeypatch.setitem(sys.modules, "airflow.operators.python", python_module)


def _load_dag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NETWORK__QDRANT_HTTP__INTERNAL_HOST", "qdrant")
    monkeypatch.setenv("NETWORK__QDRANT_HTTP__INTERNAL_PORT", "7000")
    _install_airflow_stubs(monkeypatch)
    sys.modules.pop("dags.rag_collection_cleanup", None)
    return importlib.reload(importlib.import_module("dags.rag_collection_cleanup"))


def test_cleanup_dag_reads_network_qdrant_env(monkeypatch: pytest.MonkeyPatch):
    cleanup_dag = _load_dag(monkeypatch)

    assert cleanup_dag.QDRANT_HOST == "qdrant"
    assert cleanup_dag.QDRANT_PORT == 7000


def test_is_rag_managed_collection_accepts_both_naming_conventions(
    monkeypatch: pytest.MonkeyPatch,
):
    cleanup_dag = _load_dag(monkeypatch)

    assert cleanup_dag._is_rag_managed_collection("rag__pytorch_reference__0123456789abcdef")
    assert cleanup_dag._is_rag_managed_collection("rag__pytorch_reference__20260101_000000")
    assert not cleanup_dag._is_rag_managed_collection("chat_documents")
    assert not cleanup_dag._is_rag_managed_collection("eval__pytorch_reference__docs__x")


def test_active_and_pending_deployments_are_always_protected(monkeypatch: pytest.MonkeyPatch):
    cleanup_dag = _load_dag(monkeypatch)
    now = datetime.now(timezone.utc)
    deployments = [
        cleanup_dag._DeploymentRow(
            kb_id="kb", alias="champion", release_id="r1", status="active", order_key=now
        ),
        cleanup_dag._DeploymentRow(
            kb_id="kb", alias="challenger", release_id="r2", status="pending", order_key=now
        ),
    ]

    protected = cleanup_dag._protected_release_ids(deployments, retain_superseded=3)

    assert protected == {"r1", "r2"}


def test_only_newest_n_superseded_deployments_per_alias_are_retained(
    monkeypatch: pytest.MonkeyPatch,
):
    cleanup_dag = _load_dag(monkeypatch)
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    deployments = [
        cleanup_dag._DeploymentRow(
            kb_id="kb",
            alias="champion",
            release_id=f"r{i}",
            status="superseded",
            order_key=base.replace(hour=i),
        )
        for i in range(5)
    ]

    protected = cleanup_dag._protected_release_ids(deployments, retain_superseded=2)

    # r4 and r3 have the largest order_key (newest).
    assert protected == {"r4", "r3"}


def test_superseded_retention_is_scoped_per_kb_and_alias(monkeypatch: pytest.MonkeyPatch):
    cleanup_dag = _load_dag(monkeypatch)
    now = datetime.now(timezone.utc)
    deployments = [
        cleanup_dag._DeploymentRow(
            kb_id="kb_a", alias="champion", release_id="a1", status="superseded", order_key=now
        ),
        cleanup_dag._DeploymentRow(
            kb_id="kb_b", alias="champion", release_id="b1", status="superseded", order_key=now
        ),
    ]

    protected = cleanup_dag._protected_release_ids(deployments, retain_superseded=1)

    assert protected == {"a1", "b1"}


def test_retirable_release_ids_excludes_protected_and_already_retired(
    monkeypatch: pytest.MonkeyPatch,
):
    cleanup_dag = _load_dag(monkeypatch)
    releases = [
        cleanup_dag._ReleaseRow(id="r1", collection_name="rag__kb__1", retired_at=None),
        cleanup_dag._ReleaseRow(id="r2", collection_name="rag__kb__2", retired_at=None),
        cleanup_dag._ReleaseRow(
            id="r3",
            collection_name="rag__kb__3",
            retired_at=datetime.now(timezone.utc),
        ),
    ]

    retirable = cleanup_dag._retirable_release_ids(releases, protected_release_ids={"r1"})

    assert retirable == ["r2"]
