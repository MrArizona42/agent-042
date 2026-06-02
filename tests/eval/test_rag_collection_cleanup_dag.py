from __future__ import annotations

import importlib
import sys
import types

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


def test_cleanup_dag_prefers_nested_platform_qdrant_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PLATFORM__QDRANT_HOST", "nested-qdrant")
    monkeypatch.setenv("PLATFORM__QDRANT_PORT", "7000")
    _install_airflow_stubs(monkeypatch)

    sys.modules.pop("dags.rag_collection_cleanup", None)
    cleanup_dag = importlib.import_module("dags.rag_collection_cleanup")
    cleanup_dag = importlib.reload(cleanup_dag)

    assert cleanup_dag.QDRANT_HOST == "nested-qdrant"
    assert cleanup_dag.QDRANT_PORT == 7000