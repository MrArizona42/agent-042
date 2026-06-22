from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _install_airflow_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    airflow_module = types.ModuleType("airflow")
    models_module = types.ModuleType("airflow.models")
    param_module = types.ModuleType("airflow.models.param")
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

    class DummyParam:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class DummyPythonOperator:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __rshift__(self, other):
            return other

    setattr(airflow_module, "DAG", DummyDAG)
    setattr(param_module, "Param", DummyParam)
    setattr(python_module, "PythonOperator", DummyPythonOperator)

    monkeypatch.setitem(sys.modules, "airflow", airflow_module)
    monkeypatch.setitem(sys.modules, "airflow.models", models_module)
    monkeypatch.setitem(sys.modules, "airflow.models.param", param_module)
    monkeypatch.setitem(sys.modules, "airflow.operators", operators_module)
    monkeypatch.setitem(sys.modules, "airflow.operators.python", python_module)


def _load_dag(monkeypatch: pytest.MonkeyPatch, project_root: Path | None = None):
    monkeypatch.setenv("CONTAINER__PROJECT_ROOT", (project_root or Path.cwd()).as_posix())
    _install_airflow_stubs(monkeypatch)
    sys.modules.pop("dags.rag_alias_apply", None)
    return importlib.import_module("dags.rag_alias_apply")


def _context(**overrides):
    params = {
        "kb_id": "pytorch_reference",
        "alias": "challenger",
        "release_id": "",
        "refresh_sources": False,
        "sync_dvc": False,
        "dvc_base_branch": "develop",
        "dvc_bot_branch": "",
        "allow_unevaluated": False,
        "allow_build_default": False,
    }
    params.update(overrides)
    return {"params": params}


def test_dag_exposes_expected_params(monkeypatch: pytest.MonkeyPatch) -> None:
    dag_module = _load_dag(monkeypatch)

    params = dag_module.dag.kwargs["params"]

    assert set(params) >= {
        "kb_id",
        "alias",
        "release_id",
        "refresh_sources",
        "sync_dvc",
        "allow_unevaluated",
        "allow_build_default",
    }


def test_apply_alias_calls_alias_service_apply(monkeypatch: pytest.MonkeyPatch) -> None:
    dag_module = _load_dag(monkeypatch)

    calls = []

    class _FakeResult:
        @staticmethod
        def model_dump(mode: str) -> dict:
            return {"action": "no_drift"}

    class _FakeService:
        def apply(self, request):
            calls.append(request)
            return _FakeResult()

    import rag.cli.factories as factories

    monkeypatch.setattr(factories, "load_catalog_config", lambda ctx: object())
    monkeypatch.setattr(factories, "build_alias_service", lambda ctx, catalog_cfg: _FakeService())

    payload = dag_module._apply_alias(**_context(refresh_sources=True))

    assert payload == {"action": "no_drift"}
    assert len(calls) == 1
    assert calls[0].kb_id == "pytorch_reference"
    assert calls[0].alias == "challenger"
    assert calls[0].refresh_sources is True


def test_rag_context_resolves_relative_data_root_under_airflow_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dag_module = _load_dag(monkeypatch, project_root=tmp_path)

    import app_config.runtime as runtime_config

    monkeypatch.setattr(
        runtime_config,
        "get_settings",
        lambda: SimpleNamespace(
            catalog=SimpleNamespace(path=Path("catalog.toml")),
            rag=SimpleNamespace(data_root=Path("assets/rag_data")),
        ),
    )

    ctx = dag_module._rag_context()

    assert ctx.catalog_path == tmp_path / "catalog.toml"
    assert ctx.data_root == tmp_path / "assets/rag_data"


def test_rag_context_preserves_absolute_runtime_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dag_module = _load_dag(monkeypatch, project_root=tmp_path / "project")
    absolute_catalog = tmp_path / "config" / "catalog.toml"
    absolute_data = tmp_path / "shared" / "rag_data"

    import app_config.runtime as runtime_config

    monkeypatch.setattr(
        runtime_config,
        "get_settings",
        lambda: SimpleNamespace(
            catalog=SimpleNamespace(path=absolute_catalog),
            rag=SimpleNamespace(data_root=absolute_data),
        ),
    )

    ctx = dag_module._rag_context()

    assert ctx.catalog_path == absolute_catalog
    assert ctx.data_root == absolute_data


def test_apply_alias_requires_kb_id_and_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    dag_module = _load_dag(monkeypatch)

    with pytest.raises(ValueError, match="kb_id"):
        dag_module._apply_alias(**_context(kb_id=""))

    with pytest.raises(ValueError, match="alias"):
        dag_module._apply_alias(**_context(alias=""))


def test_sync_dvc_skips_when_not_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    dag_module = _load_dag(monkeypatch)

    result = dag_module._sync_dvc(**_context(sync_dvc=False))

    assert result == {"synced": False, "paths": []}


def test_sync_dvc_skips_when_no_artifacts_exist_yet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dag_module = _load_dag(monkeypatch)

    import rag.cli.factories as factories

    class _FakeIndex:
        @staticmethod
        def corpus_for_kb(kb_id: str):
            return []

    monkeypatch.setattr(factories, "load_catalog_config", lambda ctx: object())
    monkeypatch.setattr(
        "app_config.catalog.build_source_instance_index", lambda catalog_cfg: _FakeIndex()
    )
    monkeypatch.setattr(factories.RagContext, "data_root", property(lambda self: tmp_path))

    result = dag_module._sync_dvc(**_context(sync_dvc=True))

    assert result == {"synced": False, "paths": []}
