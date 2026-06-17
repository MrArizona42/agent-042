from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from tests.catalog_samples import write_chat_and_code_catalog

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _reset_kb_catalog():
    import app_config.runtime as cfg

    cfg.clear_knowledge_base_caches()
    yield
    cfg.clear_knowledge_base_caches()


@pytest.fixture()
def catalog_file(tmp_path: Path) -> Path:
    return write_chat_and_code_catalog(tmp_path / "catalog.toml")


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


@pytest.fixture()
def loaded_kb_catalog(catalog_file: Path):
    from app_config.catalog import catalog_override, load_catalog

    catalog, index = load_catalog(catalog_file)
    with catalog_override(catalog, index=index):
        yield catalog


def test_eval_dags_kb_options_use_shared_registry(
    monkeypatch: pytest.MonkeyPatch,
    loaded_kb_catalog,
):
    monkeypatch.setenv("CONTAINER__PROJECT_ROOT", str(PROJECT_ROOT))
    _install_airflow_stubs(monkeypatch)

    sys.modules.pop("dags.eval_dags", None)
    eval_dags = importlib.import_module("dags.eval_dags")
    eval_dags = importlib.reload(eval_dags)

    assert eval_dags._list_knowledge_base_names() == ["ml_papers_core", "pytorch_reference"]
    assert eval_dags._kb_options == ["ml_papers_core", "pytorch_reference"]


def test_eval_dags_alias_options_use_runtime_adapter_registry(
    monkeypatch: pytest.MonkeyPatch,
    loaded_kb_catalog,
):
    monkeypatch.setenv("CONTAINER__PROJECT_ROOT", str(PROJECT_ROOT))
    monkeypatch.setenv("ADAPTER_REGISTRY__SYNC_ALIASES", "champion,shadow")
    _install_airflow_stubs(monkeypatch)

    sys.modules.pop("dags.eval_dags", None)
    eval_dags = importlib.import_module("dags.eval_dags")
    eval_dags = importlib.reload(eval_dags)

    assert eval_dags._sync_aliases == ["champion", "challenger"]
    assert eval_dags._lora_alias_options == ["none", "champion", "challenger"]
    assert eval_dags._kb_alias_options == ["none", "challenger", "champion"]


def test_eval_dags_resolve_params_supports_auto_kb_mode(
    monkeypatch: pytest.MonkeyPatch,
    loaded_kb_catalog,
):
    monkeypatch.setenv("CONTAINER__PROJECT_ROOT", str(PROJECT_ROOT))
    _install_airflow_stubs(monkeypatch)

    sys.modules.pop("dags.eval_dags", None)
    eval_dags = importlib.import_module("dags.eval_dags")
    eval_dags = importlib.reload(eval_dags)

    resolved = eval_dags._resolve_params(
        {
            "params": {
                "knowledge_base_mode": "auto",
                "knowledge_base": None,
                "knowledge_base_aliases": ["champion"],
                "metrics": ["relevance"],
                "lora_aliases": ["none"],
                "custom_params": "",
            }
        }
    )

    assert resolved["knowledge_base"] is None
    assert resolved["use_auto_rag"] is True


def test_eval_dags_resolve_params_requires_kb_for_explicit_mode(
    monkeypatch: pytest.MonkeyPatch,
    loaded_kb_catalog,
):
    monkeypatch.setenv("CONTAINER__PROJECT_ROOT", str(PROJECT_ROOT))
    _install_airflow_stubs(monkeypatch)

    sys.modules.pop("dags.eval_dags", None)
    eval_dags = importlib.import_module("dags.eval_dags")
    eval_dags = importlib.reload(eval_dags)

    with pytest.raises(ValueError, match="knowledge_base"):
        eval_dags._resolve_params(
            {
                "params": {
                    "knowledge_base_mode": "explicit",
                    "knowledge_base": None,
                    "knowledge_base_aliases": ["champion"],
                    "metrics": ["relevance"],
                    "lora_aliases": ["none"],
                    "custom_params": "",
                }
            }
        )


def test_generation_dag_params_expose_kb_mode_controls(
    monkeypatch: pytest.MonkeyPatch,
    loaded_kb_catalog,
):
    monkeypatch.setenv("CONTAINER__PROJECT_ROOT", str(PROJECT_ROOT))
    _install_airflow_stubs(monkeypatch)

    sys.modules.pop("dags.eval_dags", None)
    eval_dags = importlib.import_module("dags.eval_dags")
    eval_dags = importlib.reload(eval_dags)

    chat_params = eval_dags.eval_chat_hotpotqa.kwargs["params"]
    retrieval_params = eval_dags.eval_retrieval_beir_scifact.kwargs["params"]

    assert "knowledge_base_mode" in chat_params
    assert chat_params["knowledge_base_mode"].kwargs["enum"] == ["explicit", "auto"]
    assert "knowledge_base" in chat_params
    assert "lora_aliases" in chat_params
    assert chat_params["knowledge_base_aliases"].kwargs["examples"] == [
        "none",
        "challenger",
        "champion",
    ]

    assert "knowledge_base_mode" not in retrieval_params
    assert "knowledge_base" in retrieval_params
    assert "lora_aliases" not in retrieval_params


def test_fetch_predictions_task_forwards_use_auto_rag(
    monkeypatch: pytest.MonkeyPatch,
    loaded_kb_catalog,
):
    monkeypatch.setenv("CONTAINER__PROJECT_ROOT", str(PROJECT_ROOT))
    _install_airflow_stubs(monkeypatch)

    sys.modules.pop("dags.eval_dags", None)
    eval_dags = importlib.import_module("dags.eval_dags")
    eval_dags = importlib.reload(eval_dags)

    forwarded: dict[str, object] = {}
    fake_runner = types.ModuleType("experiments.eval.eval_scripts.runner")

    def _fake_fetch_predictions(**kwargs):
        forwarded.update(kwargs)
        return {"bundles": []}

    fake_runner.fetch_predictions = _fake_fetch_predictions
    monkeypatch.setitem(sys.modules, "experiments.eval.eval_scripts.runner", fake_runner)

    with patch("builtins.open", mock_open()), patch("json.dump"):
        output_path = eval_dags._fetch_predictions_task(
            eval_task="chat",
            dataset="hotpotqa",
            run_id="manual__001",
            params={
                "knowledge_base_mode": "auto",
                "knowledge_base": None,
                "knowledge_base_aliases": ["champion"],
                "metrics": ["relevance"],
                "lora_aliases": ["none"],
                "custom_params": "",
            },
        )

    assert forwarded["task"] == "chat"
    assert forwarded["dataset_name"] == "hotpotqa"
    assert forwarded["kb_name"] is None
    assert forwarded["use_auto_rag"] is True
    assert forwarded["rag_aliases"] == ["champion"]
    assert forwarded["lora_aliases"] == ["none"]
    assert output_path.endswith("eval_predictions_manual__001.json")
