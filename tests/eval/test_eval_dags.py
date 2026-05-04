from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _reset_kb_registry():
    import shared.config as cfg

    cfg._KB_REGISTRY = None
    cfg._KB_INDEX = None
    yield
    cfg._KB_REGISTRY = None
    cfg._KB_INDEX = None


@pytest.fixture()
def kb_json_file(tmp_path: Path) -> Path:
    data = [
        {
            "task": "chat",
            "label": "General knowledge",
            "routing_description": "General ML research discussion.",
            "adapter": {"name": "", "alias": "", "enabled": False},
            "knowledge_bases": [
                {
                    "name": "arxiv",
                    "default_alias": "champion",
                    "aliases": {
                        "champion": {
                            "top_k": 5,
                            "score_threshold": 0.35,
                            "reranker": None,
                            "retrieval_strategy": "dense",
                            "reranker_multiplier": 4,
                        }
                    },
                    "update_strategy": "incremental",
                    "label": "ArXiv papers",
                    "description": "ML papers",
                    "selection_description": "Research papers and literature-grounded answers.",
                }
            ],
        },
        {
            "task": "code",
            "label": "Coding assistance",
            "routing_description": "Programming help for ML systems.",
            "adapter": {"name": "", "alias": "", "enabled": False},
            "knowledge_bases": [
                {
                    "name": "pytorch_docs",
                    "default_alias": "champion",
                    "aliases": {
                        "champion": {
                            "top_k": 5,
                            "score_threshold": 0.35,
                            "reranker": None,
                            "retrieval_strategy": "dense",
                            "reranker_multiplier": 4,
                        }
                    },
                    "update_strategy": "replace",
                    "label": "PyTorch docs",
                    "description": "PyTorch documentation",
                    "selection_description": "PyTorch API reference and implementation guidance.",
                }
            ],
        },
    ]
    path = tmp_path / "knowledge_bases.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


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


def test_eval_dags_kb_options_use_shared_registry(
    monkeypatch: pytest.MonkeyPatch,
    kb_json_file: Path,
):
    import shared.config as cfg
    from shared.config import _load_knowledge_bases

    monkeypatch.setenv("PROJECT_ROOT", str(PROJECT_ROOT))
    _install_airflow_stubs(monkeypatch)
    cfg._KB_REGISTRY, cfg._KB_INDEX = _load_knowledge_bases(kb_json_file)

    sys.modules.pop("dags.eval_dags", None)
    eval_dags = importlib.import_module("dags.eval_dags")
    eval_dags = importlib.reload(eval_dags)

    assert eval_dags._list_knowledge_base_names() == ["arxiv", "pytorch_docs"]
    assert eval_dags._kb_options == ["arxiv", "pytorch_docs"]


def test_eval_dags_resolve_params_supports_auto_kb_mode(
    monkeypatch: pytest.MonkeyPatch,
    kb_json_file: Path,
):
    import shared.config as cfg
    from shared.config import _load_knowledge_bases

    monkeypatch.setenv("PROJECT_ROOT", str(PROJECT_ROOT))
    _install_airflow_stubs(monkeypatch)
    cfg._KB_REGISTRY, cfg._KB_INDEX = _load_knowledge_bases(kb_json_file)

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
    kb_json_file: Path,
):
    import shared.config as cfg
    from shared.config import _load_knowledge_bases

    monkeypatch.setenv("PROJECT_ROOT", str(PROJECT_ROOT))
    _install_airflow_stubs(monkeypatch)
    cfg._KB_REGISTRY, cfg._KB_INDEX = _load_knowledge_bases(kb_json_file)

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
