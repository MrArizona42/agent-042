from __future__ import annotations

import importlib
import subprocess
import sys
import types
from pathlib import Path

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


def _load_dag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CONTAINER__PROJECT_ROOT", Path.cwd().as_posix())
    _install_airflow_stubs(monkeypatch)
    sys.modules.pop("dags.rag_lifecycle", None)
    return importlib.import_module("dags.rag_lifecycle")


def _context(**overrides):
    params = {
        "catalog": "catalog.toml",
        "kb": "pytorch_reference",
        "source": "docs",
        "alias_config": "challenger",
        "rag_data_root": "assets/rag_data",
        "document_ids": "",
        "limit": 0,
        "collection": "",
        "promote_alias": "",
        "sync_dvc": False,
        "dvc_artifacts": "",
        "dvc_base_branch": "develop",
        "dvc_bot_branch": "",
        "build_run_id": "",
        "dry_run": False,
        "force_fetch": False,
        "force_extract": False,
        "force_chunk": False,
        "force_recreate": False,
    }
    params.update(overrides)
    return {"params": params}


def test_rag_lifecycle_dag_exposes_generic_params(monkeypatch: pytest.MonkeyPatch) -> None:
    dag_module = _load_dag(monkeypatch)

    params = dag_module.dag.kwargs["params"]

    assert set(params) >= {
            "catalog",
            "kb",
            "source_instance",
            "alias_config",
        "rag_data_root",
        "collection",
        "promote_alias",
        "sync_dvc",
        "dvc_artifacts",
        "build_run_id",
        "dry_run",
    }


def test_build_source_task_constructs_source_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    dag_module = _load_dag(monkeypatch)
    calls: list[list[str]] = []

    def fake_run_cli(args: list[str]):
        calls.append(args)
        return {"ok": True}

    monkeypatch.setattr(dag_module, "_run_cli", fake_run_cli)
    monkeypatch.setattr(
        dag_module,
        "_resolve_build_source_instance_ids",
        lambda params: ["pytorch_reference.docs"],
    )

    result = dag_module._build_source(
        **_context(
            document_ids="torch.nn, torch.Tensor",
            limit=2,
            force_fetch=True,
            force_extract=True,
            force_chunk=True,
        )
    )

    assert result == {"ok": True}
    assert calls == [
            [
                "build-source",
                "--persist-build-run",
                "--catalog",
                "catalog.toml",
                "--source-instance",
                "pytorch_reference.docs",
            "--rag-data-root",
            "assets/rag_data",
            "--document-id",
            "torch.nn",
            "--document-id",
            "torch.Tensor",
            "--limit",
            "2",
            "--force-fetch",
            "--force-extract",
            "--force-chunk",
        ]
    ]


def test_build_source_task_accepts_multiple_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    dag_module = _load_dag(monkeypatch)
    calls: list[list[str]] = []

    monkeypatch.setattr(dag_module, "_run_cli", lambda args: calls.append(args) or {"ok": True})
    monkeypatch.setattr(
        dag_module,
        "_resolve_build_source_instance_ids",
        lambda params: ["pytorch_reference.docs", "pytorch_reference.tutorials"],
    )

    dag_module._build_source(
        **_context(source_instance="pytorch_reference.docs,pytorch_reference.tutorials")
    )

    assert calls == [
            [
                "build-source",
                "--persist-build-run",
                "--catalog",
                "catalog.toml",
                "--source-instance",
                "pytorch_reference.docs",
            "--source-instance",
            "pytorch_reference.tutorials",
            "--rag-data-root",
            "assets/rag_data",
        ]
    ]


def test_build_source_task_accepts_build_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    dag_module = _load_dag(monkeypatch)
    calls: list[list[str]] = []

    monkeypatch.setattr(dag_module, "_run_cli", lambda args: calls.append(args) or {"ok": True})

    dag_module._build_source(**_context(build_run_id="airflow-run-1"))

    assert calls[0][:3] == ["build-source", "--persist-build-run", "--build-run-id"]
    assert calls[0][3] == "airflow-run-1"


def test_build_source_task_passes_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    dag_module = _load_dag(monkeypatch)
    calls: list[list[str]] = []

    monkeypatch.setattr(dag_module, "_run_cli", lambda args: calls.append(args) or {"ok": True})

    dag_module._build_source(**_context(dry_run=True))

    assert "--dry-run" in calls[0]


def test_run_cli_prints_subprocess_output_before_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dag_module = _load_dag(monkeypatch)
    calls: list[dict[str, object]] = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="partial stdout\n",
            stderr="real traceback\n",
        )

    monkeypatch.setattr(dag_module.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        dag_module._run_cli(["build-source"])

    captured = capsys.readouterr()
    assert "partial stdout" in captured.out
    assert "real traceback" in captured.err
    pythonpath = str(calls[0]["env"]["PYTHONPATH"])  # type: ignore[index]
    assert str(dag_module.PROJECT_ROOT / "src") in pythonpath.split(dag_module.os.pathsep)
    assert str(dag_module.PROJECT_ROOT) in pythonpath.split(dag_module.os.pathsep)


def test_materialize_task_uses_alias_config(monkeypatch: pytest.MonkeyPatch) -> None:
    dag_module = _load_dag(monkeypatch)
    calls: list[list[str]] = []

    monkeypatch.setattr(
        dag_module,
        "_run_cli",
        lambda args: calls.append(args) or {"summary": {"collection_name": "rag__x"}},
    )

    dag_module._materialize(**_context(collection="rag__pytorch_reference__test"))

    assert "--alias-config" in calls[0]
    assert "challenger" in calls[0]
    assert "--collection" in calls[0]
    assert "rag__pytorch_reference__test" in calls[0]


def test_materialize_task_accepts_build_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    dag_module = _load_dag(monkeypatch)
    calls: list[list[str]] = []

    monkeypatch.setattr(
        dag_module,
        "_run_cli",
        lambda args: calls.append(args) or {"summary": {"collection_name": "rag__x"}},
    )

    dag_module._materialize(**_context(build_run_id="airflow-run-1"))

    assert calls[0][:3] == ["materialize", "--persist-build-run", "--build-run-id"]
    assert calls[0][3] == "airflow-run-1"


def test_lifecycle_tasks_derive_safe_build_run_id_from_airflow_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dag_module = _load_dag(monkeypatch)
    calls: list[list[str]] = []

    monkeypatch.setattr(
        dag_module,
        "_run_cli",
        lambda args: calls.append(args) or {"summary": {"collection_name": "rag__x"}},
    )

    dag_module._build_source(
        **{
            **_context(),
            "run_id": "manual__2026-06-16T12:00:00+00:00",
        }
    )
    dag_module._materialize(
        **{
            **_context(),
            "run_id": "manual__2026-06-16T12:00:00+00:00",
        }
    )

    assert calls[0][:4] == [
        "build-source",
        "--persist-build-run",
        "--build-run-id",
        "manual_2026-06-16T12_00_00_00_00",
    ]
    assert calls[1][:4] == [
        "materialize",
        "--persist-build-run",
        "--build-run-id",
        "manual_2026-06-16T12_00_00_00_00",
    ]


def test_promote_task_skips_without_promote_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    dag_module = _load_dag(monkeypatch)
    monkeypatch.setattr(dag_module, "_run_cli", lambda args: pytest.fail("should not run CLI"))

    assert dag_module._promote_alias(**_context()) == {"promoted": False}


def test_promote_task_uses_materialize_xcom_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    dag_module = _load_dag(monkeypatch)
    calls: list[list[str]] = []

    class _TI:
        def xcom_pull(self, task_ids: str):
            assert task_ids == "materialize"
            return {"summary": {"collection_name": "rag__pytorch_reference__20260605_120000"}}

    monkeypatch.setattr(
        dag_module,
        "_run_cli",
        lambda args: calls.append(args) or {"alias_name": "rag__pytorch_reference__challenger"},
    )

    result = dag_module._promote_alias(
        **{
            **_context(promote_alias="challenger"),
            "ti": _TI(),
        }
    )

    assert result["promoted"] is True
    assert calls == [
        [
            "promote-alias",
            "--persist-build-run",
            "--catalog",
            "catalog.toml",
            "--kb",
            "pytorch_reference",
            "--alias",
            "challenger",
            "--collection",
            "rag__pytorch_reference__20260605_120000",
            "--rag-data-root",
            "assets/rag_data",
        ]
    ]


def test_sync_dvc_skips_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    dag_module = _load_dag(monkeypatch)

    assert dag_module._sync_dvc(**_context()) == {"synced": False, "paths": []}


def test_sync_dvc_syncs_generated_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dag_module = _load_dag(monkeypatch)
    monkeypatch.setattr(dag_module, "PROJECT_ROOT", tmp_path)
    (tmp_path / "catalog.toml").write_text(
        "\n".join(
            [
                "schema_version = 3",
                "",
                "[[knowledge_bases]]",
                'id = "pytorch_reference"',
                'description = "PyTorch docs"',
                'default_alias = "champion"',
                "",
                "[knowledge_bases.aliases.champion]",
                "top_k = 5",
                "score_threshold = 0.35",
                'retrieval_strategy = "dense"',
                "reranker_multiplier = 1",
                "",
                "[[source_adapters]]",
                'id = "generic.http_html"',
                'version = "1"',
                'description = "d"',
                'factory = "rag.ingest.adapters:make_http_html_adapter"',
                "",
                "[[source_instances]]",
                'id = "pytorch_reference.docs"',
                'description = "Official docs."',
                'role = "corpus"',
                'knowledge_base = "pytorch_reference"',
                'adapter = { id = "generic.http_html", version = "1" }',
                "",
            ]
        ),
        encoding="utf-8",
    )
    root = tmp_path / "rag_data"
    (root / "source_instances" / "pytorch_reference.docs" / "extracted").mkdir(parents=True)
    (root / "source_instances" / "pytorch_reference.docs" / "chunks").mkdir(parents=True)
    calls: list[dict[str, object]] = []

    def fake_sync(**kwargs):
        calls.append(kwargs)
        return {"changed": True, "dataset": kwargs["dataset_rel_path"]}

    fake_module = types.ModuleType("shared.airflow_git_sync")
    fake_module.sync_dvc_dataset_via_temp_clone = fake_sync
    monkeypatch.setitem(sys.modules, "shared.airflow_git_sync", fake_module)

    result = dag_module._sync_dvc(
        **_context(
            sync_dvc=True,
            rag_data_root=root.as_posix(),
            dvc_artifacts="extracted,chunks,raw",
            dvc_bot_branch="data-sync/rag-test",
        )
    )

    assert result["synced"] is True
    assert result["paths"] == [
        "rag_data/source_instances/pytorch_reference.docs/extracted",
        "rag_data/source_instances/pytorch_reference.docs/chunks",
    ]
    assert [call["dataset_rel_path"] for call in calls] == result["paths"]
    assert {call["bot_branch"] for call in calls} == {"data-sync/rag-test"}


def test_sync_dvc_syncs_kb_scoped_artifacts_separately(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dag_module = _load_dag(monkeypatch)
    monkeypatch.setattr(dag_module, "PROJECT_ROOT", tmp_path)
    (tmp_path / "catalog.toml").write_text("schema_version = 2\n", encoding="utf-8")
    root = tmp_path / "rag_data"
    (root / "knowledge_bases" / "pytorch_reference" / "manifests").mkdir(parents=True)
    (root / "knowledge_bases" / "pytorch_reference" / "metadata").mkdir(parents=True)

    fake_module = types.ModuleType("shared.airflow_git_sync")
    fake_module.sync_dvc_dataset_via_temp_clone = lambda **kwargs: {
        "changed": True,
        "dataset": kwargs["dataset_rel_path"],
    }
    monkeypatch.setitem(sys.modules, "shared.airflow_git_sync", fake_module)

    result = dag_module._sync_dvc(
        **_context(
            sync_dvc=True,
            rag_data_root=root.as_posix(),
            dvc_artifacts="manifests,metadata",
        )
    )

    assert result["paths"] == [
        "rag_data/knowledge_bases/pytorch_reference/manifests",
        "rag_data/knowledge_bases/pytorch_reference/metadata",
    ]
