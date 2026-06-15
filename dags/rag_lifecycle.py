"""Generic DAG: RAG source build, materialization, and optional alias promotion."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(os.environ["CONTAINER__PROJECT_ROOT"])


def _bootstrap_project_imports() -> None:
    """Ensure task-time imports can resolve the repo's src/ layout."""
    for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_project_imports()


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
}


def _params(context: dict[str, Any]) -> dict[str, Any]:
    return dict(context["params"])


def _csv_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _optional_positive_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _append_common_source_args(cmd: list[str], params: dict[str, Any]) -> None:
    cmd.extend(
        [
            "--catalog",
            str(params["catalog"]),
            "--kb",
            str(params["kb"]),
            "--source",
            str(params["source"]),
            "--rag-data-root",
            str(params["rag_data_root"]),
        ]
    )

    for document_id in _csv_values(params.get("document_ids")):
        cmd.extend(["--document-id", document_id])

    limit = _optional_positive_int(params.get("limit"))
    if limit is not None:
        cmd.extend(["--limit", str(limit)])


def _run_cli(args: list[str]) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "rag.sources.cli", *args]
    env = os.environ.copy()
    python_path_parts = [
        str(PROJECT_ROOT / "src"),
        str(PROJECT_ROOT),
        *[path for path in env.get("PYTHONPATH", "").split(os.pathsep) if path],
    ]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(python_path_parts))
    print("+", " ".join(cmd))
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    if completed.stdout:
        print(completed.stdout)
    completed.check_returncode()
    return json.loads(completed.stdout) if completed.stdout.strip() else {}


def _build_source(**context: Any) -> dict[str, Any]:
    params = _params(context)
    cmd = ["build-source"]
    _append_common_source_args(cmd, params)
    if params.get("force_fetch"):
        cmd.append("--force-fetch")
    if params.get("force_extract"):
        cmd.append("--force-extract")
    if params.get("force_chunk"):
        cmd.append("--force-chunk")
    return _run_cli(cmd)


def _materialize(**context: Any) -> dict[str, Any]:
    params = _params(context)
    cmd = ["materialize"]
    _append_common_source_args(cmd, params)
    cmd.extend(["--alias-config", str(params["alias_config"])])
    collection = str(params.get("collection") or "").strip()
    if collection:
        cmd.extend(["--collection", collection])
    if params.get("force_recreate"):
        cmd.append("--force-recreate")
    return _run_cli(cmd)


def _materialized_collection(context: dict[str, Any]) -> str:
    params = _params(context)
    collection = str(params.get("collection") or "").strip()
    if collection:
        return collection

    ti = context["ti"]
    payload = ti.xcom_pull(task_ids="materialize") or {}
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if isinstance(summary, dict) and summary.get("collection_name"):
        return str(summary["collection_name"])
    if isinstance(payload, dict) and payload.get("collection_name"):
        return str(payload["collection_name"])
    raise RuntimeError("Could not resolve materialized collection name from XCom")


def _promote_alias(**context: Any) -> dict[str, Any]:
    params = _params(context)
    promote_alias = str(params.get("promote_alias") or "").strip()
    if not promote_alias:
        print("No promote_alias configured; skipping promotion.")
        return {"promoted": False}

    collection = _materialized_collection(context)
    cmd = [
        "promote-alias",
        "--catalog",
        str(params["catalog"]),
        "--kb",
        str(params["kb"]),
        "--alias",
        promote_alias,
        "--collection",
        collection,
    ]
    result = _run_cli(cmd)
    return {"promoted": True, **result}


def _dvc_artifact_rel_paths(params: dict[str, Any]) -> list[str]:
    """Return generated RAG artifact paths that should be DVC-tracked."""
    rag_root = Path(str(params["rag_data_root"]))
    kb_id = str(params["kb"])
    root = rag_root / kb_id
    requested = _csv_values(params.get("dvc_artifacts"))
    artifact_names = requested or ["extracted", "chunks", "manifests", "metadata"]
    rel_paths: list[str] = []
    for artifact_name in artifact_names:
        path = root / artifact_name
        absolute_path = path if path.is_absolute() else PROJECT_ROOT / path
        if not absolute_path.exists():
            continue
        try:
            rel_path = absolute_path.relative_to(PROJECT_ROOT)
        except ValueError as exc:
            raise ValueError(
                f"DVC artifact path must be under CONTAINER__PROJECT_ROOT: {absolute_path}"
            ) from exc
        rel_paths.append(rel_path.as_posix())
    return rel_paths


def _sync_dvc(**context: Any) -> dict[str, Any]:
    """Optionally sync generated RAG artifacts to DVC through a temp clone."""
    params = _params(context)
    if not params.get("sync_dvc"):
        print("sync_dvc=false; skipping DVC sync.")
        return {"synced": False, "paths": []}

    from shared.airflow_git_sync import sync_dvc_dataset_via_temp_clone

    rel_paths = _dvc_artifact_rel_paths(params)
    if not rel_paths:
        print("No generated RAG artifact paths exist yet; skipping DVC sync.")
        return {"synced": False, "paths": []}

    kb_id = str(params["kb"])
    base_branch = str(params.get("dvc_base_branch") or "develop")
    bot_branch = str(params.get("dvc_bot_branch") or f"data-sync/rag-{kb_id}")
    results = []
    for rel_path in rel_paths:
        results.append(
            sync_dvc_dataset_via_temp_clone(
                repo_root=PROJECT_ROOT,
                dataset_rel_path=rel_path,
                commit_message=f"Sync RAG artifacts for {kb_id}: {Path(rel_path).name}",
                pr_title=f"Sync RAG artifacts for {kb_id}",
                pr_body=(
                    "Automated RAG artifact sync.\n\n"
                    f"- KB: `{kb_id}`\n"
                    f"- Artifact path: `{rel_path}`\n"
                    "- Raw cache is intentionally not DVC-tracked by this DAG."
                ),
                base_branch=base_branch,
                bot_branch=bot_branch,
            )
        )

    return {
        "synced": True,
        "paths": rel_paths,
        "results": results,
    }


with DAG(
    dag_id="rag_lifecycle",
    default_args=default_args,
    description="Generic RAG lifecycle: build-source, materialize, optional promote-alias",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["rag", "lifecycle"],
    params={
        "catalog": Param("catalog.toml", type="string"),
        "kb": Param("pytorch_reference", type="string"),
        "source": Param("docs", type="string"),
        "alias_config": Param("challenger", type="string"),
        "rag_data_root": Param("assets/rag_data", type="string"),
        "document_ids": Param("", type=["null", "string", "array"]),
        "limit": Param(0, type="integer", minimum=0),
        "collection": Param("", type=["null", "string"]),
        "promote_alias": Param("", type=["null", "string"]),
        "sync_dvc": Param(False, type="boolean"),
        "dvc_artifacts": Param("", type=["null", "string", "array"]),
        "dvc_base_branch": Param("develop", type="string"),
        "dvc_bot_branch": Param("", type=["null", "string"]),
        "force_fetch": Param(False, type="boolean"),
        "force_extract": Param(False, type="boolean"),
        "force_chunk": Param(False, type="boolean"),
        "force_recreate": Param(False, type="boolean"),
    },
) as dag:
    build_source = PythonOperator(
        task_id="build_source",
        python_callable=_build_source,
    )

    materialize = PythonOperator(
        task_id="materialize",
        python_callable=_materialize,
    )

    promote_alias = PythonOperator(
        task_id="promote_alias",
        python_callable=_promote_alias,
    )

    sync_dvc = PythonOperator(
        task_id="sync_dvc",
        python_callable=_sync_dvc,
    )

    build_source >> materialize >> sync_dvc >> promote_alias
