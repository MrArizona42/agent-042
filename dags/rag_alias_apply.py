"""DAG: make one KB alias match its catalog declaration.

Calls `AliasService.apply()` directly -- the same application service the
`rag alias apply` CLI command calls -- rather than shelling through the CLI.
"""

from __future__ import annotations

import json
import os
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


def _project_path(path: Path | str) -> Path:
    """Resolve project-relative runtime paths inside the Airflow project mount."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _rag_context():
    """Build a CLI-compatible context with paths rooted at the mounted project."""
    from app_config.runtime import get_settings
    from rag.cli.factories import RagContext

    settings = get_settings()
    return RagContext(
        catalog_path_override=_project_path(settings.catalog.path),
        data_root_override=_project_path(settings.rag.data_root),
        as_json=True,
    )


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
}


def _params(context: dict[str, Any]) -> dict[str, Any]:
    return dict(context["params"])


def _apply_alias(**context: Any) -> dict[str, Any]:
    params = _params(context)
    kb_id = str(params["kb_id"]).strip()
    alias = str(params["alias"]).strip()
    if not kb_id:
        raise ValueError("kb_id is required")
    if not alias:
        raise ValueError("alias is required")

    from rag.cli.factories import build_alias_service, load_catalog_config
    from rag.control_plane.alias_service import AliasApplyRequest

    ctx = _rag_context()
    catalog_cfg = load_catalog_config(ctx)
    service = build_alias_service(ctx, catalog_cfg=catalog_cfg)

    refresh_sources = bool(params.get("refresh_sources"))
    result = service.apply(
        AliasApplyRequest(
            kb_id=kb_id,
            alias=alias,
            release_id=str(params.get("release_id") or "").strip() or None,
            allow_unevaluated=bool(params.get("allow_unevaluated")),
            allow_build_default=bool(params.get("allow_build_default")),
            refresh_sources=refresh_sources,
        )
    )

    payload = result.model_dump(mode="json")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return payload


def _sync_dvc(**context: Any) -> dict[str, Any]:
    """Optionally sync the release's source-instance artifacts to DVC."""
    params = _params(context)
    if not params.get("sync_dvc"):
        print("sync_dvc=false; skipping DVC sync.")
        return {"synced": False, "paths": []}

    from airflow_git_sync import sync_dvc_dataset_via_temp_clone
    from app_config.catalog import build_source_instance_index
    from rag.cli.factories import load_catalog_config

    ctx = _rag_context()
    catalog_cfg = load_catalog_config(ctx)
    kb_id = str(params["kb_id"]).strip()
    source_index = build_source_instance_index(catalog_cfg)
    source_instance_ids = [instance.id for instance in source_index.corpus_for_kb(kb_id)]

    rag_root = ctx.data_root
    candidate_paths = [
        rag_root / "source_instances" / source_instance_id
        for source_instance_id in source_instance_ids
    ]
    rel_paths: list[str] = []
    for path in candidate_paths:
        absolute_path = path if path.is_absolute() else PROJECT_ROOT / path
        if not absolute_path.exists():
            continue
        rel_paths.append(absolute_path.relative_to(PROJECT_ROOT).as_posix())

    if not rel_paths:
        print("No generated RAG source artifact paths exist yet; skipping DVC sync.")
        return {"synced": False, "paths": []}

    base_branch = str(params.get("dvc_base_branch") or "develop")
    bot_branch = str(params.get("dvc_bot_branch") or f"data/rag/{kb_id}")
    results = [
        sync_dvc_dataset_via_temp_clone(
            repo_root=PROJECT_ROOT,
            dataset_rel_path=rel_path,
            commit_message=f"Sync RAG source artifacts for {kb_id}: {Path(rel_path).name}",
            pr_title=f"Sync RAG source artifacts for {kb_id}",
            pr_body=(
                "Automated RAG source artifact sync.\n\n"
                f"- KB: `{kb_id}`\n"
                f"- Artifact path: `{rel_path}`\n"
                "- Raw cache is intentionally not DVC-tracked by this DAG."
            ),
            base_branch=base_branch,
            bot_branch=bot_branch,
        )
        for rel_path in rel_paths
    ]
    return {"synced": True, "paths": rel_paths, "results": results}


with DAG(
    dag_id="rag_alias_apply",
    default_args=default_args,
    description="Make one KB alias match its catalog declaration via AliasService.apply().",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["rag", "alias"],
    params={
        "kb_id": Param("", type="string"),
        "alias": Param("challenger", type="string"),
        "release_id": Param("", type=["null", "string"]),
        "refresh_sources": Param(False, type="boolean"),
        "sync_dvc": Param(False, type="boolean"),
        "dvc_base_branch": Param("develop", type="string"),
        "dvc_bot_branch": Param("", type=["null", "string"]),
        "allow_unevaluated": Param(False, type="boolean"),
        "allow_build_default": Param(False, type="boolean"),
    },
) as dag:
    apply_alias = PythonOperator(
        task_id="apply_alias",
        python_callable=_apply_alias,
    )

    sync_dvc = PythonOperator(
        task_id="sync_dvc",
        python_callable=_sync_dvc,
    )

    apply_alias >> sync_dvc
