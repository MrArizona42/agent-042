"""Shared RAG lifecycle stage functions used by CLI and Airflow wrappers."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from rag.lifecycle.models import BuildRequest, BuildRun, LifecycleStageResult


def _json_payload(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, list):
        return [_json_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_json_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_payload(item) for key, item in value.items()}
    return value


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _digest_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _run_id(*, kb_id: str, created_at: datetime) -> str:
    return f"rag_build_{kb_id}_{created_at.strftime('%Y%m%d_%H%M%S')}"


def create_build_run(
    request: BuildRequest,
    *,
    run_id: str | None = None,
    created_at: datetime | None = None,
) -> BuildRun:
    """Create a planned BuildRun for a request."""
    created_at = created_at or datetime.now(tz=UTC)
    profile_payload = request.model_dump(
        mode="json",
        exclude={"catalog_path", "rag_data_root"},
        exclude_none=True,
    )
    return BuildRun(
        run_id=run_id or _run_id(kb_id=request.kb_id, created_at=created_at),
        kb_id=request.kb_id,
        source_ids=request.source_ids,
        status="planned",
        catalog_path=request.catalog_path,
        rag_data_root=request.rag_data_root,
        alias_config=request.alias_config,
        collection_name=request.collection_name,
        catalog_digest=_sha256_file(Path(request.catalog_path)),
        build_profile_digest=_digest_payload(profile_payload),
        started_at=created_at,
    )


def build_run_path(*, rag_data_root: Path | str, kb_id: str, run_id: str) -> Path:
    """Return the conventional BuildRun artifact path."""
    return Path(rag_data_root) / kb_id / "metadata" / "build_runs" / f"{run_id}.json"


def write_build_run(build_run: BuildRun) -> Path:
    """Persist a BuildRun JSON artifact."""
    path = build_run_path(
        rag_data_root=build_run.rag_data_root,
        kb_id=build_run.kb_id,
        run_id=build_run.run_id,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_run.model_dump(mode="json", exclude_none=True), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def _default_build_catalog_source_fn() -> Callable[..., Any]:
    from rag.sources.build import build_catalog_source

    return build_catalog_source


def _default_build_catalog_sources_fn() -> Callable[..., Any]:
    from rag.sources.build import build_catalog_sources

    return build_catalog_sources


def run_source_build_stage(
    request: BuildRequest,
    *,
    run_id: str | None = None,
    build_catalog_source_fn: Callable[..., Any] | None = None,
    build_catalog_sources_fn: Callable[..., Any] | None = None,
    persist: bool = True,
    **build_kwargs: Any,
) -> LifecycleStageResult:
    """Run the source build stage and update a BuildRun."""
    build_catalog_source_fn = build_catalog_source_fn or _default_build_catalog_source_fn()
    build_catalog_sources_fn = build_catalog_sources_fn or _default_build_catalog_sources_fn()
    build_run = create_build_run(request, run_id=run_id).model_copy(
        update={"status": "running", "current_stage": "build_source"}
    )
    try:
        if request.source_ids is not None and len(request.source_ids) == 1:
            result = build_catalog_source_fn(
                catalog_path=request.catalog_path,
                kb_id=request.kb_id,
                source_instance_id=request.source_ids[0],
                rag_data_root=request.rag_data_root,
                document_ids=request.document_ids,
                limit=request.limit,
                force_fetch=request.force_fetch,
                force_extract=request.force_extract,
                force_chunk=request.force_chunk,
                **build_kwargs,
            )
        else:
            result = build_catalog_sources_fn(
                catalog_path=request.catalog_path,
                kb_id=request.kb_id,
                source_instance_ids=request.source_ids,
                rag_data_root=request.rag_data_root,
                document_ids=request.document_ids,
                limit=request.limit,
                force_fetch=request.force_fetch,
                force_extract=request.force_extract,
                force_chunk=request.force_chunk,
                **build_kwargs,
            )
        build_run = build_run.model_copy(
            update={
                "status": "succeeded",
                "finished_at": datetime.now(tz=UTC),
                "stage_results": {"build_source": _json_payload(result)},
            }
        )
        if persist:
            write_build_run(build_run)
        return LifecycleStageResult(build_run=build_run, result=result)
    except Exception as exc:
        build_run = build_run.model_copy(
            update={
                "status": "failed",
                "finished_at": datetime.now(tz=UTC),
                "errors": [str(exc)],
            }
        )
        if persist:
            write_build_run(build_run)
        raise
