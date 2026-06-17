"""Shared RAG lifecycle stage functions used by CLI and Airflow wrappers."""

from __future__ import annotations

import hashlib
import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from app_config.catalog import CatalogConfig, materialize_catalog
from rag.lifecycle.models import (
    BuildRequest,
    BuildRun,
    LifecycleStageResult,
    PlanResult,
    SourcePlanEntry,
)
from rag.sources.cache import sha256_bytes


def _json_payload(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, (list, tuple)):
        return [_json_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_payload(item) for key, item in value.items()}
    return value


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return sha256_bytes(path.read_bytes())


def _digest_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _run_id(*, kb_id: str, created_at: datetime) -> str:
    return f"rag_build_{kb_id}_{created_at.strftime('%Y%m%d_%H%M%S')}"


def _manifest_path(*, catalog_path: Path, manifest_ref: str) -> Path:
    path = Path(manifest_ref)
    if path.is_absolute():
        return path
    catalog_relative = catalog_path.parent / path
    return catalog_relative if catalog_relative.exists() else path


def _source_attestation(request: BuildRequest) -> tuple[dict[str, str], dict[str, str]]:
    catalog_path = Path(request.catalog_path)
    if not catalog_path.exists() or not catalog_path.is_file():
        return {}, {}

    try:
        catalog = CatalogConfig(**tomllib.loads(catalog_path.read_text(encoding="utf-8")))
    except (OSError, tomllib.TOMLDecodeError, ValueError):
        return {}, {}

    selected = set(request.source_ids) if request.source_ids is not None else None
    manifest_digests: dict[str, str] = {}
    adapter_versions: dict[str, str] = {}

    for source in catalog.sources:
        if source.kb != request.kb_id:
            continue
        if selected is not None and source.id not in selected:
            continue

        digest = _sha256_file(
            _manifest_path(catalog_path=catalog_path, manifest_ref=source.manifest)
        )
        if digest is not None:
            manifest_digests[source.id] = digest

        adapter_versions[source.id] = f"{source.ingest_adapter.id}@{source.ingest_adapter.version}"

    return manifest_digests, adapter_versions


def create_build_run(
    request: BuildRequest,
    *,
    run_id: str | None = None,
    created_at: datetime | None = None,
) -> BuildRun:
    """Create a planned BuildRun for a request."""
    created_at = created_at or datetime.now(tz=UTC)
    manifest_digests, adapter_versions = _source_attestation(request)
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
        manifest_digests=manifest_digests,
        adapter_versions=adapter_versions,
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


def read_build_run(*, rag_data_root: Path | str, kb_id: str, run_id: str) -> BuildRun:
    """Load a persisted BuildRun JSON artifact."""
    path = build_run_path(rag_data_root=rag_data_root, kb_id=kb_id, run_id=run_id)
    return BuildRun(**json.loads(path.read_text(encoding="utf-8")))


def load_or_create_build_run(request: BuildRequest, *, run_id: str | None) -> BuildRun:
    if run_id is None:
        return create_build_run(request)
    path = build_run_path(
        rag_data_root=request.rag_data_root,
        kb_id=request.kb_id,
        run_id=run_id,
    )
    if path.exists():
        return read_build_run(
            rag_data_root=request.rag_data_root, kb_id=request.kb_id, run_id=run_id
        )
    return create_build_run(request, run_id=run_id)


def _run_recorded_stage(
    request: BuildRequest,
    *,
    stage_name: str,
    stage_fn: Callable[[], Any],
    run_id: str | None = None,
    persist: bool = True,
    success_status: str = "succeeded",
) -> LifecycleStageResult:
    build_run = load_or_create_build_run(request, run_id=run_id).model_copy(
        update={
            "status": "running",
            "current_stage": stage_name,
            "finished_at": None,
            "alias_config": request.alias_config,
            "collection_name": request.collection_name,
        }
    )
    if request.dry_run:
        result = {
            "dry_run": True,
            "stage": stage_name,
            "request": request.model_dump(mode="json", exclude_none=True),
        }
        build_run = build_run.model_copy(
            update={
                "status": "planned",
                "current_stage": stage_name,
                "stage_results": {**build_run.stage_results, stage_name: result},
                "errors": [],
            }
        )
        if persist:
            write_build_run(build_run)
        return LifecycleStageResult(build_run=build_run, result=result)

    try:
        result = stage_fn()
        stage_results = {**build_run.stage_results, stage_name: _json_payload(result)}
        build_run = build_run.model_copy(
            update={
                "status": success_status,
                "finished_at": datetime.now(tz=UTC),
                "stage_results": stage_results,
                "errors": [],
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


def plan_build(
    request: BuildRequest,
    *,
    adapter_registry: Any | None = None,
) -> PlanResult:
    """Validate catalog, sources, and adapters for a KB build without executing anything."""
    catalog_path = Path(request.catalog_path)
    result = PlanResult(
        kb_id=request.kb_id,
        catalog_path=request.catalog_path,
        catalog_reachable=False,
        kb_found=False,
    )

    if not catalog_path.exists() or not catalog_path.is_file():
        return result.model_copy(update={"errors": [f"Catalog not found: {catalog_path}"]})

    try:
        catalog = CatalogConfig(**tomllib.loads(catalog_path.read_text(encoding="utf-8")))
        materialize_catalog(catalog)
    except Exception as exc:
        return result.model_copy(
            update={
                "catalog_reachable": True,
                "errors": [f"Catalog parse error: {exc}"],
            }
        )

    result = result.model_copy(update={"catalog_reachable": True})

    kb_sources = [s for s in catalog.sources if s.kb == request.kb_id]
    if not kb_sources:
        return result.model_copy(update={"errors": [f"No sources found for KB '{request.kb_id}'"]})

    result = result.model_copy(update={"kb_found": True})

    if request.source_ids is not None:
        selected_ids = set(request.source_ids)
        missing = sorted(selected_ids - {s.id for s in kb_sources})
        if missing:
            return result.model_copy(
                update={"errors": [f"Source IDs not found in catalog: {missing}"]}
            )
        kb_sources = [s for s in kb_sources if s.id in selected_ids]

    if adapter_registry is None:
        from rag.ingest import DEFAULT_SOURCE_ADAPTERS

        adapter_registry = DEFAULT_SOURCE_ADAPTERS

    entries: list[SourcePlanEntry] = []
    for source in kb_sources:
        adapter_id = source.ingest_adapter.id
        adapter_version = source.ingest_adapter.version
        manifest_ref = source.manifest
        manifest_path_resolved = _manifest_path(
            catalog_path=catalog_path, manifest_ref=manifest_ref
        )
        manifest_reachable = manifest_path_resolved.exists() and manifest_path_resolved.is_file()

        adapter_registered = False
        source_type_matches = False
        errors: list[str] = []

        try:
            adapter = adapter_registry.get(adapter_id, version=adapter_version)
            adapter_registered = True
            if adapter.source_type != source.type:
                errors.append(
                    f"Adapter '{adapter_id}@{adapter_version}' expects source_type "
                    f"'{adapter.source_type}' but catalog declares '{source.type}'"
                )
            else:
                source_type_matches = True
        except Exception as exc:
            errors.append(f"Adapter '{adapter_id}@{adapter_version}' not registered: {exc}")

        if not manifest_reachable:
            errors.append(f"Source manifest not found: {manifest_path_resolved}")

        entries.append(
            SourcePlanEntry(
                source_id=source.id,
                adapter_id=adapter_id,
                adapter_version=adapter_version,
                manifest_ref=manifest_ref,
                manifest_reachable=manifest_reachable,
                adapter_registered=adapter_registered,
                source_type_matches=source_type_matches,
                errors=errors,
            )
        )

    return result.model_copy(update={"sources": entries})


def list_build_runs(*, rag_data_root: Path | str, kb_id: str) -> list[BuildRun]:
    """Return all persisted BuildRun artifacts for a KB, newest first."""
    runs_dir = Path(rag_data_root) / kb_id / "metadata" / "build_runs"
    if not runs_dir.is_dir():
        return []
    runs: list[BuildRun] = []
    for path in sorted(runs_dir.glob("*.json"), reverse=True):
        try:
            runs.append(BuildRun(**json.loads(path.read_text(encoding="utf-8"))))
        except Exception:
            pass
    return runs


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

    def _stage() -> Any:
        if request.source_ids is not None and len(request.source_ids) == 1:
            return build_catalog_source_fn(
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
        return build_catalog_sources_fn(
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

    return _run_recorded_stage(
        request,
        stage_name="build_source",
        stage_fn=_stage,
        run_id=run_id,
        persist=persist,
    )


def run_materialize_stage(
    request: BuildRequest,
    *,
    stage_fn: Callable[[], Any],
    run_id: str | None = None,
    persist: bool = True,
) -> LifecycleStageResult:
    """Run materialization and update a BuildRun."""
    return _run_recorded_stage(
        request,
        stage_name="materialize",
        stage_fn=stage_fn,
        run_id=run_id,
        persist=persist,
    )


def run_alias_promotion_stage(
    request: BuildRequest,
    *,
    stage_fn: Callable[[], Any],
    run_id: str | None = None,
    persist: bool = True,
) -> LifecycleStageResult:
    """Run alias promotion and update a BuildRun."""
    return _run_recorded_stage(
        request,
        stage_name="promote_alias",
        stage_fn=stage_fn,
        run_id=run_id,
        persist=persist,
        success_status="promoted",
    )
