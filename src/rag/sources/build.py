"""High-level source build orchestration before index materialization."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app_config.catalog import (
    CatalogConfig,
    SourceConfig,
    SourceInstanceRole,
    build_source_instance_index,
    conventional_manifest_path,
    legacy_source_instance_id,
    materialize_catalog,
    resolve_corpus_source_instance_ids,
)
from rag.ingest import (
    DEFAULT_SOURCE_ADAPTERS,
    SourceAdapter,
    SourceAdapterRegistry,
    load_adapter,
)
from rag.sources.chunks import (
    ChunkingConfig,
    SourceInstanceChunkingSummary,
    chunk_source_instance,
)
from rag.sources.processing import SourceProcessingSummary, process_source_instance

SourceBuildStatus = Literal["empty", "success", "partial", "failed"]


class SourceBuildSummary(BaseModel):
    """Summary for one complete pre-index source instance build."""

    model_config = ConfigDict(extra="forbid")

    kb_id: str
    source_instance_id: str
    source_type: str
    status: SourceBuildStatus
    processing: SourceProcessingSummary
    chunking: SourceInstanceChunkingSummary


class CatalogSourceBuildSummary(BaseModel):
    """Summary for one catalog-addressed source instance build."""

    model_config = ConfigDict(extra="forbid")

    catalog_path: str
    source_instance_id: str
    role: SourceInstanceRole
    source: SourceConfig | None = None
    build: SourceBuildSummary


class CatalogSourcesBuildSummary(BaseModel):
    """Summary for one or more catalog-addressed source instance builds."""

    model_config = ConfigDict(extra="forbid")

    catalog_path: str
    kb_id: str
    sources: list[CatalogSourceBuildSummary]


class GlobalSourceBuildSummary(BaseModel):
    """Summary for one source instance build addressed by its global id."""

    model_config = ConfigDict(extra="forbid")

    catalog_path: str
    source_instance_id: str
    role: SourceInstanceRole
    build: SourceBuildSummary


def _build_status(
    *,
    processing: SourceProcessingSummary,
    chunking: SourceInstanceChunkingSummary,
) -> SourceBuildStatus:
    if processing.total_selected == 0:
        return "empty"

    failures = len(processing.failed) + len(chunking.failed)
    successes = chunking.chunked + chunking.from_cache
    if successes == 0:
        return "failed"
    if failures:
        return "partial"
    return "success"


def _load_catalog_config(catalog_path: Path | str) -> CatalogConfig:
    path = Path(catalog_path)
    if path.suffix.lower() != ".toml":
        raise ValueError(f"Catalog must be a TOML file (got '{path.name}')")
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    catalog = CatalogConfig(**raw)
    materialize_catalog(catalog)
    return catalog


def _declared_adapter_config(
    catalog: CatalogConfig,
    *,
    adapter_id: str,
    version: str,
):
    for config in (*catalog.source_adapters, *catalog.benchmark_adapters):
        if config.id == adapter_id and config.version == version:
            return config
    return None


def _resolve_adapter_ref(
    catalog: CatalogConfig,
    *,
    adapter_id: str,
    version: str,
    adapter_registry: SourceAdapterRegistry | None,
) -> SourceAdapter:
    if adapter_registry is not None:
        return adapter_registry.get(adapter_id, version=version)
    if catalog.source_adapters or catalog.benchmark_adapters:
        config = _declared_adapter_config(catalog, adapter_id=adapter_id, version=version)
        if config is None:
            raise ValueError(f"Catalog references undeclared adapter '{adapter_id}@{version}'")
        return load_adapter(config, required_capabilities=frozenset({"source"}))
    return DEFAULT_SOURCE_ADAPTERS.get(adapter_id, version=version)


def _catalog_manifest_path(*, catalog_path: Path, manifest_ref: str) -> Path:
    path = Path(manifest_ref)
    if path.is_absolute():
        return path
    catalog_relative = catalog_path.parent / path
    return catalog_relative if catalog_relative.exists() else path


def _find_source_config(
    catalog: CatalogConfig,
    *,
    kb_id: str,
    source_instance_id: str,
) -> SourceConfig:
    for source in catalog.sources:
        if source.kb == kb_id and source.id == source_instance_id:
            return source
    raise ValueError(
        f"Catalog source not found for kb_id='{kb_id}' "
        f"and source_instance_id='{source_instance_id}'"
    )


def _legacy_source_for_instance_id(
    catalog: CatalogConfig,
    source_instance_id: str,
) -> SourceConfig | None:
    for source in catalog.sources:
        legacy_id = legacy_source_instance_id(kb_id=source.kb, local_source_id=source.id)
        if legacy_id == source_instance_id:
            return source
    return None


def resolve_catalog_sources(
    catalog: CatalogConfig,
    *,
    kb_id: str,
    source_instance_ids: list[str] | None = None,
) -> list[SourceConfig]:
    """Resolve all or selected source configs for a KB."""
    selected_ids = set(source_instance_ids) if source_instance_ids is not None else None
    sources = [
        source
        for source in catalog.sources
        if source.kb == kb_id and (selected_ids is None or source.id in selected_ids)
    ]
    if selected_ids is not None:
        found_ids = {source.id for source in sources}
        missing_ids = sorted(selected_ids - found_ids)
        if missing_ids:
            raise ValueError(
                f"Catalog sources not found for kb_id='{kb_id}' "
                f"and source_instance_ids={missing_ids}"
            )
    if not sources:
        raise ValueError(f"Catalog has no sources for kb_id='{kb_id}'")
    return sources


def build_source_instance(
    *,
    kb_id: str,
    source_instance_id: str,
    manifest_path: Path | str,
    rag_data_root: Path | str,
    source_adapter: SourceAdapter,
    document_ids: list[str] | None = None,
    limit: int | None = None,
    force_fetch: bool = False,
    force_extract: bool = False,
    force_chunk: bool = False,
    chunking: ChunkingConfig | None = None,
) -> SourceBuildSummary:
    """Run fetch/extract/chunk lifecycle for one source instance."""
    processing = process_source_instance(
        kb_id=kb_id,
        source_instance_id=source_instance_id,
        manifest_path=manifest_path,
        rag_data_root=rag_data_root,
        source_adapter=source_adapter,
        limit=limit,
        document_ids=document_ids,
        force_fetch=force_fetch,
        force_extract=force_extract,
    )
    chunking_summary = chunk_source_instance(
        rag_data_root=rag_data_root,
        kb_id=kb_id,
        source_instance_id=source_instance_id,
        document_ids=document_ids,
        limit=limit,
        config=chunking,
        force=force_chunk,
    )
    return SourceBuildSummary(
        kb_id=kb_id,
        source_instance_id=source_instance_id,
        source_type=source_adapter.source_type,
        status=_build_status(processing=processing, chunking=chunking_summary),
        processing=processing,
        chunking=chunking_summary,
    )


def _resolve_source_adapter(
    source: SourceConfig,
    *,
    catalog: CatalogConfig,
    adapter_registry: SourceAdapterRegistry | None,
) -> SourceAdapter:
    adapter = _resolve_adapter_ref(
        catalog,
        adapter_id=source.ingest_adapter.id,
        version=source.ingest_adapter.version,
        adapter_registry=adapter_registry,
    )
    if adapter.source_type != source.type:
        raise ValueError(
            f"Catalog source '{source.kb}/{source.id}' has type '{source.type}' but "
            f"ingest adapter '{adapter.adapter_id}@{adapter.version}' expects "
            f"source_type '{adapter.source_type}'"
        )
    return adapter


def build_catalog_source(
    *,
    catalog_path: Path | str,
    kb_id: str,
    source_instance_id: str,
    rag_data_root: Path | str,
    document_ids: list[str] | None = None,
    limit: int | None = None,
    force_fetch: bool = False,
    force_extract: bool = False,
    force_chunk: bool = False,
    chunking: ChunkingConfig | None = None,
    adapter_registry: SourceAdapterRegistry | None = None,
) -> CatalogSourceBuildSummary:
    """Build one source instance addressed by its catalog (kb, source id) pair."""
    catalog_path = Path(catalog_path)
    catalog = _load_catalog_config(catalog_path)
    source = _find_source_config(
        catalog,
        kb_id=kb_id,
        source_instance_id=source_instance_id,
    )
    source_adapter = _resolve_source_adapter(
        source,
        catalog=catalog,
        adapter_registry=adapter_registry,
    )
    build = build_source_instance(
        kb_id=source.kb,
        source_instance_id=legacy_source_instance_id(kb_id=source.kb, local_source_id=source.id),
        manifest_path=_catalog_manifest_path(
            catalog_path=catalog_path,
            manifest_ref=source.manifest,
        ),
        rag_data_root=rag_data_root,
        source_adapter=source_adapter,
        document_ids=document_ids,
        limit=limit,
        force_fetch=force_fetch,
        force_extract=force_extract,
        force_chunk=force_chunk,
        chunking=chunking,
    )
    return CatalogSourceBuildSummary(
        catalog_path=catalog_path.as_posix(),
        source_instance_id=legacy_source_instance_id(kb_id=source.kb, local_source_id=source.id),
        role="corpus",
        source=source,
        build=build,
    )


def build_catalog_sources(
    *,
    catalog_path: Path | str,
    kb_id: str,
    source_instance_ids: list[str] | None,
    rag_data_root: Path | str,
    document_ids: list[str] | None = None,
    limit: int | None = None,
    force_fetch: bool = False,
    force_extract: bool = False,
    force_chunk: bool = False,
    chunking: ChunkingConfig | None = None,
    adapter_registry: SourceAdapterRegistry | None = None,
) -> CatalogSourcesBuildSummary:
    """Build all or selected source instances for a KB."""
    catalog_path = Path(catalog_path)
    catalog = _load_catalog_config(catalog_path)
    resolved_ids = resolve_corpus_source_instance_ids(
        catalog,
        kb_id=kb_id,
        source_ids=source_instance_ids,
    )
    built = build_source_instances_by_global_id(
        catalog_path=catalog_path,
        source_instance_ids=resolved_ids,
        rag_data_root=rag_data_root,
        document_ids=document_ids,
        limit=limit,
        force_fetch=force_fetch,
        force_extract=force_extract,
        force_chunk=force_chunk,
        chunking=chunking,
        adapter_registry=adapter_registry,
    )
    index = build_source_instance_index(catalog)
    summaries = [
        CatalogSourceBuildSummary(
            catalog_path=catalog_path.as_posix(),
            source_instance_id=summary.source_instance_id,
            role=summary.role,
            source=_legacy_source_for_instance_id(catalog, summary.source_instance_id)
            if index.is_legacy(summary.source_instance_id)
            else None,
            build=summary.build,
        )
        for summary in built
    ]
    return CatalogSourcesBuildSummary(
        catalog_path=catalog_path.as_posix(),
        kb_id=kb_id,
        sources=summaries,
    )


def _reject_benchmark_target(*, source_instance_id: str, role: SourceInstanceRole) -> None:
    if role == "benchmark":
        raise ValueError(
            f"Source instance '{source_instance_id}' has role 'benchmark'; build-source only "
            "builds role 'corpus' instances. Use prepare-benchmark for benchmark sources."
        )


def build_source_instance_by_global_id(
    *,
    catalog_path: Path | str,
    source_instance_id: str,
    rag_data_root: Path | str,
    document_ids: list[str] | None = None,
    limit: int | None = None,
    force_fetch: bool = False,
    force_extract: bool = False,
    force_chunk: bool = False,
    chunking: ChunkingConfig | None = None,
    adapter_registry: SourceAdapterRegistry | None = None,
) -> GlobalSourceBuildSummary:
    """Build one source instance addressed by its globally unique id.

    Rejects `role = "benchmark"` targets; those are built via the benchmark
    preparation pipeline instead.
    """
    catalog_path = Path(catalog_path)
    catalog = _load_catalog_config(catalog_path)
    index = build_source_instance_index(catalog)
    instance = index.get(source_instance_id)
    _reject_benchmark_target(source_instance_id=source_instance_id, role=instance.role)

    if index.is_legacy(source_instance_id):
        legacy_source = next(
            source
            for source in catalog.sources
            if legacy_source_instance_id(kb_id=source.kb, local_source_id=source.id)
            == source_instance_id
        )
        manifest_path = _catalog_manifest_path(
            catalog_path=catalog_path,
            manifest_ref=legacy_source.manifest,
        )
        source_adapter = _resolve_source_adapter(
            legacy_source,
            catalog=catalog,
            adapter_registry=adapter_registry,
        )
    else:
        manifest_path = conventional_manifest_path(rag_data_root, instance.id)
        source_adapter = _resolve_adapter_ref(
            catalog,
            adapter_id=instance.adapter.id,
            version=instance.adapter.version,
            adapter_registry=adapter_registry,
        )

    build = build_source_instance(
        kb_id=instance.knowledge_base,
        source_instance_id=instance.id,
        manifest_path=manifest_path,
        rag_data_root=rag_data_root,
        source_adapter=source_adapter,
        document_ids=document_ids,
        limit=limit,
        force_fetch=force_fetch,
        force_extract=force_extract,
        force_chunk=force_chunk,
        chunking=chunking,
    )
    return GlobalSourceBuildSummary(
        catalog_path=catalog_path.as_posix(),
        source_instance_id=instance.id,
        role=instance.role,
        build=build,
    )


def build_source_instances_by_global_id(
    *,
    catalog_path: Path | str,
    source_instance_ids: list[str],
    rag_data_root: Path | str,
    document_ids: list[str] | None = None,
    limit: int | None = None,
    force_fetch: bool = False,
    force_extract: bool = False,
    force_chunk: bool = False,
    chunking: ChunkingConfig | None = None,
    adapter_registry: SourceAdapterRegistry | None = None,
) -> list[GlobalSourceBuildSummary]:
    """Build multiple source instances addressed by their globally unique ids."""
    return [
        build_source_instance_by_global_id(
            catalog_path=catalog_path,
            source_instance_id=source_instance_id,
            rag_data_root=rag_data_root,
            document_ids=document_ids,
            limit=limit,
            force_fetch=force_fetch,
            force_extract=force_extract,
            force_chunk=force_chunk,
            chunking=chunking,
            adapter_registry=adapter_registry,
        )
        for source_instance_id in source_instance_ids
    ]
