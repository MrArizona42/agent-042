"""High-level source build orchestration before index materialization."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app_config.catalog import CatalogConfig, SourceConfig, materialize_catalog
from rag.ingest import (
    DEFAULT_SOURCE_ADAPTERS,
    SourceAdapter,
    SourceAdapterRegistry,
    build_catalog_adapter_registry,
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
    source: SourceConfig
    build: SourceBuildSummary


class CatalogSourcesBuildSummary(BaseModel):
    """Summary for one or more catalog-addressed source instance builds."""

    model_config = ConfigDict(extra="forbid")

    catalog_path: str
    kb_id: str
    sources: list[CatalogSourceBuildSummary]


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


def _resolve_default_adapter_registry(catalog: CatalogConfig) -> SourceAdapterRegistry:
    """Prefer catalog-declared adapters; fall back to defaults for legacy catalogs."""
    if catalog.source_adapters or catalog.benchmark_adapters:
        return build_catalog_adapter_registry(catalog)
    return DEFAULT_SOURCE_ADAPTERS


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
    adapter_registry: SourceAdapterRegistry,
) -> SourceAdapter:
    adapter = adapter_registry.get(source.ingest_adapter.id, version=source.ingest_adapter.version)
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
        adapter_registry=adapter_registry or _resolve_default_adapter_registry(catalog),
    )
    build = build_source_instance(
        kb_id=source.kb,
        source_instance_id=source.id,
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
    sources = resolve_catalog_sources(
        catalog,
        kb_id=kb_id,
        source_instance_ids=source_instance_ids,
    )
    summaries = [
        build_catalog_source(
            catalog_path=catalog_path,
            kb_id=kb_id,
            source_instance_id=source.id,
            rag_data_root=rag_data_root,
            document_ids=document_ids,
            limit=limit,
            force_fetch=force_fetch,
            force_extract=force_extract,
            force_chunk=force_chunk,
            chunking=chunking,
            adapter_registry=adapter_registry,
        )
        for source in sources
    ]
    return CatalogSourcesBuildSummary(
        catalog_path=catalog_path.as_posix(),
        kb_id=kb_id,
        sources=summaries,
    )
