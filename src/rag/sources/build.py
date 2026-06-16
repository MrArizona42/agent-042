"""High-level source build orchestration before index materialization."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app_config.catalog import CatalogConfig, SourceConfig, materialize_catalog
from rag.ingest import DEFAULT_SOURCE_ADAPTERS, SourceAdapter, SourceAdapterRegistry
from rag.sources.chunks import (
    ChunkingConfig,
    SourceInstanceChunkingSummary,
    chunk_source_instance,
)
from rag.sources.extractors import SourceExtractor
from rag.sources.fetchers import SourceFetcher
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


def build_source_instance(
    *,
    kb_id: str,
    source_instance_id: str,
    source_type: str,
    manifest_path: Path | str,
    rag_data_root: Path | str,
    document_ids: list[str] | None = None,
    limit: int | None = None,
    force_fetch: bool = False,
    force_extract: bool = False,
    force_chunk: bool = False,
    chunking: ChunkingConfig | None = None,
    fetchers: dict[str, SourceFetcher] | None = None,
    extractors: dict[str, SourceExtractor] | None = None,
    source_adapter: SourceAdapter | None = None,
) -> SourceBuildSummary:
    """Run fetch/extract/chunk lifecycle for one source instance."""
    processing = process_source_instance(
        kb_id=kb_id,
        source_instance_id=source_instance_id,
        source_type=source_type,
        manifest_path=manifest_path,
        rag_data_root=rag_data_root,
        limit=limit,
        document_ids=document_ids,
        force_fetch=force_fetch,
        force_extract=force_extract,
        source_adapter=source_adapter,
        fetchers=fetchers,
        extractors=extractors,
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
        source_type=source_adapter.source_type if source_adapter is not None else source_type,
        status=_build_status(processing=processing, chunking=chunking_summary),
        processing=processing,
        chunking=chunking_summary,
    )


def _resolve_source_adapter(
    source: SourceConfig,
    *,
    adapter_registry: SourceAdapterRegistry,
) -> SourceAdapter:
    ingest_adapter = source.ingest_adapter
    if ingest_adapter is None:
        raise ValueError(f"Catalog source '{source.kb}/{source.id}' is missing ingest_adapter")
    adapter = adapter_registry.get(ingest_adapter.id, version=ingest_adapter.version)
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
    fetchers: dict[str, SourceFetcher] | None = None,
    extractors: dict[str, SourceExtractor] | None = None,
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
        adapter_registry=adapter_registry or DEFAULT_SOURCE_ADAPTERS,
    )
    build = build_source_instance(
        kb_id=source.kb,
        source_instance_id=source.id,
        source_type=source.type,
        manifest_path=source.manifest,
        rag_data_root=rag_data_root,
        document_ids=document_ids,
        limit=limit,
        force_fetch=force_fetch,
        force_extract=force_extract,
        force_chunk=force_chunk,
        chunking=chunking,
        fetchers=fetchers,
        extractors=extractors,
        source_adapter=source_adapter,
    )
    return CatalogSourceBuildSummary(
        catalog_path=catalog_path.as_posix(),
        source=source,
        build=build,
    )
