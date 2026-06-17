"""Unified source-instance index merging legacy `[[sources]]` and `[[source_instances]]`."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from app_config.catalog.schema import (
    CatalogConfig,
    SourceConfig,
    SourceInstanceAdapterRef,
    SourceInstanceConfig,
)


def _normalize_legacy_source(source: SourceConfig) -> SourceInstanceConfig:
    """Project a legacy `[[sources]]` entry into the new source-instance shape."""
    return SourceInstanceConfig(
        id=f"{source.kb}.{source.id}",
        description=f"Legacy source '{source.id}' for KB '{source.kb}' (type '{source.type}')",
        role="corpus",
        knowledge_base=source.kb,
        adapter=SourceInstanceAdapterRef(
            id=source.ingest_adapter.id,
            version=source.ingest_adapter.version,
        ),
    )


@dataclass(frozen=True, slots=True)
class SourceInstanceIndex:
    """Queryable index over all source instances declared or normalized from a catalog."""

    by_id: dict[str, SourceInstanceConfig]
    legacy_ids: frozenset[str]

    def get(self, source_instance_id: str) -> SourceInstanceConfig:
        """Return the source instance for the given global id, or raise."""
        instance = self.by_id.get(source_instance_id)
        if instance is None:
            raise ValueError(f"Unknown source instance id '{source_instance_id}'")
        return instance

    def all(self) -> list[SourceInstanceConfig]:
        """Return every source instance, declared and legacy-normalized."""
        return list(self.by_id.values())

    def corpus_for_kb(self, kb_id: str) -> list[SourceInstanceConfig]:
        """Return corpus-role source instances attached to the given KB."""
        return [
            instance
            for instance in self.by_id.values()
            if instance.knowledge_base == kb_id and instance.role == "corpus"
        ]

    def benchmark_for_kb(self, kb_id: str) -> list[SourceInstanceConfig]:
        """Return benchmark-role source instances attached to the given KB."""
        return [
            instance
            for instance in self.by_id.values()
            if instance.knowledge_base == kb_id and instance.role == "benchmark"
        ]

    def is_legacy(self, source_instance_id: str) -> bool:
        """Whether a source instance id was normalized from a legacy `[[sources]]` entry."""
        return source_instance_id in self.legacy_ids


def build_source_instance_index(catalog_cfg: CatalogConfig) -> SourceInstanceIndex:
    """Merge legacy `[[sources]]` and declared `[[source_instances]]` into one index."""
    kb_ids = {kb.id.strip() for kb in catalog_cfg.knowledge_bases}
    adapter_ids = {(a.id, a.version) for a in catalog_cfg.source_adapters} | {
        (a.id, a.version) for a in catalog_cfg.benchmark_adapters
    }
    benchmark_adapter_ids = {(a.id, a.version) for a in catalog_cfg.benchmark_adapters}

    by_id: dict[str, SourceInstanceConfig] = {}
    legacy_ids: set[str] = set()

    for source in catalog_cfg.sources:
        if source.kb not in kb_ids:
            raise ValueError(f"Source '{source.id}' references unknown KB '{source.kb}'")
        instance = _normalize_legacy_source(source)
        if instance.id in by_id:
            raise ValueError(f"Duplicate source instance id '{instance.id}'")
        by_id[instance.id] = instance
        legacy_ids.add(instance.id)

    for instance in catalog_cfg.source_instances:
        if instance.id in by_id:
            raise ValueError(f"Duplicate source instance id '{instance.id}'")
        if instance.knowledge_base not in kb_ids:
            raise ValueError(
                f"Source instance '{instance.id}' references unknown KB '{instance.knowledge_base}'"
            )
        adapter_key = (instance.adapter.id, instance.adapter.version)
        if adapter_key not in adapter_ids:
            raise ValueError(
                f"Source instance '{instance.id}' references undeclared adapter "
                f"'{instance.adapter.id}@{instance.adapter.version}'"
            )
        if instance.role == "benchmark" and adapter_key not in benchmark_adapter_ids:
            raise ValueError(
                f"Benchmark source instance '{instance.id}' must use a benchmark-capable "
                f"adapter, got '{instance.adapter.id}@{instance.adapter.version}'"
            )
        by_id[instance.id] = instance

    return SourceInstanceIndex(by_id=by_id, legacy_ids=frozenset(legacy_ids))


def conventional_manifest_path(rag_data_root: Path | str, source_instance_id: str) -> Path:
    """Derive the conventional manifest path for a source instance id."""
    return Path(rag_data_root) / "source_instances" / source_instance_id / "manifest.toml"


def validate_source_instance_manifests_exist(
    index: SourceInstanceIndex,
    *,
    rag_data_root: Path | str,
) -> None:
    """Raise if a declared (non-legacy) source instance has no readable conventional manifest."""
    for instance in index.all():
        if index.is_legacy(instance.id):
            continue
        manifest_path = conventional_manifest_path(rag_data_root, instance.id)
        if not manifest_path.is_file():
            raise ValueError(
                f"Source instance '{instance.id}' has no readable manifest at '{manifest_path}'"
            )


def load_source_instance_index(path: Path | str) -> SourceInstanceIndex:
    """Load a catalog TOML file and build its merged source-instance index."""
    path = Path(path)
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    return build_source_instance_index(CatalogConfig(**raw))
