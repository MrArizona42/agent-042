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


def legacy_source_instance_id(*, kb_id: str, local_source_id: str) -> str:
    """Return the global source instance id derived from a legacy `(kb, id)` pair."""
    return f"{kb_id}.{local_source_id}"


def _normalize_legacy_source(source: SourceConfig) -> SourceInstanceConfig:
    """Project a legacy `[[sources]]` entry into the new source-instance shape."""
    return SourceInstanceConfig(
        id=legacy_source_instance_id(kb_id=source.kb, local_source_id=source.id),
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


def _selector_aliases(instance: SourceInstanceConfig) -> set[str]:
    aliases = {instance.id}
    kb_prefix = f"{instance.knowledge_base}."
    if instance.id.startswith(kb_prefix):
        aliases.add(instance.id.removeprefix(kb_prefix))
    return aliases


def resolve_corpus_source_instance_ids(
    catalog_cfg: CatalogConfig,
    *,
    kb_id: str,
    source_ids: list[str] | None = None,
) -> list[str]:
    """Resolve corpus source-instance ids for a KB, with legacy local-id support."""
    kb_ids = {kb.id for kb in catalog_cfg.knowledge_bases}
    if kb_id not in kb_ids:
        raise ValueError(f"Unknown KB '{kb_id}'")

    index = build_source_instance_index(catalog_cfg)
    corpus_instances = index.corpus_for_kb(kb_id)
    if not corpus_instances:
        raise ValueError(f"Catalog has no corpus source instances for KB '{kb_id}'")

    if source_ids is None:
        return [instance.id for instance in corpus_instances]

    selected = {source_id.strip() for source_id in source_ids if source_id.strip()}
    matched_ids: list[str] = []
    matches_by_selector: dict[str, list[str]] = {selector: [] for selector in selected}

    for instance in corpus_instances:
        aliases = _selector_aliases(instance)
        matching_selectors = selected & aliases
        if matching_selectors:
            matched_ids.append(instance.id)
            for selector in matching_selectors:
                matches_by_selector[selector].append(instance.id)

    ambiguous = {
        selector: matches for selector, matches in matches_by_selector.items() if len(matches) > 1
    }
    if ambiguous:
        raise ValueError(f"Ambiguous source selectors for KB '{kb_id}': {ambiguous}")

    missing = sorted(selector for selector, matches in matches_by_selector.items() if not matches)
    if missing:
        raise ValueError(
            f"Corpus source instances not found for kb_id='{kb_id}' and source_ids={missing}"
        )

    return matched_ids


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


def migrate_legacy_source_manifest(
    *,
    source: SourceConfig,
    catalog_path: Path | str,
    rag_data_root: Path | str,
    force: bool = False,
) -> Path:
    """Copy a legacy `[[sources]].manifest` file to its conventional source-instance path.

    One-time migration helper for moving checked-in manifests ahead of the
    Phase 7 schema flip. Does not mutate the catalog or remove the original
    file; returns the destination path.
    """
    catalog_path = Path(catalog_path)
    manifest_ref = Path(source.manifest)
    source_path = manifest_ref if manifest_ref.is_absolute() else catalog_path.parent / manifest_ref
    if not source_path.is_file():
        raise FileNotFoundError(f"Legacy manifest not found: {source_path}")

    destination = conventional_manifest_path(
        rag_data_root,
        legacy_source_instance_id(kb_id=source.kb, local_source_id=source.id),
    )
    if destination.exists() and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source_path.read_bytes())
    return destination
