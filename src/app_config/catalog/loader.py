"""Catalog TOML parsing and runtime materialization."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TypeVar

from app_config.catalog.models import AliasConfig, KBConfig, TaskConfig
from app_config.catalog.schema import (
    CatalogConfig,
    CatalogKBConfig,
    CatalogTaskConfig,
    SourceConfig,
)

_CatalogItem = TypeVar("_CatalogItem", CatalogTaskConfig, CatalogKBConfig, SourceConfig)


def _index_by_id(items: list[_CatalogItem], section: str) -> dict[str, _CatalogItem]:
    """Return items keyed by id, preserving TOML order and rejecting duplicates."""
    index: dict[str, _CatalogItem] = {}
    for item in items:
        item_id = item.id.strip()
        if not item_id:
            raise ValueError(f"{section} entries require non-empty id")
        if item_id in index:
            raise ValueError(f"Duplicate {section} id '{item_id}'")
        index[item_id] = item
    return index


def materialize_catalog(
    catalog_cfg: CatalogConfig,
) -> tuple[dict[str, TaskConfig], dict[str, KBConfig]]:
    """Build the runtime task catalog and flat KB index from TOML schema."""
    catalog_kbs = _index_by_id(catalog_cfg.knowledge_bases, "knowledge_bases")
    catalog_tasks = _index_by_id(catalog_cfg.tasks, "tasks")
    source_keys: set[tuple[str, str]] = set()
    for source_cfg in catalog_cfg.sources:
        source_kb = source_cfg.kb.strip()
        source_id = source_cfg.id.strip()
        if not source_kb or not source_id:
            raise ValueError("sources entries require non-empty kb and id")
        if source_kb not in catalog_kbs:
            raise ValueError(f"Source '{source_id}' references unknown KB '{source_kb}'")
        source_key = (source_kb, source_id)
        if source_key in source_keys:
            raise ValueError(f"Duplicate source id '{source_id}' for KB '{source_kb}'")
        source_keys.add(source_key)

    kb_index: dict[str, KBConfig] = {}
    for kb_name, kb_cfg in catalog_kbs.items():
        if not kb_cfg.enabled:
            continue

        aliases = {
            alias_name: AliasConfig(
                top_k=alias_cfg.top_k,
                score_threshold=alias_cfg.score_threshold,
                reranker=alias_cfg.reranker,
                retrieval_strategy=alias_cfg.retrieval_strategy,
                reranker_multiplier=alias_cfg.reranker_multiplier,
            )
            for alias_name, alias_cfg in kb_cfg.aliases.items()
        }

        kb_index[kb_name] = KBConfig(
            name=kb_name,
            default_alias=kb_cfg.default_alias,
            aliases=aliases,
            update_strategy=kb_cfg.update_strategy,
            label=kb_cfg.label,
            description=kb_cfg.description,
            selection_description=kb_cfg.selection_description,
        )

    task_catalog: dict[str, TaskConfig] = {}
    for task_name, task_cfg in catalog_tasks.items():
        if not task_cfg.enabled:
            continue

        task_knowledge_bases: list[KBConfig] = []
        for kb_name in task_cfg.kb_refs:
            if kb_name not in catalog_kbs:
                raise ValueError(f"Task '{task_name}' references unknown KB '{kb_name}'")
            kb_runtime_cfg = kb_index.get(kb_name)
            if kb_runtime_cfg is not None:
                task_knowledge_bases.append(kb_runtime_cfg)

        task_catalog[task_name] = TaskConfig(
            task=task_name,
            label=task_cfg.label,
            routing_description=task_cfg.routing_description,
            adapter=task_cfg.adapter.model_copy(deep=True),
            knowledge_bases=task_knowledge_bases,
        )

    return task_catalog, kb_index


def load_catalog(path: Path | str) -> tuple[dict[str, TaskConfig], dict[str, KBConfig]]:
    """Load the catalog from a TOML file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Catalog config file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Catalog config path is not a file: {path}")

    if path.suffix.lower() != ".toml":
        raise ValueError(f"Catalog must be a TOML file (got '{path.name}')")

    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    return materialize_catalog(CatalogConfig(**raw))
