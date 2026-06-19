"""Catalog TOML parsing and runtime materialization."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TypeVar

from app_config.catalog.models import AliasConfig, KBConfig, TaskConfig
from app_config.catalog.schema import CatalogConfig, CatalogKBConfig, CatalogTaskConfig
from app_config.catalog.source_instances import (
    SourceInstanceIndex,
    build_source_instance_index,
)

_CatalogItem = TypeVar("_CatalogItem", CatalogTaskConfig, CatalogKBConfig)


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

    # Validates declared source-instance references as a side effect (duplicate ids,
    # unknown KB/adapter references); the resulting index is available via
    # load_catalog_with_source_index() for callers that need it.
    build_source_instance_index(catalog_cfg)

    kb_index: dict[str, KBConfig] = {}
    for kb_name, kb_cfg in catalog_kbs.items():
        # Runtime KBConfig.aliases stays query-time-only (AliasConfig). The
        # desired `build` block is control-plane/build-pipeline state, not a
        # runtime query concern; consumers needing it read the raw
        # CatalogConfig/CatalogAliasConfig directly.
        aliases = {
            alias_name: AliasConfig(
                top_k=alias_cfg.retrieve.top_k,
                score_threshold=alias_cfg.retrieve.score_threshold,
                reranker=alias_cfg.retrieve.reranker,
                retrieval_strategy=alias_cfg.retrieve.strategy,
                reranker_multiplier=alias_cfg.retrieve.reranker_multiplier,
            )
            for alias_name, alias_cfg in kb_cfg.aliases.items()
        }

        kb_index[kb_name] = KBConfig(
            name=kb_name,
            default_alias=kb_cfg.default_alias,
            aliases=aliases,
            update_strategy=kb_cfg.update_strategy,
            description=kb_cfg.description,
        )

    task_catalog: dict[str, TaskConfig] = {}
    for task_name, task_cfg in catalog_tasks.items():
        task_knowledge_bases: list[KBConfig] = []
        for kb_name in task_cfg.knowledge_bases:
            if kb_name not in catalog_kbs:
                raise ValueError(f"Task '{task_name}' references unknown KB '{kb_name}'")
            task_knowledge_bases.append(kb_index[kb_name])

        task_catalog[task_name] = TaskConfig(
            task=task_name,
            description=task_cfg.description,
            adapter=task_cfg.lora_adapter.model_copy(deep=True),
            knowledge_bases=task_knowledge_bases,
        )

    return task_catalog, kb_index


def _read_catalog_config(path: Path | str) -> CatalogConfig:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Catalog config file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Catalog config path is not a file: {path}")

    if path.suffix.lower() != ".toml":
        raise ValueError(f"Catalog must be a TOML file (got '{path.name}')")

    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    return CatalogConfig(**raw)


def load_catalog(path: Path | str) -> tuple[dict[str, TaskConfig], dict[str, KBConfig]]:
    """Load the catalog from a TOML file."""
    return materialize_catalog(_read_catalog_config(path))


def load_catalog_with_source_index(
    path: Path | str,
) -> tuple[dict[str, TaskConfig], dict[str, KBConfig], SourceInstanceIndex]:
    """Load the catalog along with its merged source-instance index."""
    catalog_cfg = _read_catalog_config(path)
    task_catalog, kb_index = materialize_catalog(catalog_cfg)
    source_index = build_source_instance_index(catalog_cfg)
    return task_catalog, kb_index, source_index
