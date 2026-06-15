"""Cached catalog loading and test override helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from shared.catalog.loader import load_catalog
from shared.catalog.models import KBConfig, TaskConfig
from shared.catalog.paths import CatalogPathSettings, resolve_catalog_path


@lru_cache(maxsize=None)
def _load_catalog_cached(path: str) -> tuple[dict[str, TaskConfig], dict[str, KBConfig]]:
    """Cached path-based loader for the materialized runtime catalog."""
    return load_catalog(path)


_CATALOG_OVERRIDE: dict[str, TaskConfig] | None = None
_CATALOG_INDEX_OVERRIDE: dict[str, KBConfig] | None = None


def build_kb_index(catalog: dict[str, TaskConfig]) -> dict[str, KBConfig]:
    """Build a flat KB index from a task-scoped runtime catalog."""
    index: dict[str, KBConfig] = {}
    for task_cfg in catalog.values():
        for kb_cfg in task_cfg.knowledge_bases:
            if kb_cfg.name in index:
                if index[kb_cfg.name] is not kb_cfg:
                    raise ValueError(
                        f"Duplicate KB name '{kb_cfg.name}' found across tasks. "
                        f"KB names must be unique."
                    )
                continue
            index[kb_cfg.name] = kb_cfg
    return index


def _restore_catalog_override(
    catalog: dict[str, TaskConfig] | None,
    index: dict[str, KBConfig] | None,
) -> None:
    global _CATALOG_OVERRIDE, _CATALOG_INDEX_OVERRIDE  # noqa: PLW0603

    _CATALOG_OVERRIDE = catalog
    _CATALOG_INDEX_OVERRIDE = index


def set_catalog_override(
    catalog: dict[str, TaskConfig],
    *,
    index: dict[str, KBConfig] | None = None,
) -> None:
    """Install an in-memory catalog override used ahead of disk-backed loading."""
    _restore_catalog_override(catalog, index if index is not None else build_kb_index(catalog))


def clear_catalog_override() -> None:
    """Remove any installed in-memory catalog override."""
    _restore_catalog_override(None, None)


@contextmanager
def catalog_override(
    catalog: dict[str, TaskConfig],
    *,
    index: dict[str, KBConfig] | None = None,
) -> Iterator[None]:
    """Temporarily install an in-memory catalog override and restore prior state."""
    previous_catalog = _CATALOG_OVERRIDE
    previous_index = _CATALOG_INDEX_OVERRIDE
    set_catalog_override(catalog, index=index)
    try:
        yield
    finally:
        _restore_catalog_override(previous_catalog, previous_index)


def get_loaded_catalog_state(
    *, settings: CatalogPathSettings | None = None
) -> tuple[dict[str, TaskConfig], dict[str, KBConfig]]:
    """Return the effective runtime catalog and flat index."""
    if _CATALOG_OVERRIDE is not None and _CATALOG_INDEX_OVERRIDE is not None:
        return _CATALOG_OVERRIDE, _CATALOG_INDEX_OVERRIDE

    if settings is None:
        from shared.config import get_settings

        settings = get_settings().catalog

    path = resolve_catalog_path(settings).resolve()
    return _load_catalog_cached(str(path))


def get_catalog(*, settings: CatalogPathSettings | None = None) -> dict[str, TaskConfig]:
    """Return the task catalog (cached after first call)."""
    catalog, _ = get_loaded_catalog_state(settings=settings)
    return catalog


def get_kb_config(
    kb_name: str,
    *,
    settings: CatalogPathSettings | None = None,
) -> KBConfig | None:
    """Look up a KB by name."""
    _, index = get_loaded_catalog_state(settings=settings)
    return index.get(kb_name)


def get_kb_names(*, settings: CatalogPathSettings | None = None) -> list[str]:
    """Flat list of all KB names across all tasks."""
    _, index = get_loaded_catalog_state(settings=settings)
    return list(index.keys())


def clear_catalog_caches() -> None:
    """Clear disk-backed cache and in-memory overrides for the catalog."""
    clear_catalog_override()
    _load_catalog_cached.cache_clear()
