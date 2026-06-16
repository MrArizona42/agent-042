"""Compatibility module for catalog cache helpers."""

from shared.catalog.cache import (
    build_kb_index,
    catalog_override,
    clear_catalog_caches,
    clear_catalog_override,
    get_catalog,
    get_kb_config,
    get_kb_names,
    set_catalog_override,
)

__all__ = [
    "build_kb_index",
    "catalog_override",
    "clear_catalog_caches",
    "clear_catalog_override",
    "get_catalog",
    "get_kb_config",
    "get_kb_names",
    "set_catalog_override",
]
