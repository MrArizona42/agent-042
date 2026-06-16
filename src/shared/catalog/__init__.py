"""Compatibility exports for the application catalog.

New production code should import from ``app_config.catalog``. This package
remains as a stable compatibility path while the config split is migrated.
"""

from app_config.catalog import (
    AdapterConfig,
    AliasConfig,
    CatalogAliasConfig,
    CatalogConfig,
    CatalogKBConfig,
    CatalogPathSettings,
    CatalogTaskConfig,
    KBConfig,
    SourceConfig,
    SourceIngestAdapterConfig,
    TaskConfig,
    build_kb_index,
    catalog_override,
    clear_catalog_caches,
    clear_catalog_override,
    get_catalog,
    get_kb_config,
    get_kb_names,
    load_catalog,
    materialize_catalog,
    resolve_catalog_path,
    set_catalog_override,
    validate_kb_alias,
)

__all__ = [
    "AdapterConfig",
    "AliasConfig",
    "CatalogAliasConfig",
    "CatalogConfig",
    "CatalogKBConfig",
    "CatalogPathSettings",
    "CatalogTaskConfig",
    "KBConfig",
    "SourceConfig",
    "SourceIngestAdapterConfig",
    "TaskConfig",
    "build_kb_index",
    "catalog_override",
    "clear_catalog_caches",
    "clear_catalog_override",
    "get_catalog",
    "get_kb_config",
    "get_kb_names",
    "load_catalog",
    "materialize_catalog",
    "resolve_catalog_path",
    "set_catalog_override",
    "validate_kb_alias",
]
