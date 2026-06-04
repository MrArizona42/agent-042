"""Shared task/knowledge-base/source catalog."""

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
from shared.catalog.loader import load_catalog, materialize_catalog
from shared.catalog.models import AdapterConfig, AliasConfig, KBConfig, TaskConfig
from shared.catalog.paths import (
    CATALOG_PATH_ENV,
    DEFAULT_CATALOG_PATH,
    CatalogPathSettings,
    resolve_catalog_path,
)
from shared.catalog.schema import (
    CatalogAliasConfig,
    CatalogConfig,
    CatalogKBConfig,
    CatalogTaskConfig,
    SourceConfig,
)
from shared.catalog.validation import validate_kb_alias

__all__ = [
    "AdapterConfig",
    "AliasConfig",
    "CATALOG_PATH_ENV",
    "CatalogAliasConfig",
    "CatalogConfig",
    "CatalogKBConfig",
    "CatalogPathSettings",
    "CatalogTaskConfig",
    "DEFAULT_CATALOG_PATH",
    "KBConfig",
    "SourceConfig",
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
