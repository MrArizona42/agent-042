"""Application task/knowledge-base/source catalog."""

from app_config.catalog.cache import (
    build_kb_index,
    catalog_override,
    clear_catalog_caches,
    clear_catalog_override,
    get_catalog,
    get_kb_config,
    get_kb_names,
    set_catalog_override,
)
from app_config.catalog.loader import load_catalog, materialize_catalog
from app_config.catalog.models import AdapterConfig, AliasConfig, KBConfig, TaskConfig
from app_config.catalog.paths import CatalogPathSettings, resolve_catalog_path
from app_config.catalog.schema import (
    CatalogAliasConfig,
    CatalogConfig,
    CatalogKBConfig,
    CatalogTaskConfig,
    SourceConfig,
    SourceIngestAdapterConfig,
)
from app_config.catalog.validation import validate_kb_alias

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
