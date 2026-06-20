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
from app_config.catalog.loader import (
    load_catalog,
    load_catalog_with_source_index,
    materialize_catalog,
    read_catalog_config,
)
from app_config.catalog.models import AdapterConfig, AliasConfig, KBConfig, TaskConfig
from app_config.catalog.paths import CatalogPathSettings, resolve_catalog_path
from app_config.catalog.schema import (
    BenchmarkAdapterConfig,
    BenchmarkSourceConfig,
    CatalogAliasConfig,
    CatalogConfig,
    CatalogKBConfig,
    CatalogTaskConfig,
    SourceAdapterConfig,
    SourceInstanceAdapterRef,
    SourceInstanceConfig,
    SourceInstanceRole,
)
from app_config.catalog.source_instances import (
    SourceInstanceIndex,
    build_source_instance_index,
    conventional_manifest_path,
    load_source_instance_index,
    resolve_corpus_source_instance_ids,
    validate_source_instance_manifests_exist,
)
from app_config.catalog.validation import validate_kb_alias

__all__ = [
    "AdapterConfig",
    "AliasConfig",
    "BenchmarkAdapterConfig",
    "BenchmarkSourceConfig",
    "CatalogAliasConfig",
    "CatalogConfig",
    "CatalogKBConfig",
    "CatalogPathSettings",
    "CatalogTaskConfig",
    "KBConfig",
    "SourceAdapterConfig",
    "SourceInstanceAdapterRef",
    "SourceInstanceConfig",
    "SourceInstanceIndex",
    "SourceInstanceRole",
    "TaskConfig",
    "build_kb_index",
    "build_source_instance_index",
    "catalog_override",
    "clear_catalog_caches",
    "clear_catalog_override",
    "conventional_manifest_path",
    "get_catalog",
    "get_kb_config",
    "get_kb_names",
    "load_catalog",
    "load_catalog_with_source_index",
    "load_source_instance_index",
    "materialize_catalog",
    "read_catalog_config",
    "resolve_corpus_source_instance_ids",
    "resolve_catalog_path",
    "set_catalog_override",
    "validate_kb_alias",
    "validate_source_instance_manifests_exist",
]
