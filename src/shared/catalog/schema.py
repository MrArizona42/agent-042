"""Compatibility re-export — real implementation is in app_config.catalog.schema."""

from app_config.catalog.schema import (
    CatalogAliasConfig,
    CatalogConfig,
    CatalogKBConfig,
    CatalogTaskConfig,
    SourceConfig,
    SourceIngestAdapterConfig,
)

__all__ = [
    "CatalogAliasConfig",
    "CatalogConfig",
    "CatalogKBConfig",
    "CatalogTaskConfig",
    "SourceConfig",
    "SourceIngestAdapterConfig",
]
