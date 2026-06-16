"""Compatibility module for TOML-backed catalog schema models."""

from shared.catalog.schema import (
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
