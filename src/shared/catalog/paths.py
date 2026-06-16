"""Compatibility re-export — real implementation is in app_config.catalog.paths."""

from app_config.catalog.paths import CatalogPathSettings, resolve_catalog_path

__all__ = ["CatalogPathSettings", "resolve_catalog_path"]
