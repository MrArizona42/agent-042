"""Compatibility re-export — real implementation is in app_config.catalog.loader."""

from app_config.catalog.loader import load_catalog, materialize_catalog

__all__ = ["load_catalog", "materialize_catalog"]
