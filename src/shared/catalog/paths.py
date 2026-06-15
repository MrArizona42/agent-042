"""Catalog path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class CatalogPathSettings(Protocol):
    """Structural type for settings objects that provide a catalog path."""

    path: Path


def configured_catalog_path(settings: CatalogPathSettings | None = None) -> Path:
    """Return the configured catalog path before path normalization."""
    if settings is None:
        raise RuntimeError("Catalog settings are required; set CONFIG__CATALOG_PATH")

    return settings.path


def resolve_catalog_path(settings: CatalogPathSettings | None = None) -> Path:
    """Resolve the explicit active catalog path."""
    configured_path = configured_catalog_path(settings)

    path = configured_path.expanduser()
    if path.is_absolute():
        return path

    return Path.cwd() / path
