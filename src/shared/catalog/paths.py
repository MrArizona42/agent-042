"""Catalog path resolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from shared.local_env import get_repo_root

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[3] / "catalog.toml"
CATALOG_PATH_ENV = "CATALOG__PATH"


class CatalogPathSettings(Protocol):
    """Structural type for settings objects that provide catalog path overrides."""

    path: Path | None


def configured_catalog_path(settings: CatalogPathSettings | None = None) -> Path | None:
    """Resolve the configured catalog path before path normalization."""
    if settings is not None:
        return settings.path

    raw_value = os.environ.get(CATALOG_PATH_ENV)
    if raw_value is None:
        return None

    stripped = raw_value.strip()
    if not stripped:
        return None
    return Path(stripped)


def resolve_catalog_path(settings: CatalogPathSettings | None = None) -> Path:
    """Resolve the active catalog path."""
    configured_path = configured_catalog_path(settings)
    if configured_path is None:
        return DEFAULT_CATALOG_PATH

    path = configured_path.expanduser()
    if path.is_absolute():
        return path

    try:
        return get_repo_root() / path
    except FileNotFoundError:
        return path
