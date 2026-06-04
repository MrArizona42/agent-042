"""Catalog validation helpers."""

from __future__ import annotations

from shared.catalog.cache import get_kb_config, get_kb_names
from shared.catalog.paths import CatalogPathSettings


def validate_kb_alias(
    kb: str,
    alias: str | None = None,
    *,
    settings: CatalogPathSettings | None = None,
) -> None:
    """Raise ValueError with a consistent message if kb or alias is unknown."""
    kb_cfg = get_kb_config(kb, settings=settings)
    if kb_cfg is None:
        raise ValueError(f"KB '{kb}' not found. Available: {get_kb_names(settings=settings)}")
    if alias is not None and alias not in kb_cfg.aliases:
        raise ValueError(
            f"Alias '{alias}' not valid for KB '{kb}'. Available: {list(kb_cfg.aliases.keys())}"
        )
