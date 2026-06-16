"""Tests for the application catalog import surface."""

from __future__ import annotations

from tests.catalog_samples import write_chat_and_code_catalog


class _CatalogSettings:
    def __init__(self, path):
        self.path = path


def test_app_config_catalog_facade_matches_legacy_catalog(tmp_path):
    from app_config.catalog import AliasConfig, SourceConfig, get_kb_names, load_catalog
    from shared.catalog import AliasConfig as LegacyAliasConfig
    from shared.catalog import SourceConfig as LegacySourceConfig
    from shared.catalog import get_kb_names as legacy_get_kb_names
    from shared.catalog import load_catalog as legacy_load_catalog

    catalog_path = write_chat_and_code_catalog(tmp_path / "catalog.toml")

    assert AliasConfig is LegacyAliasConfig
    assert SourceConfig is LegacySourceConfig
    assert load_catalog(catalog_path) == legacy_load_catalog(catalog_path)
    assert get_kb_names(settings=_CatalogSettings(catalog_path)) == legacy_get_kb_names(
        settings=_CatalogSettings(catalog_path)
    )
