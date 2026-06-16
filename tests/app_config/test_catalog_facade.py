"""Tests for the application catalog import surface."""

from __future__ import annotations

import tomllib

from tests.catalog_samples import write_chat_and_code_catalog


class _CatalogSettings:
    def __init__(self, path):
        self.path = path


def test_app_config_catalog_facade_matches_legacy_catalog(tmp_path):
    from app_config.catalog import (
        AliasConfig,
        SourceConfig,
        SourceIngestAdapterConfig,
        get_kb_names,
        load_catalog,
    )
    from shared.catalog import AliasConfig as LegacyAliasConfig
    from shared.catalog import SourceConfig as LegacySourceConfig
    from shared.catalog import SourceIngestAdapterConfig as LegacySourceIngestAdapterConfig
    from shared.catalog import get_kb_names as legacy_get_kb_names
    from shared.catalog import load_catalog as legacy_load_catalog

    catalog_path = write_chat_and_code_catalog(tmp_path / "catalog.toml")

    assert AliasConfig is LegacyAliasConfig
    assert SourceConfig is LegacySourceConfig
    assert SourceIngestAdapterConfig is LegacySourceIngestAdapterConfig
    assert load_catalog(catalog_path) == legacy_load_catalog(catalog_path)
    assert get_kb_names(settings=_CatalogSettings(catalog_path)) == legacy_get_kb_names(
        settings=_CatalogSettings(catalog_path)
    )


def test_source_config_declares_explicit_ingest_adapter(tmp_path):
    from app_config.catalog.schema import CatalogConfig

    catalog_path = write_chat_and_code_catalog(tmp_path / "catalog.toml")
    raw = catalog_path.read_text(encoding="utf-8")

    catalog = CatalogConfig.model_validate(tomllib.loads(raw))
    sources = {(source.kb, source.id): source for source in catalog.sources}

    docs = sources[("pytorch_reference", "docs")]
    assert docs.type == "html_docs"
    assert docs.ingest_adapter is not None
    assert docs.ingest_adapter.id == "generic.http_html"
    assert docs.ingest_adapter.version == "1"
    assert docs.ingest_adapter.settings == {}

    papers = sources[("ml_papers_core", "papers")]
    assert papers.ingest_adapter is not None
    assert papers.ingest_adapter.id == "generic.arxiv_paper"


def test_source_config_defaults_ingest_adapter_for_legacy_catalog_entries():
    from app_config.catalog import SourceConfig

    source = SourceConfig(
        type="html_docs",
        kb="pytorch_reference",
        id="docs",
        manifest="assets/rag_data/pytorch_reference/sources.toml",
    )

    assert source.ingest_adapter is not None
    assert source.ingest_adapter.id == "html_docs"
    assert source.ingest_adapter.version == "legacy"
