"""Tests for catalog-declared adapter factory loading (Phase 2)."""

from __future__ import annotations

import pytest

from app_config.catalog.schema import BenchmarkAdapterConfig, CatalogConfig, SourceAdapterConfig
from rag.ingest.adapters import AdapterCapability, ManifestSourceAdapter


def _mismatched_capability_adapter() -> ManifestSourceAdapter:
    """A source-only adapter wrongly declared as a benchmark adapter."""
    return ManifestSourceAdapter(
        adapter_id="test.mismatched",
        version="1",
        source_type="html_docs",
        _fetcher_factory=lambda: None,
        _extractor_factory=lambda: None,
        capabilities=frozenset({"source"}),
    )


def _wrong_identity_adapter() -> ManifestSourceAdapter:
    """An adapter whose own id/version don't match its declared catalog entry."""
    return ManifestSourceAdapter(
        adapter_id="generic.http_html",
        version="2",
        source_type="html_docs",
        _fetcher_factory=lambda: None,
        _extractor_factory=lambda: None,
    )


class _FakeBenchmarkAdapter:
    adapter_id = "test.benchmark"
    version = "1"
    source_type = "qa"
    capabilities: frozenset[AdapterCapability] = frozenset({"source", "benchmark"})

    def validate_manifest(self, manifest):
        return manifest

    def list_documents(self, manifest):
        return []

    def fetcher(self):
        return None

    def extractor(self):
        return None

    def prepare_benchmark(self, manifest):
        return None


def _benchmark_adapter() -> _FakeBenchmarkAdapter:
    return _FakeBenchmarkAdapter()


class TestImportFactory:
    def test_imports_existing_factory(self):
        from rag.ingest.adapter_loading import import_factory

        factory = import_factory("rag.ingest.adapters:make_http_html_adapter")
        adapter = factory()
        assert adapter.adapter_id == "generic.http_html"

    def test_rejects_malformed_reference(self):
        from rag.ingest.adapter_loading import import_factory

        with pytest.raises(ValueError, match="must be in 'module:function' form"):
            import_factory("rag.ingest.adapters.make_http_html_adapter")

    def test_rejects_unimportable_module(self):
        from rag.ingest.adapter_loading import import_factory

        with pytest.raises(ValueError, match="Cannot import adapter factory module"):
            import_factory("rag.ingest.no_such_module:make_thing")

    def test_rejects_missing_callable(self):
        from rag.ingest.adapter_loading import import_factory

        with pytest.raises(ValueError, match="no callable 'no_such_function'"):
            import_factory("rag.ingest.adapters:no_such_function")


class TestLoadAdapter:
    def test_loads_and_validates_source_adapter(self):
        from rag.ingest.adapter_loading import load_adapter

        config = SourceAdapterConfig(
            id="generic.http_html",
            version="1",
            description="Fetches HTML pages.",
            factory="rag.ingest.adapters:make_http_html_adapter",
        )
        adapter = load_adapter(config, required_capabilities=frozenset({"source"}))
        assert adapter.adapter_id == "generic.http_html"

    def test_factory_raising_is_wrapped(self):
        from rag.ingest.adapter_loading import load_adapter

        config = SourceAdapterConfig(
            id="x",
            version="1",
            description="d",
            factory="rag.ingest.adapters:make_http_html_adapter",
        )

        def _boom():
            raise RuntimeError("boom")

        import rag.ingest.adapters as adapters_module

        original = adapters_module.make_http_html_adapter
        adapters_module.make_http_html_adapter = _boom
        try:
            with pytest.raises(ValueError, match="raised: boom"):
                load_adapter(config, required_capabilities=frozenset({"source"}))
        finally:
            adapters_module.make_http_html_adapter = original

    def test_rejects_identity_mismatch(self):
        from rag.ingest.adapter_loading import load_adapter

        config = SourceAdapterConfig(
            id="generic.http_html",
            version="1",
            description="d",
            factory="tests.rag.test_adapter_loading:_wrong_identity_adapter",
        )
        with pytest.raises(ValueError, match="returned an adapter identified as"):
            load_adapter(config, required_capabilities=frozenset({"source"}))

    def test_rejects_capability_mismatch(self):
        from rag.ingest.adapter_loading import load_adapter

        config = BenchmarkAdapterConfig(
            id="test.mismatched",
            version="1",
            description="d",
            factory="tests.rag.test_adapter_loading:_mismatched_capability_adapter",
        )
        with pytest.raises(ValueError, match="missing required capabilities"):
            load_adapter(config, required_capabilities=frozenset({"source", "benchmark"}))

    def test_benchmark_adapter_requires_prepare_benchmark_method(self):
        from rag.ingest.adapter_loading import load_adapter

        config = BenchmarkAdapterConfig(
            id="test.benchmark",
            version="1",
            description="d",
            factory="tests.rag.test_adapter_loading:_benchmark_adapter",
        )
        adapter = load_adapter(config, required_capabilities=frozenset({"source", "benchmark"}))
        assert adapter.prepare_benchmark(None) is None


class TestBuildCatalogAdapterRegistry:
    def test_builds_registry_from_declared_adapters(self):
        from rag.ingest.adapter_loading import build_catalog_adapter_registry

        catalog_cfg = CatalogConfig(
            source_adapters=[
                SourceAdapterConfig(
                    id="generic.http_html",
                    version="1",
                    description="d",
                    factory="rag.ingest.adapters:make_http_html_adapter",
                )
            ],
            benchmark_adapters=[
                BenchmarkAdapterConfig(
                    id="test.benchmark",
                    version="1",
                    description="d",
                    factory="tests.rag.test_adapter_loading:_benchmark_adapter",
                )
            ],
        )
        registry = build_catalog_adapter_registry(catalog_cfg)
        assert registry.get("generic.http_html", version="1").adapter_id == "generic.http_html"
        assert registry.get("test.benchmark", version="1").adapter_id == "test.benchmark"

    def test_benchmark_adapter_missing_benchmark_capability_fails_fast(self):
        from rag.ingest.adapter_loading import build_catalog_adapter_registry

        catalog_cfg = CatalogConfig(
            benchmark_adapters=[
                BenchmarkAdapterConfig(
                    id="test.mismatched",
                    version="1",
                    description="d",
                    factory="tests.rag.test_adapter_loading:_mismatched_capability_adapter",
                )
            ],
        )
        with pytest.raises(ValueError, match="missing required capabilities"):
            build_catalog_adapter_registry(catalog_cfg)
