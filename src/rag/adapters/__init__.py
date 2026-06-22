"""Catalog-declared source and benchmark adapters."""

from rag.adapters.capabilities import (
    AdapterCapability,
    BenchmarkAdapter,
    SourceAdapter,
    SourceAdapterContext,
    SourceAdapterRegistry,
)
from rag.adapters.loading import build_catalog_adapter_registry, import_factory, load_adapter
from rag.adapters.sources import (
    ManifestSourceAdapter,
    make_arxiv_paper_adapter,
    make_http_html_adapter,
)

__all__ = [
    "AdapterCapability",
    "BenchmarkAdapter",
    "ManifestSourceAdapter",
    "SourceAdapter",
    "SourceAdapterContext",
    "SourceAdapterRegistry",
    "build_catalog_adapter_registry",
    "import_factory",
    "load_adapter",
    "make_arxiv_paper_adapter",
    "make_http_html_adapter",
]
