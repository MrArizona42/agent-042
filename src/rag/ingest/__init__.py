"""Generic RAG ingest lifecycle."""

from rag.ingest.adapter_loading import build_catalog_adapter_registry, import_factory, load_adapter
from rag.ingest.adapters import (
    AdapterCapability,
    BenchmarkAdapter,
    ManifestSourceAdapter,
    SourceAdapter,
    SourceAdapterRegistry,
    make_arxiv_paper_adapter,
    make_http_html_adapter,
)

__all__ = [
    "AdapterCapability",
    "BenchmarkAdapter",
    "ManifestSourceAdapter",
    "SourceAdapter",
    "SourceAdapterRegistry",
    "build_catalog_adapter_registry",
    "import_factory",
    "load_adapter",
    "make_arxiv_paper_adapter",
    "make_http_html_adapter",
]
