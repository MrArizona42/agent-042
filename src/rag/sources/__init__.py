"""RAG source manifest contracts and connector registry."""

from rag.sources.connectors import (
    DEFAULT_SOURCE_CONNECTORS,
    ManifestOnlyConnector,
    SourceConnector,
    SourceConnectorRegistry,
)
from rag.sources.extractors import ArxivPdfExtractor, HtmlDocsExtractor, SourceExtractor
from rag.sources.fetchers import (
    ArxivPaperFetcher,
    HtmlDocsFetcher,
    SourceFetcher,
    SourceFetchResult,
)
from rag.sources.manifests import load_source_manifest
from rag.sources.models import (
    ArxivPaperEntry,
    HtmlDocsEntry,
    SourceManifest,
    SourceType,
    source_manifest_from_raw,
)

__all__ = [
    "DEFAULT_SOURCE_CONNECTORS",
    "ArxivPaperEntry",
    "ArxivPaperFetcher",
    "ArxivPdfExtractor",
    "HtmlDocsEntry",
    "HtmlDocsFetcher",
    "HtmlDocsExtractor",
    "ManifestOnlyConnector",
    "SourceConnector",
    "SourceConnectorRegistry",
    "SourceExtractor",
    "SourceFetcher",
    "SourceFetchResult",
    "SourceManifest",
    "SourceType",
    "load_source_manifest",
    "source_manifest_from_raw",
]
