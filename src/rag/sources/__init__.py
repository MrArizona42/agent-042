"""RAG source manifest contracts and connector registry."""

from rag.sources.connectors import (
    DEFAULT_SOURCE_CONNECTORS,
    ManifestOnlyConnector,
    SourceConnector,
    SourceConnectorRegistry,
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
    "HtmlDocsEntry",
    "ManifestOnlyConnector",
    "SourceConnector",
    "SourceConnectorRegistry",
    "SourceManifest",
    "SourceType",
    "load_source_manifest",
    "source_manifest_from_raw",
]
