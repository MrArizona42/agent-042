"""RAG source manifest contracts and connector registry."""

from rag.sources.artifacts import (
    ExtractedDocumentArtifact,
    ExtractionArtifactMeta,
    RawArtifactRef,
    extracted_artifact_from_result,
    extracted_artifact_path,
    read_extracted_artifact,
    write_extracted_artifact,
)
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
    "ExtractedDocumentArtifact",
    "ExtractionArtifactMeta",
    "HtmlDocsEntry",
    "HtmlDocsFetcher",
    "HtmlDocsExtractor",
    "ManifestOnlyConnector",
    "RawArtifactRef",
    "SourceConnector",
    "SourceConnectorRegistry",
    "SourceExtractor",
    "SourceFetcher",
    "SourceFetchResult",
    "SourceManifest",
    "SourceType",
    "extracted_artifact_from_result",
    "extracted_artifact_path",
    "load_source_manifest",
    "read_extracted_artifact",
    "source_manifest_from_raw",
    "write_extracted_artifact",
]
