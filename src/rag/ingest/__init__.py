"""Generic RAG ingest lifecycle."""

from rag.ingest.adapters import (
    DEFAULT_SOURCE_ADAPTERS,
    ManifestSourceAdapter,
    SourceAdapter,
    SourceAdapterRegistry,
)

__all__ = [
    "DEFAULT_SOURCE_ADAPTERS",
    "ManifestSourceAdapter",
    "SourceAdapter",
    "SourceAdapterRegistry",
]
