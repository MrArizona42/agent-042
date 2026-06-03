"""Source connector interfaces and registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag.domain import SourceDocument
from rag.sources.models import SourceManifest, SourceType


class SourceConnector(Protocol):
    """Connector contract for one source type."""

    source_type: SourceType

    def list_documents(self, manifest: SourceManifest) -> list[SourceDocument]:
        """Return source document contracts declared by a manifest."""
        ...


@dataclass(frozen=True, slots=True)
class ManifestOnlyConnector:
    """Connector for reviewed manifests before network fetch is implemented."""

    source_type: SourceType

    def list_documents(self, manifest: SourceManifest) -> list[SourceDocument]:
        if manifest.source_type != self.source_type:
            raise ValueError(
                f"Connector '{self.source_type}' cannot load manifest "
                f"with source_type '{manifest.source_type}'"
            )
        return manifest.to_source_documents()


class SourceConnectorRegistry:
    """Small source connector registry keyed by source type."""

    def __init__(self) -> None:
        self._connectors: dict[SourceType, SourceConnector] = {}

    def register(self, connector: SourceConnector) -> None:
        if connector.source_type in self._connectors:
            raise ValueError(f"Source connector '{connector.source_type}' already registered")
        self._connectors[connector.source_type] = connector

    def get(self, source_type: SourceType) -> SourceConnector:
        connector = self._connectors.get(source_type)
        if connector is None:
            raise ValueError(f"Unknown source connector '{source_type}'")
        return connector

    @classmethod
    def with_defaults(cls) -> "SourceConnectorRegistry":
        registry = cls()
        registry.register(ManifestOnlyConnector("arxiv_paper"))
        registry.register(ManifestOnlyConnector("html_docs"))
        return registry


DEFAULT_SOURCE_CONNECTORS = SourceConnectorRegistry.with_defaults()
