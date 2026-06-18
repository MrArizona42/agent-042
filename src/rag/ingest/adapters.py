"""Source ingest adapter contracts and registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from rag.contracts import SourceDocument
from rag.evaluation.models import BenchmarkPreparedArtifacts

AdapterCapability = Literal["source", "benchmark"]


class SourceFetcherFactory(Protocol):
    """Factory for a fetcher implementation."""

    def __call__(self):
        """Return a source fetcher."""
        ...


class SourceExtractorFactory(Protocol):
    """Factory for an extractor implementation."""

    def __call__(self):
        """Return a source extractor."""
        ...


class SourceAdapter(Protocol):
    """Adapter contract for one source ingest family."""

    adapter_id: str
    version: str
    source_type: str
    capabilities: frozenset[AdapterCapability]

    def validate_manifest(self, manifest: Any) -> Any:
        """Validate an already-loaded source manifest for this adapter."""
        ...

    def list_documents(self, manifest: Any) -> list[SourceDocument]:
        """Return platform source documents declared by the manifest."""
        ...

    def fetcher(self):
        """Return the fetcher selected by this adapter."""
        ...

    def extractor(self):
        """Return the extractor selected by this adapter."""
        ...


class BenchmarkAdapter(SourceAdapter, Protocol):
    """A source adapter that also implements benchmark preparation."""

    def prepare_benchmark(self, manifest: Any) -> BenchmarkPreparedArtifacts:
        """Emit normalized benchmark cases and labels for this manifest."""
        ...


def _html_docs_fetcher():
    from rag.sources.fetchers import HtmlDocsFetcher

    return HtmlDocsFetcher()


def _html_docs_extractor():
    from rag.sources.extractors import HtmlDocsExtractor

    return HtmlDocsExtractor()


def _arxiv_paper_fetcher():
    from rag.sources.fetchers import ArxivPaperFetcher

    return ArxivPaperFetcher()


def _arxiv_paper_extractor():
    from rag.sources.extractors import ArxivPdfExtractor

    return ArxivPdfExtractor()


@dataclass(frozen=True, slots=True)
class ManifestSourceAdapter:
    """Adapter for current manifest-backed source families."""

    adapter_id: str
    version: str
    source_type: str
    _fetcher_factory: SourceFetcherFactory = field(repr=False)
    _extractor_factory: SourceExtractorFactory = field(repr=False)
    capabilities: frozenset[AdapterCapability] = frozenset({"source"})

    def validate_manifest(self, manifest: Any) -> Any:
        """Validate that the manifest belongs to this adapter's source type."""
        if manifest.source_type != self.source_type:
            raise ValueError(
                f"Adapter '{self.adapter_id}@{self.version}' expects source_type "
                f"'{self.source_type}' (got '{manifest.source_type}')"
            )
        return manifest

    def list_documents(self, manifest: Any) -> list[SourceDocument]:
        """Convert a validated manifest to platform source documents."""
        return self.validate_manifest(manifest).to_source_documents()

    def fetcher(self):
        """Return the fetcher selected by this adapter."""
        return self._fetcher_factory()

    def extractor(self):
        """Return the extractor selected by this adapter."""
        return self._extractor_factory()


class SourceAdapterRegistry:
    """Registry keyed by adapter id and version."""

    def __init__(self) -> None:
        self._adapters: dict[tuple[str, str], SourceAdapter] = {}

    def register(self, adapter: SourceAdapter) -> None:
        key = (adapter.adapter_id, adapter.version)
        if key in self._adapters:
            raise ValueError(
                f"Source adapter '{adapter.adapter_id}@{adapter.version}' already registered"
            )
        self._adapters[key] = adapter

    def get(self, adapter_id: str, *, version: str = "1") -> SourceAdapter:
        key = (adapter_id, version)
        adapter = self._adapters.get(key)
        if adapter is None:
            raise ValueError(f"Unknown source adapter '{adapter_id}@{version}'")
        return adapter


def make_http_html_adapter() -> ManifestSourceAdapter:
    """Factory for the built-in generic HTTP/HTML source adapter.

    Referenced from catalog `[[source_adapters]]` entries as
    `rag.ingest.adapters:make_http_html_adapter`.
    """
    return ManifestSourceAdapter(
        adapter_id="generic.http_html",
        version="1",
        source_type="html_docs",
        _fetcher_factory=_html_docs_fetcher,
        _extractor_factory=_html_docs_extractor,
    )


def make_arxiv_paper_adapter() -> ManifestSourceAdapter:
    """Factory for the built-in generic arXiv paper source adapter.

    Referenced from catalog `[[source_adapters]]` entries as
    `rag.ingest.adapters:make_arxiv_paper_adapter`.
    """
    return ManifestSourceAdapter(
        adapter_id="generic.arxiv_paper",
        version="1",
        source_type="arxiv_paper",
        _fetcher_factory=_arxiv_paper_fetcher,
        _extractor_factory=_arxiv_paper_extractor,
    )
