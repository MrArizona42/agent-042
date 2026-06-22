"""Built-in catalog-declared source adapter factories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from llama_index.core import Document

from rag.adapters.capabilities import AdapterCapability, SourceAdapterContext
from rag.contracts.metadata import source_document


class SourceFetcherFactory(Protocol):
    def __call__(self): ...


class SourceExtractorFactory(Protocol):
    def __call__(self): ...


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
    """Adapter for the generic manifest-backed source contract."""

    adapter_id: str
    version: str
    default_uri_prefix: str
    _fetcher_factory: SourceFetcherFactory = field(repr=False)
    _extractor_factory: SourceExtractorFactory = field(repr=False)
    capabilities: frozenset[AdapterCapability] = frozenset({"source"})

    def validate_manifest(self, manifest: Any) -> Any:
        """Return a validated generic manifest."""
        return manifest

    def list_documents(
        self,
        manifest: Any,
        *,
        context: SourceAdapterContext,
    ) -> list[Document]:
        """Convert manifest entries into identity-complete LlamaIndex documents."""
        validated = self.validate_manifest(manifest)
        return [
            source_document(
                local_document_id=entry.id,
                title=entry.title,
                source_uri=entry.uri or entry.url or f"{self.default_uri_prefix}:{entry.id}",
                metadata=entry.metadata,
                kb_id=context.kb_id,
                source_instance_id=context.source_instance_id,
                adapter_id=self.adapter_id,
                adapter_version=self.version,
                manifest_digest=context.manifest_digest,
            )
            for entry in validated.documents
        ]

    def fetcher(self):
        return self._fetcher_factory()

    def extractor(self):
        return self._extractor_factory()


def make_http_html_adapter() -> ManifestSourceAdapter:
    return ManifestSourceAdapter(
        adapter_id="generic.http_html",
        version="1",
        default_uri_prefix="http_html",
        _fetcher_factory=_html_docs_fetcher,
        _extractor_factory=_html_docs_extractor,
    )


def make_arxiv_paper_adapter() -> ManifestSourceAdapter:
    return ManifestSourceAdapter(
        adapter_id="generic.arxiv_paper",
        version="1",
        default_uri_prefix="arxiv",
        _fetcher_factory=_arxiv_paper_fetcher,
        _extractor_factory=_arxiv_paper_extractor,
    )
