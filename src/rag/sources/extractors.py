"""Source extraction helpers producing RAG domain contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from bs4 import BeautifulSoup
from pypdf import PdfReader

from rag.domain import DocumentSection, ExtractedDocument
from rag.sources.fetchers import SourceFetchResult


class SourceExtractor(Protocol):
    """Extract source text from a fetched raw artifact."""

    def extract(self, fetch_result: SourceFetchResult) -> ExtractedDocument:
        """Extract a domain document from a fetched source."""
        ...


class HtmlDocsExtractor:
    """Extract text and heading sections from cached HTML documentation."""

    extraction_method = "html_bs4"

    def extract(self, fetch_result: SourceFetchResult) -> ExtractedDocument:
        source_document = fetch_result.source_document
        html = Path(fetch_result.raw_path).read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")
        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        sections: list[DocumentSection] = []
        current_title: str | None = None
        current_level: int | None = None
        current_parts: list[str] = []

        def flush_section() -> None:
            nonlocal current_parts
            text = "\n".join(part for part in current_parts if part.strip()).strip()
            if text:
                sections.append(
                    DocumentSection(
                        title=current_title,
                        text=text,
                        level=current_level,
                        ordinal=len(sections),
                    )
                )
            current_parts = []

        body = soup.body or soup
        for element in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "code"]):
            name = element.name or ""
            text = " ".join(element.get_text(" ", strip=True).split())
            if not text:
                continue
            if name in {"h1", "h2", "h3", "h4"}:
                flush_section()
                current_title = text
                current_level = int(name[1])
            else:
                current_parts.append(text)
        flush_section()

        full_text = "\n\n".join(section.text for section in sections).strip()
        if not full_text:
            full_text = " ".join(body.get_text(" ", strip=True).split())

        return ExtractedDocument(
            id=source_document.id,
            source_document_id=source_document.id,
            text=full_text,
            sections=sections,
            extraction_method=self.extraction_method,
            metadata={
                "title": source_document.title,
                "uri": source_document.uri,
                "raw_path": fetch_result.raw_path.as_posix(),
                "checksum": fetch_result.checksum,
            },
        )


class ArxivPdfExtractor:
    """Extract plain text from a cached ArXiv PDF."""

    extraction_method = "pdf_pypdf"

    def extract(self, fetch_result: SourceFetchResult) -> ExtractedDocument:
        source_document = fetch_result.source_document
        warnings: list[str] = []
        reader = PdfReader(str(fetch_result.raw_path))
        page_texts: list[str] = []
        sections: list[DocumentSection] = []
        for page_index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                warnings.append(f"Page {page_index + 1} produced no text")
                continue
            page_texts.append(text)
            sections.append(
                DocumentSection(
                    title=f"Page {page_index + 1}",
                    text=text,
                    level=1,
                    ordinal=page_index,
                )
            )

        full_text = "\n\n".join(page_texts).strip()
        return ExtractedDocument(
            id=source_document.id,
            source_document_id=source_document.id,
            text=full_text or source_document.title,
            sections=sections,
            extraction_method=self.extraction_method,
            extraction_warnings=warnings,
            metadata={
                "title": source_document.title,
                "uri": source_document.uri,
                "raw_path": fetch_result.raw_path.as_posix(),
                "checksum": fetch_result.checksum,
            },
        )
