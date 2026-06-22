"""Raw source extraction into LlamaIndex documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from bs4 import BeautifulSoup
from llama_index.core import Document
from pypdf import PdfReader

from rag.sources.fetchers import SourceFetchResult


class SourceExtractor(Protocol):
    """Extract one fetched raw artifact into a LlamaIndex document."""

    def extract(self, fetch_result: SourceFetchResult) -> Document: ...


class HtmlDocsExtractor:
    extraction_method = "html_bs4"

    def extract(self, fetch_result: SourceFetchResult) -> Document:
        source = fetch_result.source_document
        html = Path(fetch_result.raw_path).read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")
        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        sections: list[dict[str, Any]] = []
        current_title: str | None = None
        current_level: int | None = None
        current_parts: list[str] = []

        def flush_section() -> None:
            nonlocal current_parts
            text = "\n".join(part for part in current_parts if part.strip()).strip()
            if text:
                sections.append(
                    {
                        "title": current_title,
                        "text": text,
                        "level": current_level,
                        "ordinal": len(sections),
                        "metadata": {},
                    }
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

        full_text = "\n\n".join(str(section["text"]) for section in sections).strip()
        if not full_text:
            full_text = " ".join(body.get_text(" ", strip=True).split())
        return _extracted_document(
            source,
            text=full_text,
            sections=sections,
            extraction_method=self.extraction_method,
            raw_path=fetch_result.raw_path,
            checksum=fetch_result.checksum,
        )


class ArxivPdfExtractor:
    extraction_method = "pdf_pypdf"

    def extract(self, fetch_result: SourceFetchResult) -> Document:
        source = fetch_result.source_document
        warnings: list[str] = []
        sections: list[dict[str, Any]] = []
        for page_index, page in enumerate(PdfReader(str(fetch_result.raw_path)).pages):
            text = (page.extract_text() or "").strip()
            if not text:
                warnings.append(f"Page {page_index + 1} produced no text")
                continue
            sections.append(
                {
                    "title": f"Page {page_index + 1}",
                    "text": text,
                    "level": 1,
                    "ordinal": page_index,
                    "metadata": {},
                }
            )

        full_text = "\n\n".join(str(section["text"]) for section in sections).strip()
        return _extracted_document(
            source,
            text=full_text or str(source.metadata["title"]),
            sections=sections,
            extraction_method=self.extraction_method,
            extraction_warnings=warnings,
            raw_path=fetch_result.raw_path,
            checksum=fetch_result.checksum,
        )


def _extracted_document(
    source: Document,
    *,
    text: str,
    sections: list[dict[str, Any]],
    extraction_method: str,
    raw_path: Path,
    checksum: str,
    extraction_warnings: list[str] | None = None,
) -> Document:
    return Document(
        text=text,
        id_=source.id_,
        metadata={
            **source.metadata,
            "sections": sections,
            "extraction_method": extraction_method,
            "extraction_warnings": extraction_warnings or [],
            "raw_path": raw_path.as_posix(),
            "checksum": checksum,
        },
    )
