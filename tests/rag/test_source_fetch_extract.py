from __future__ import annotations

from unittest.mock import patch

import httpx

from rag.domain import SourceDocument
from rag.sources.extractors import ArxivPdfExtractor, HtmlDocsExtractor
from rag.sources.fetchers import ArxivPaperFetcher, HtmlDocsFetcher, SourceFetchResult


def _client(content: bytes, *, content_type: str) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=content,
            headers={"content-type": content_type},
            request=request,
        )

    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_html_fetcher_writes_raw_html_and_metadata_immutably(tmp_path) -> None:
    source = SourceDocument(
        id="html:tensors",
        source_type="html_docs",
        uri="https://pytorch.org/docs/stable/tensors.html",
        title="Tensors",
    )
    fetcher = HtmlDocsFetcher(
        client=_client(b"<html><body><h1>Tensors</h1></body></html>", content_type="text/html")
    )

    first = fetcher.fetch(
        source,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        rag_data_root=tmp_path,
    )
    first.raw_path.write_bytes(b"cached")
    second = fetcher.fetch(
        source,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        rag_data_root=tmp_path,
    )
    assert second.raw_path.read_bytes() == b"cached"

    forced = fetcher.fetch(
        source,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        rag_data_root=tmp_path,
        force=True,
    )

    assert first.raw_path.as_posix().endswith("pytorch_reference/raw/docs/html_tensors/page.html")
    assert first.metadata_path.exists()
    assert second.from_cache is True
    assert forced.from_cache is False
    assert forced.raw_path.read_bytes() == b"<html><body><h1>Tensors</h1></body></html>"


def test_arxiv_fetcher_stores_pdf_bytes(tmp_path) -> None:
    source = SourceDocument(
        id="arxiv:1706.03762",
        source_type="arxiv_paper",
        uri="arxiv:1706.03762",
        title="Attention Is All You Need",
        metadata={"arxiv_id": "1706.03762"},
    )
    fetcher = ArxivPaperFetcher(
        client=_client(b"%PDF fake", content_type="application/pdf")
    )

    result = fetcher.fetch(
        source,
        kb_id="ml_papers_core",
        source_instance_id="papers",
        rag_data_root=tmp_path,
    )

    assert result.raw_path.as_posix().endswith(
        "ml_papers_core/raw/papers/arxiv_1706.03762/paper.pdf"
    )
    assert result.raw_path.read_bytes() == b"%PDF fake"
    assert result.source_document.uri == "https://arxiv.org/pdf/1706.03762"
    assert result.checksum.startswith("sha256:")


def test_html_extractor_preserves_heading_sections(tmp_path) -> None:
    raw_path = tmp_path / "page.html"
    raw_path.write_text(
        """
        <html><body>
          <h1>Tensors</h1>
          <p>A tensor is a multidimensional array.</p>
          <h2>Creation</h2>
          <p>Use torch.tensor.</p>
          <pre>x = torch.tensor([1])</pre>
        </body></html>
        """,
        encoding="utf-8",
    )
    fetch_result = SourceFetchResult(
        source_document=SourceDocument(
            id="html:tensors",
            source_type="html_docs",
            uri="https://pytorch.org/docs/stable/tensors.html",
            title="Tensors",
        ),
        raw_path=raw_path,
        metadata_path=tmp_path / "metadata.json",
        checksum="sha256:test",
    )

    extracted = HtmlDocsExtractor().extract(fetch_result)

    assert extracted.source_document_id == "html:tensors"
    assert extracted.extraction_method == "html_bs4"
    assert [section.title for section in extracted.sections] == ["Tensors", "Creation"]
    assert "torch.tensor" in extracted.text


def test_arxiv_pdf_extractor_uses_pypdf_reader(tmp_path) -> None:
    raw_path = tmp_path / "paper.pdf"
    raw_path.write_bytes(b"%PDF fake")
    fetch_result = SourceFetchResult(
        source_document=SourceDocument(
            id="arxiv:1706.03762",
            source_type="arxiv_paper",
            uri="https://arxiv.org/pdf/1706.03762",
            title="Attention Is All You Need",
        ),
        raw_path=raw_path,
        metadata_path=tmp_path / "metadata.json",
        checksum="sha256:test",
    )

    class _Page:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class _Reader:
        pages = [_Page("Attention text"), _Page("")]

        def __init__(self, path: str) -> None:
            assert path == str(raw_path)

    with patch("rag.sources.extractors.PdfReader", _Reader):
        extracted = ArxivPdfExtractor().extract(fetch_result)

    assert extracted.text == "Attention text"
    assert extracted.sections[0].title == "Page 1"
    assert extracted.extraction_warnings == ["Page 2 produced no text"]
