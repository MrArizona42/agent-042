from __future__ import annotations

from unittest.mock import patch

import httpx

from rag.contracts.metadata import source_document
from rag.sources.artifacts import (
    extracted_artifact_from_result,
    extracted_artifact_path,
    read_extracted_artifact,
    write_extracted_artifact,
)
from rag.sources.extractors import ArxivPdfExtractor, HtmlDocsExtractor
from rag.sources.fetchers import ArxivPaperFetcher, HtmlDocsFetcher, SourceFetchResult


def _source(*, local_id: str, title: str, uri: str, source_instance_id: str):
    return source_document(
        local_document_id=local_id,
        title=title,
        source_uri=uri,
        kb_id="pytorch_reference",
        source_instance_id=source_instance_id,
        adapter_id="generic.http_html",
        adapter_version="1",
        manifest_digest="sha256:manifest",
    )


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
    source = _source(
        local_id="tensors",
        title="Tensors",
        uri="https://pytorch.org/docs/stable/tensors.html",
        source_instance_id="docs",
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
    forced = fetcher.fetch(
        source,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        rag_data_root=tmp_path,
        force=True,
    )

    assert first.raw_path.as_posix().endswith("source_instances/docs/raw/docs_tensors/page.html")
    assert first.metadata_path.exists()
    assert second.from_cache is True
    assert forced.from_cache is False


def test_arxiv_fetcher_resolves_adapter_uri_to_pdf(tmp_path) -> None:
    source = source_document(
        local_document_id="1706.03762",
        title="Attention Is All You Need",
        source_uri="arxiv:1706.03762",
        kb_id="ml_papers_core",
        source_instance_id="papers",
        adapter_id="generic.arxiv_paper",
        adapter_version="1",
        manifest_digest="sha256:manifest",
    )
    fetcher = ArxivPaperFetcher(client=_client(b"%PDF fake", content_type="application/pdf"))

    result = fetcher.fetch(
        source,
        kb_id="ml_papers_core",
        source_instance_id="papers",
        rag_data_root=tmp_path,
    )

    assert result.raw_path.as_posix().endswith(
        "source_instances/papers/raw/papers_1706.03762/paper.pdf"
    )
    assert result.source_document.metadata["source_uri"] == "https://arxiv.org/pdf/1706.03762"
    assert result.checksum.startswith("sha256:")


def test_html_extractor_preserves_heading_sections_in_document_metadata(tmp_path) -> None:
    raw_path = tmp_path / "page.html"
    raw_path.write_text(
        """
        <html><body><h1>Tensors</h1><p>A tensor is an array.</p>
        <h2>Creation</h2><p>Use torch.tensor.</p></body></html>
        """,
        encoding="utf-8",
    )
    source = _source(
        local_id="tensors",
        title="Tensors",
        uri="https://pytorch.org/docs/stable/tensors.html",
        source_instance_id="docs",
    )
    extracted = HtmlDocsExtractor().extract(
        SourceFetchResult(
            source_document=source,
            raw_path=raw_path,
            metadata_path=tmp_path / "metadata.json",
            checksum="sha256:test",
        )
    )

    assert extracted.id_ == "docs:tensors"
    assert extracted.metadata["extraction_method"] == "html_bs4"
    assert [section["title"] for section in extracted.metadata["sections"]] == [
        "Tensors",
        "Creation",
    ]
    assert "torch.tensor" in extracted.text


def test_arxiv_pdf_extractor_uses_pypdf_reader(tmp_path) -> None:
    raw_path = tmp_path / "paper.pdf"
    raw_path.write_bytes(b"%PDF fake")
    source = source_document(
        local_document_id="1706.03762",
        title="Attention Is All You Need",
        source_uri="https://arxiv.org/pdf/1706.03762",
        kb_id="ml_papers_core",
        source_instance_id="papers",
        adapter_id="generic.arxiv_paper",
        adapter_version="1",
        manifest_digest="sha256:manifest",
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
        extracted = ArxivPdfExtractor().extract(
            SourceFetchResult(
                source_document=source,
                raw_path=raw_path,
                metadata_path=tmp_path / "metadata.json",
                checksum="sha256:test",
            )
        )

    assert extracted.text == "Attention text"
    assert extracted.metadata["sections"][0]["title"] == "Page 1"
    assert extracted.metadata["extraction_warnings"] == ["Page 2 produced no text"]


def test_extracted_artifact_round_trips_native_document(tmp_path) -> None:
    source = _source(
        local_id="tensors",
        title="Tensors",
        uri="https://pytorch.org/docs/stable/tensors.html",
        source_instance_id="docs",
    )
    fetch_result = HtmlDocsFetcher(
        client=_client(
            b"<html><body><h1>Tensors</h1><p>Tensor text.</p></body></html>",
            content_type="text/html",
        )
    ).fetch(
        source,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        rag_data_root=tmp_path,
    )
    extracted = HtmlDocsExtractor().extract(fetch_result)
    artifact = extracted_artifact_from_result(
        kb_id="pytorch_reference",
        source_instance_id="docs",
        fetch_result=fetch_result,
        extracted_document=extracted,
    )
    path = extracted_artifact_path(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        source_document_id=source.id_,
    )

    write_extracted_artifact(path, artifact)
    restored = read_extracted_artifact(path)

    assert restored.schema_version == 2
    assert restored.document.id_ == "docs:tensors"
    assert restored.document.text == "Tensor text."
    assert restored.document.metadata["adapter_id"] == "generic.http_html"
