from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import httpx

from rag.ingest.adapters import ManifestSourceAdapter
from rag.sources import process_source_instance
from rag.sources.artifacts import extracted_artifact_path, read_extracted_artifact
from rag.sources.extractors import HtmlDocsExtractor
from rag.sources.fetchers import HtmlDocsFetcher


def _write_manifest(path: Path, content: str) -> Path:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return path


def _html_client() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/broken.html"):
            return httpx.Response(500, content=b"broken", request=request)
        page_id = request.url.path.rsplit("/", 1)[-1].removesuffix(".html")
        content = (
            f"<html><body><h1>{page_id.title()}</h1><p>{page_id} body text.</p></body></html>"
        ).encode("utf-8")
        return httpx.Response(
            200,
            content=content,
            headers={"content-type": "text/html"},
            request=request,
        )

    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def _html_adapter() -> ManifestSourceAdapter:
    return ManifestSourceAdapter(
        adapter_id="generic.http_html",
        version="1",
        source_type="html_docs",
        _fetcher_factory=lambda: HtmlDocsFetcher(client=_html_client()),
        _extractor_factory=HtmlDocsExtractor,
    )


def test_process_source_instance_writes_extracted_artifact(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "sources.toml",
        """
        schema_version = 1
        source_type = "html_docs"

        [[documents]]
        id = "tensors"
        title = "Tensors"
        url = "https://docs.test/tensors.html"
        """,
    )

    summary = process_source_instance(
        kb_id="pytorch_reference",
        source_instance_id="docs",
        manifest_path=manifest_path,
        rag_data_root=tmp_path,
        source_adapter=_html_adapter(),
    )
    artifact_path = extracted_artifact_path(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        source_document_id="html_docs:tensors",
    )
    artifact = read_extracted_artifact(artifact_path)

    assert summary.total_selected == 1
    assert summary.fetched == 1
    assert summary.fetched_from_cache == 0
    assert summary.extracted == 1
    assert summary.extracted_from_cache == 0
    assert summary.failed == []
    assert artifact.document.text == "tensors body text."
    assert artifact.raw.path.endswith("source_instances/docs/raw/html_docs_tensors/page.html")


def test_process_source_instance_reuses_extracted_artifact(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "sources.toml",
        """
        schema_version = 1
        source_type = "html_docs"

        [[documents]]
        id = "tensors"
        title = "Tensors"
        url = "https://docs.test/tensors.html"
        """,
    )

    first = process_source_instance(
        kb_id="pytorch_reference",
        source_instance_id="docs",
        manifest_path=manifest_path,
        rag_data_root=tmp_path,
        source_adapter=_html_adapter(),
    )
    second = process_source_instance(
        kb_id="pytorch_reference",
        source_instance_id="docs",
        manifest_path=manifest_path,
        rag_data_root=tmp_path,
        source_adapter=_html_adapter(),
    )
    forced = process_source_instance(
        kb_id="pytorch_reference",
        source_instance_id="docs",
        manifest_path=manifest_path,
        rag_data_root=tmp_path,
        force_extract=True,
        source_adapter=_html_adapter(),
    )

    assert first.extracted == 1
    assert second.fetched == 0
    assert second.extracted == 0
    assert second.extracted_from_cache == 1
    assert forced.fetched == 1
    assert forced.fetched_from_cache == 1
    assert forced.extracted == 1


def test_process_source_instance_filters_documents_and_collects_failures(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path / "sources.toml",
        """
        schema_version = 1
        source_type = "html_docs"

        [[documents]]
        id = "tensors"
        title = "Tensors"
        url = "https://docs.test/tensors.html"

        [[documents]]
        id = "broken"
        title = "Broken"
        url = "https://docs.test/broken.html"
        """,
    )

    summary = process_source_instance(
        kb_id="pytorch_reference",
        source_instance_id="docs",
        manifest_path=manifest_path,
        rag_data_root=tmp_path,
        document_ids=["html_docs:broken"],
        source_adapter=_html_adapter(),
    )

    assert summary.total_selected == 1
    assert summary.fetched == 0
    assert summary.extracted == 0
    assert len(summary.failed) == 1
    assert summary.failed[0].document_id == "html_docs:broken"
    assert summary.failed[0].error_type == "HTTPStatusError"


def test_process_source_instance_rejects_source_type_mismatch(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path / "sources.toml",
        """
        schema_version = 1
        source_type = "html_docs"

        [[documents]]
        id = "tensors"
        title = "Tensors"
        url = "https://docs.test/tensors.html"
        """,
    )

    from rag.ingest.adapters import ManifestSourceAdapter
    from rag.sources.extractors import ArxivPdfExtractor
    from rag.sources.fetchers import ArxivPaperFetcher

    arxiv_adapter = ManifestSourceAdapter(
        adapter_id="generic.arxiv_paper",
        version="1",
        source_type="arxiv_paper",
        _fetcher_factory=ArxivPaperFetcher,
        _extractor_factory=ArxivPdfExtractor,
    )

    try:
        process_source_instance(
            kb_id="pytorch_reference",
            source_instance_id="docs",
            manifest_path=manifest_path,
            rag_data_root=tmp_path,
            source_adapter=arxiv_adapter,
        )
    except ValueError as exc:
        assert "arxiv_paper" in str(exc)
    else:
        raise AssertionError("expected source_type mismatch to raise ValueError")
