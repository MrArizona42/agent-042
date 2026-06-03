from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import httpx

from rag.sources import process_source_instance
from rag.sources.artifacts import extracted_artifact_path, read_extracted_artifact
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
            f"<html><body><h1>{page_id.title()}</h1>"
            f"<p>{page_id} body text.</p></body></html>"
        ).encode("utf-8")
        return httpx.Response(
            200,
            content=content,
            headers={"content-type": "text/html"},
            request=request,
        )

    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


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
        source_type="html_docs",
        manifest_path=manifest_path,
        rag_data_root=tmp_path,
        fetchers={"html_docs": HtmlDocsFetcher(client=_html_client())},
    )
    artifact_path = extracted_artifact_path(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        source_document_id="html:tensors",
    )
    artifact = read_extracted_artifact(artifact_path)

    assert summary.total_selected == 1
    assert summary.fetched == 1
    assert summary.fetched_from_cache == 0
    assert summary.extracted == 1
    assert summary.extracted_from_cache == 0
    assert summary.failed == []
    assert artifact.document.text == "tensors body text."
    assert artifact.raw.path.endswith("pytorch_reference/raw/docs/html_tensors/page.html")


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
    fetchers = {"html_docs": HtmlDocsFetcher(client=_html_client())}

    first = process_source_instance(
        kb_id="pytorch_reference",
        source_instance_id="docs",
        source_type="html_docs",
        manifest_path=manifest_path,
        rag_data_root=tmp_path,
        fetchers=fetchers,
    )
    second = process_source_instance(
        kb_id="pytorch_reference",
        source_instance_id="docs",
        source_type="html_docs",
        manifest_path=manifest_path,
        rag_data_root=tmp_path,
        fetchers=fetchers,
    )
    forced = process_source_instance(
        kb_id="pytorch_reference",
        source_instance_id="docs",
        source_type="html_docs",
        manifest_path=manifest_path,
        rag_data_root=tmp_path,
        force_extract=True,
        fetchers=fetchers,
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
        source_type="html_docs",
        manifest_path=manifest_path,
        rag_data_root=tmp_path,
        document_ids=["html:broken"],
        fetchers={"html_docs": HtmlDocsFetcher(client=_html_client())},
    )

    assert summary.total_selected == 1
    assert summary.fetched == 0
    assert summary.extracted == 0
    assert len(summary.failed) == 1
    assert summary.failed[0].document_id == "html:broken"
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

    try:
        process_source_instance(
            kb_id="pytorch_reference",
            source_instance_id="docs",
            source_type="arxiv_paper",
            manifest_path=manifest_path,
            rag_data_root=tmp_path,
        )
    except ValueError as exc:
        assert "expected 'arxiv_paper'" in str(exc)
    else:
        raise AssertionError("expected source_type mismatch to fail")

