from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import httpx

from rag.sources import ChunkingConfig, build_source_instance
from rag.sources.chunks import chunk_artifact_path, read_chunk_artifact
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
            f"<p>{page_id} body text. More useful text.</p></body></html>"
        ).encode("utf-8")
        return httpx.Response(
            200,
            content=content,
            headers={"content-type": "text/html"},
            request=request,
        )

    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def _manifest(tmp_path: Path) -> Path:
    return _write_manifest(
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


def test_build_source_instance_runs_full_pre_index_lifecycle(tmp_path: Path) -> None:
    summary = build_source_instance(
        kb_id="pytorch_reference",
        source_instance_id="docs",
        source_type="html_docs",
        manifest_path=_manifest(tmp_path),
        rag_data_root=tmp_path,
        document_ids=["html:tensors"],
        chunking=ChunkingConfig(chunk_size=24, chunk_overlap=4),
        fetchers={"html_docs": HtmlDocsFetcher(client=_html_client())},
    )
    path = chunk_artifact_path(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        source_document_id="html:tensors",
    )
    artifact = read_chunk_artifact(path)

    assert summary.status == "success"
    assert summary.processing.total_selected == 1
    assert summary.processing.extracted == 1
    assert summary.chunking.total_selected == 1
    assert summary.chunking.chunked == 1
    assert summary.chunking.chunk_count == len(artifact.chunks)
    assert artifact.chunks[0].metadata["kb_id"] == "pytorch_reference"


def test_build_source_instance_reuses_artifact_caches(tmp_path: Path) -> None:
    fetchers = {"html_docs": HtmlDocsFetcher(client=_html_client())}
    kwargs = {
        "kb_id": "pytorch_reference",
        "source_instance_id": "docs",
        "source_type": "html_docs",
        "manifest_path": _manifest(tmp_path),
        "rag_data_root": tmp_path,
        "document_ids": ["html:tensors"],
        "chunking": ChunkingConfig(chunk_size=24, chunk_overlap=4),
        "fetchers": fetchers,
    }

    first = build_source_instance(**kwargs)
    second = build_source_instance(**kwargs)
    forced = build_source_instance(
        **{
            **kwargs,
            "force_extract": True,
            "force_chunk": True,
        }
    )

    assert first.status == "success"
    assert second.status == "success"
    assert second.processing.extracted == 0
    assert second.processing.extracted_from_cache == 1
    assert second.chunking.chunked == 0
    assert second.chunking.from_cache == 1
    assert forced.processing.extracted == 1
    assert forced.processing.fetched_from_cache == 1
    assert forced.chunking.chunked == 1


def test_build_source_instance_reports_partial_and_failed_statuses(tmp_path: Path) -> None:
    partial = build_source_instance(
        kb_id="pytorch_reference",
        source_instance_id="docs",
        source_type="html_docs",
        manifest_path=_manifest(tmp_path),
        rag_data_root=tmp_path,
        chunking=ChunkingConfig(chunk_size=24, chunk_overlap=4),
        fetchers={"html_docs": HtmlDocsFetcher(client=_html_client())},
    )
    failed = build_source_instance(
        kb_id="pytorch_reference",
        source_instance_id="docs_failed",
        source_type="html_docs",
        manifest_path=_manifest(tmp_path),
        rag_data_root=tmp_path,
        document_ids=["html:broken"],
        chunking=ChunkingConfig(chunk_size=24, chunk_overlap=4),
        fetchers={"html_docs": HtmlDocsFetcher(client=_html_client())},
    )

    assert partial.status == "partial"
    assert partial.processing.extracted == 1
    assert len(partial.processing.failed) == 1
    assert partial.chunking.chunked == 1
    assert failed.status == "failed"
    assert failed.processing.total_selected == 1
    assert len(failed.processing.failed) == 1
    assert failed.chunking.total_selected == 0


def test_build_source_instance_reports_empty_selection(tmp_path: Path) -> None:
    summary = build_source_instance(
        kb_id="pytorch_reference",
        source_instance_id="docs",
        source_type="html_docs",
        manifest_path=_manifest(tmp_path),
        rag_data_root=tmp_path,
        document_ids=["html:missing"],
        fetchers={"html_docs": HtmlDocsFetcher(client=_html_client())},
    )

    assert summary.status == "empty"
    assert summary.processing.total_selected == 0
    assert summary.chunking.total_selected == 0

