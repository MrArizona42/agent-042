from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import httpx

from rag.sources import ChunkingConfig, build_catalog_source, build_source_instance
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


def _catalog(tmp_path: Path, manifest_path: Path) -> Path:
    return _write_manifest(
        tmp_path / "catalog.toml",
        f"""
        schema_version = 2

        [[tasks]]
        id = "code"
        enabled = true
        label = "Code"
        routing_description = "Coding help"
        kb_refs = ["pytorch_reference"]
        adapter = {{ enabled = false }}

        [[knowledge_bases]]
        id = "pytorch_reference"
        enabled = true
        label = "PyTorch reference"
        description = "PyTorch documentation"
        selection_description = "PyTorch docs"
        update_strategy = "replace"
        default_alias = "champion"
        aliases.champion.top_k = 5
        aliases.champion.score_threshold = 0.35
        aliases.champion.retrieval_strategy = "dense"
        aliases.champion.reranker_multiplier = 1

        [[knowledge_bases]]
        id = "other_reference"
        enabled = true
        label = "Other reference"
        description = "Other documentation"
        selection_description = "Other docs"
        update_strategy = "replace"
        default_alias = "champion"
        aliases.champion.top_k = 5
        aliases.champion.score_threshold = 0.35
        aliases.champion.retrieval_strategy = "dense"
        aliases.champion.reranker_multiplier = 1

        [[sources]]
        type = "html_docs"
        kb = "pytorch_reference"
        id = "docs"
        manifest = "{manifest_path.as_posix()}"
        ingest_adapter = {{ id = "generic.http_html", version = "1" }}

        [[sources]]
        type = "html_docs"
        kb = "other_reference"
        id = "docs"
        manifest = "{manifest_path.as_posix()}"
        ingest_adapter = {{ id = "generic.http_html", version = "1" }}
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


def test_build_catalog_source_uses_kb_and_source_instance_pair(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    catalog_path = _catalog(tmp_path, manifest_path)

    summary = build_catalog_source(
        catalog_path=catalog_path,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        rag_data_root=tmp_path,
        document_ids=["html:tensors"],
        chunking=ChunkingConfig(chunk_size=24, chunk_overlap=4),
        fetchers={"html_docs": HtmlDocsFetcher(client=_html_client())},
    )

    assert summary.catalog_path == catalog_path.as_posix()
    assert summary.source.kb == "pytorch_reference"
    assert summary.source.id == "docs"
    assert summary.source.ingest_adapter is not None
    assert summary.source.ingest_adapter.id == "generic.http_html"
    assert summary.build.status == "success"
    assert summary.build.kb_id == "pytorch_reference"
    assert summary.build.source_instance_id == "docs"


def test_build_catalog_source_rejects_missing_kb_source_pair(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    catalog_path = _catalog(tmp_path, manifest_path)

    try:
        build_catalog_source(
            catalog_path=catalog_path,
            kb_id="missing_reference",
            source_instance_id="docs",
            rag_data_root=tmp_path,
        )
    except ValueError as exc:
        assert "kb_id='missing_reference'" in str(exc)
        assert "source_instance_id='docs'" in str(exc)
    else:
        raise AssertionError("expected missing catalog source pair to fail")
