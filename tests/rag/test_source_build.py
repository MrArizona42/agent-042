from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import httpx

from app_config.catalog.schema import CatalogConfig
from rag.ingest.adapters import ManifestSourceAdapter, SourceAdapterRegistry
from rag.sources import (
    ChunkingConfig,
    build_catalog_source,
    build_catalog_sources,
    build_source_instance,
    build_source_instance_by_global_id,
    build_source_instances_by_global_id,
    resolve_catalog_sources,
)
from rag.sources.chunks import chunk_artifact_path, read_chunk_artifact
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


def _mock_registry() -> SourceAdapterRegistry:
    registry = SourceAdapterRegistry()
    registry.register(_html_adapter())
    return registry


def _html_adapter() -> ManifestSourceAdapter:
    return ManifestSourceAdapter(
        adapter_id="generic.http_html",
        version="1",
        source_type="html_docs",
        _fetcher_factory=lambda: HtmlDocsFetcher(client=_html_client()),
        _extractor_factory=HtmlDocsExtractor,
    )


def _catalog_declared_html_adapter() -> ManifestSourceAdapter:
    """Factory referenced by `[[source_adapters]]` in `_catalog_with_declared_adapters`."""
    return _html_adapter()


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


def _catalog_with_declared_adapters(tmp_path: Path, manifest_path: Path) -> Path:
    return _write_manifest(
        tmp_path / "catalog.toml",
        f"""
        schema_version = 3

        [[source_adapters]]
        id = "generic.http_html"
        version = "1"
        description = "Fetches HTTP HTML pages."
        factory = "tests.rag.test_source_build:_catalog_declared_html_adapter"

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

        [[sources]]
        type = "html_docs"
        kb = "pytorch_reference"
        id = "docs"
        manifest = "{manifest_path.as_posix()}"
        ingest_adapter = {{ id = "generic.http_html", version = "1" }}
        """,
    )


def test_build_source_instance_runs_full_pre_index_lifecycle(tmp_path: Path) -> None:
    summary = build_source_instance(
        kb_id="pytorch_reference",
        source_instance_id="docs",
        manifest_path=_manifest(tmp_path),
        rag_data_root=tmp_path,
        source_adapter=_html_adapter(),
        document_ids=["html_docs:tensors"],
        chunking=ChunkingConfig(chunk_size=24, chunk_overlap=4),
    )
    path = chunk_artifact_path(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        source_document_id="html_docs:tensors",
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
    kwargs = {
        "kb_id": "pytorch_reference",
        "source_instance_id": "docs",
        "manifest_path": _manifest(tmp_path),
        "rag_data_root": tmp_path,
        "source_adapter": _html_adapter(),
        "document_ids": ["html_docs:tensors"],
        "chunking": ChunkingConfig(chunk_size=24, chunk_overlap=4),
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
        manifest_path=_manifest(tmp_path),
        rag_data_root=tmp_path,
        source_adapter=_html_adapter(),
        chunking=ChunkingConfig(chunk_size=24, chunk_overlap=4),
    )
    failed = build_source_instance(
        kb_id="pytorch_reference",
        source_instance_id="docs_failed",
        manifest_path=_manifest(tmp_path),
        rag_data_root=tmp_path,
        source_adapter=_html_adapter(),
        document_ids=["html_docs:broken"],
        chunking=ChunkingConfig(chunk_size=24, chunk_overlap=4),
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
        manifest_path=_manifest(tmp_path),
        rag_data_root=tmp_path,
        source_adapter=_html_adapter(),
        document_ids=["html_docs:missing"],
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
        document_ids=["html_docs:tensors"],
        chunking=ChunkingConfig(chunk_size=24, chunk_overlap=4),
        adapter_registry=_mock_registry(),
    )

    assert summary.catalog_path == catalog_path.as_posix()
    assert summary.source.kb == "pytorch_reference"
    assert summary.source.id == "docs"
    assert summary.source.ingest_adapter.id == "generic.http_html"
    assert summary.build.status == "success"
    assert summary.build.kb_id == "pytorch_reference"
    assert summary.build.source_instance_id == "pytorch_reference.docs"


def test_build_catalog_source_resolves_manifest_relative_to_catalog(
    tmp_path: Path,
) -> None:
    manifest_path = _manifest(tmp_path)
    catalog_path = _write_manifest(
        tmp_path / "catalog.toml",
        """
        schema_version = 2

        [[tasks]]
        id = "code"
        enabled = true
        label = "Code"
        routing_description = "Coding help"
        kb_refs = ["pytorch_reference"]
        adapter = { enabled = false }

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

        [[sources]]
        type = "html_docs"
        kb = "pytorch_reference"
        id = "docs"
        manifest = "sources.toml"
        ingest_adapter = { id = "generic.http_html", version = "1" }
        """,
    )

    summary = build_catalog_source(
        catalog_path=catalog_path,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        rag_data_root=tmp_path,
        document_ids=["html_docs:tensors"],
        chunking=ChunkingConfig(chunk_size=24, chunk_overlap=4),
        adapter_registry=_mock_registry(),
    )

    assert manifest_path.exists()
    assert summary.build.status == "success"


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


def test_resolve_catalog_sources_supports_all_and_subset(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    catalog_path = _catalog(tmp_path, manifest_path)
    catalog = CatalogConfig.model_validate(
        __import__("tomllib").loads(catalog_path.read_text(encoding="utf-8"))
    )

    resolved_all = resolve_catalog_sources(catalog, kb_id="pytorch_reference")
    assert [source.id for source in resolved_all] == ["docs"]
    assert [
        source.id
        for source in resolve_catalog_sources(
            catalog,
            kb_id="pytorch_reference",
            source_instance_ids=["docs"],
        )
    ] == ["docs"]

    try:
        resolve_catalog_sources(
            catalog,
            kb_id="pytorch_reference",
            source_instance_ids=["missing"],
        )
    except ValueError as exc:
        assert "source_instance_ids=['missing']" in str(exc)
    else:
        raise AssertionError("expected missing source subset to fail")


def test_resolve_catalog_sources_excludes_declared_benchmark_instances(tmp_path: Path) -> None:
    """A normal 'build all for kb' must never pick up a role='benchmark' source instance."""
    manifest_path = _manifest(tmp_path)
    content = _catalog(tmp_path, manifest_path).read_text(encoding="utf-8") + dedent(
        """

        [[benchmark_adapters]]
        id = "benchmark.fake"
        version = "1"
        description = "Fake benchmark adapter."
        factory = "rag.ingest.adapters:make_http_html_adapter"

        [[source_instances]]
        id = "pytorch_reference.qa_benchmark"
        description = "QA benchmark cases."
        role = "benchmark"
        knowledge_base = "pytorch_reference"
        adapter = { id = "benchmark.fake", version = "1" }

        [source_instances.benchmark]
        suites = ["generation_quality"]
        """
    )
    catalog_path = _write_manifest(tmp_path / "catalog.toml", content)
    catalog = CatalogConfig.model_validate(
        __import__("tomllib").loads(catalog_path.read_text(encoding="utf-8"))
    )

    resolved_all = resolve_catalog_sources(catalog, kb_id="pytorch_reference")

    assert [source.id for source in resolved_all] == ["docs"]


def test_build_catalog_sources_builds_selected_sources(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    catalog_path = _catalog(tmp_path, manifest_path)

    summary = build_catalog_sources(
        catalog_path=catalog_path,
        kb_id="pytorch_reference",
        source_instance_ids=["docs"],
        rag_data_root=tmp_path,
        document_ids=["html_docs:tensors"],
        chunking=ChunkingConfig(chunk_size=24, chunk_overlap=4),
        adapter_registry=_mock_registry(),
    )

    assert summary.kb_id == "pytorch_reference"
    assert [source_summary.source.id for source_summary in summary.sources] == ["docs"]
    assert summary.sources[0].build.status == "success"


def test_build_catalog_source_uses_catalog_declared_adapter_by_default(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    catalog_path = _catalog_with_declared_adapters(tmp_path, manifest_path)

    summary = build_catalog_source(
        catalog_path=catalog_path,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        rag_data_root=tmp_path,
        document_ids=["html_docs:tensors"],
        chunking=ChunkingConfig(chunk_size=24, chunk_overlap=4),
    )

    assert summary.build.status == "success"


def _catalog_with_corpus_and_benchmark_instances(tmp_path: Path, manifest_path: Path) -> Path:
    return _write_manifest(
        tmp_path / "catalog.toml",
        """
        schema_version = 3

        [[source_adapters]]
        id = "generic.http_html"
        version = "1"
        description = "Fetches HTTP HTML pages."
        factory = "tests.rag.test_source_build:_catalog_declared_html_adapter"

        [[benchmark_adapters]]
        id = "benchmark.fake"
        version = "1"
        description = "Fake benchmark adapter; never loaded by build-source."
        factory = "rag.ingest.adapters:make_http_html_adapter"

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

        [[source_instances]]
        id = "pytorch_reference.docs"
        description = "Official PyTorch documentation pages."
        role = "corpus"
        knowledge_base = "pytorch_reference"
        adapter = { id = "generic.http_html", version = "1" }

        [[source_instances]]
        id = "pytorch_reference.qa_benchmark"
        description = "QA benchmark cases for PyTorch documentation."
        role = "benchmark"
        knowledge_base = "pytorch_reference"
        adapter = { id = "benchmark.fake", version = "1" }

        [source_instances.benchmark]
        suites = ["generation_quality"]
        """,
    )


def test_build_source_instance_by_global_id_delegates_to_legacy_source(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    catalog_path = _catalog(tmp_path, manifest_path)

    summary = build_source_instance_by_global_id(
        catalog_path=catalog_path,
        source_instance_id="pytorch_reference.docs",
        rag_data_root=tmp_path,
        document_ids=["html_docs:tensors"],
        chunking=ChunkingConfig(chunk_size=24, chunk_overlap=4),
        adapter_registry=_mock_registry(),
    )

    assert summary.source_instance_id == "pytorch_reference.docs"
    assert summary.role == "corpus"
    assert summary.build.status == "success"
    assert summary.build.kb_id == "pytorch_reference"


def test_build_source_instance_by_global_id_rejects_benchmark_role(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    catalog_path = _catalog_with_corpus_and_benchmark_instances(tmp_path, manifest_path)

    try:
        build_source_instance_by_global_id(
            catalog_path=catalog_path,
            source_instance_id="pytorch_reference.qa_benchmark",
            rag_data_root=tmp_path,
        )
    except ValueError as exc:
        assert "role 'benchmark'" in str(exc)
        assert "prepare-benchmark" in str(exc)
    else:
        raise AssertionError("expected benchmark-role target to be rejected")


def test_build_source_instance_by_global_id_builds_declared_corpus_instance(
    tmp_path: Path,
) -> None:
    catalog_path = _catalog_with_corpus_and_benchmark_instances(tmp_path, tmp_path / "unused.toml")
    conventional_manifest = (
        tmp_path / "rag_data" / "source_instances" / "pytorch_reference.docs" / "manifest.toml"
    )
    conventional_manifest.parent.mkdir(parents=True)
    _write_manifest(
        conventional_manifest,
        """
        schema_version = 1
        source_type = "html_docs"

        [[documents]]
        id = "tensors"
        title = "Tensors"
        url = "https://docs.test/tensors.html"
        """,
    )

    summary = build_source_instance_by_global_id(
        catalog_path=catalog_path,
        source_instance_id="pytorch_reference.docs",
        rag_data_root=tmp_path / "rag_data",
        document_ids=["html_docs:tensors"],
        chunking=ChunkingConfig(chunk_size=24, chunk_overlap=4),
        adapter_registry=_mock_registry(),
    )

    assert summary.role == "corpus"
    assert summary.build.status == "success"


def test_build_catalog_sources_builds_declared_v3_corpus_instances_only(
    tmp_path: Path,
) -> None:
    catalog_path = _catalog_with_corpus_and_benchmark_instances(tmp_path, tmp_path / "unused.toml")
    conventional_manifest = (
        tmp_path / "rag_data" / "source_instances" / "pytorch_reference.docs" / "manifest.toml"
    )
    conventional_manifest.parent.mkdir(parents=True)
    _write_manifest(
        conventional_manifest,
        """
        schema_version = 1
        source_type = "html_docs"

        [[documents]]
        id = "tensors"
        title = "Tensors"
        url = "https://docs.test/tensors.html"
        """,
    )

    summary = build_catalog_sources(
        catalog_path=catalog_path,
        kb_id="pytorch_reference",
        source_instance_ids=None,
        rag_data_root=tmp_path / "rag_data",
        document_ids=["html_docs:tensors"],
        chunking=ChunkingConfig(chunk_size=24, chunk_overlap=4),
        adapter_registry=_mock_registry(),
    )

    assert [source.source_instance_id for source in summary.sources] == ["pytorch_reference.docs"]
    assert summary.sources[0].source is None
    assert summary.sources[0].build.status == "success"


def test_build_source_instances_by_global_id_builds_multiple(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    catalog_path = _catalog(tmp_path, manifest_path)

    summaries = build_source_instances_by_global_id(
        catalog_path=catalog_path,
        source_instance_ids=["pytorch_reference.docs", "other_reference.docs"],
        rag_data_root=tmp_path,
        document_ids=["html_docs:tensors"],
        chunking=ChunkingConfig(chunk_size=24, chunk_overlap=4),
        adapter_registry=_mock_registry(),
    )

    assert [summary.source_instance_id for summary in summaries] == [
        "pytorch_reference.docs",
        "other_reference.docs",
    ]
    assert all(summary.build.status == "success" for summary in summaries)
