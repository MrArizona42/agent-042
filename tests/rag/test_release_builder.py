"""Tests for `rag.control_plane.release_builder.build_release`.

Covers transformation-digest-scoped node isolation, content-addressed
release/collection naming, idempotent reuse of an identical release, and
build failure cleanup -- the phase 2 acceptance criteria from the
declarative alias workflow plan.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from textwrap import dedent

import httpx
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from app_config.catalog.schema import AliasBuildConfig, CatalogConfig
from rag.adapters import ManifestSourceAdapter, SourceAdapterRegistry
from rag.contracts.manifests import read_release_manifest, release_manifest_path
from rag.control_plane.release_builder import build_release
from rag.indexing.llamaindex_qdrant import QdrantCollectionManager
from rag.sources.extractors import HtmlDocsExtractor
from rag.sources.fetchers import HtmlDocsFetcher


class _EmbeddingClient:
    dimension = 3

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [1.0, 0.0, 0.0] if "tensor" in text.lower() else [0.0, 1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class _SparseClient:
    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector(indices=[1], values=[1.0]) for _ in texts]


def _html_client() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        page_id = request.url.path.rsplit("/", 1)[-1].removesuffix(".html")
        content = (
            f"<html><body><h1>{page_id.title()}</h1>"
            f"<p>{page_id} body text. More useful text for chunking.</p></body></html>"
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
        default_uri_prefix="http_html",
        _fetcher_factory=lambda: HtmlDocsFetcher(client=_html_client()),
        _extractor_factory=HtmlDocsExtractor,
    )


def _registry() -> SourceAdapterRegistry:
    registry = SourceAdapterRegistry()
    registry.register(_html_adapter())
    return registry


def _write_manifest(path: Path, content: str) -> Path:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return path


def _setup_source_manifest(rag_data_root: Path) -> None:
    manifest_dir = rag_data_root / "source_instances" / "pytorch_reference.docs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        manifest_dir / "manifest.toml",
        """
        schema_version = 1
        [[documents]]
        id = "tensors"
        title = "Tensors"
        url = "https://docs.test/tensors.html"
        """,
    )


def _catalog_config(*, chunk_size: int = 512, chunk_overlap: int = 64) -> CatalogConfig:
    text = dedent(
        f"""
        schema_version = 4

        [[knowledge_bases]]
        id = "pytorch_reference"
        description = "PyTorch docs"
        default_alias = "champion"

        [knowledge_bases.aliases.champion.build.chunking]
        strategy = "sentence"
        chunk_size = {chunk_size}
        chunk_overlap = {chunk_overlap}

        [knowledge_bases.aliases.champion.build.dense_encoder]
        model = "test-embedding"
        dimension = 3

        [knowledge_bases.aliases.champion.retrieve]
        strategy = "dense"
        top_k = 5
        score_threshold = 0.35
        reranker_multiplier = 1

        [[source_adapters]]
        id = "generic.http_html"
        version = "1"
        description = "Fetches HTTP HTML pages."
        factory = "rag.adapters.sources:make_http_html_adapter"

        [[source_instances]]
        id = "pytorch_reference.docs"
        description = "Official docs."
        role = "corpus"
        knowledge_base = "pytorch_reference"
        adapter = {{ id = "generic.http_html", version = "1" }}
        """
    ).strip()
    return CatalogConfig(**tomllib.loads(text))


def _build_config(catalog_cfg: CatalogConfig) -> AliasBuildConfig:
    return catalog_cfg.knowledge_bases[0].aliases["champion"].build


@pytest.fixture()
def qdrant_client() -> QdrantClient:
    return QdrantClient(":memory:")


def _manager_factory(qdrant_client: QdrantClient):
    def _factory(collection_name: str) -> QdrantCollectionManager:
        return QdrantCollectionManager(client=qdrant_client, collection_name=collection_name)

    return _factory


def test_build_release_produces_content_addressed_release(tmp_path: Path, qdrant_client) -> None:
    _setup_source_manifest(tmp_path)
    catalog_cfg = _catalog_config()
    build_config = _build_config(catalog_cfg)

    release = build_release(
        kb_id="pytorch_reference",
        build_config=build_config,
        catalog_digest="sha256:" + "a" * 64,
        catalog_cfg=catalog_cfg,
        rag_data_root=tmp_path,
        collection_manager_factory=_manager_factory(qdrant_client),
        embedding_client=_EmbeddingClient(),
        adapter_registry=_registry(),
    )

    assert release.id.startswith("ragrel_pytorch_reference_")
    assert release.collection_name.startswith("rag__pytorch_reference__")
    assert release.chunk_count > 0
    assert qdrant_client.collection_exists(release.collection_name)

    manifest_path = release_manifest_path(
        rag_data_root=tmp_path, kb_id="pytorch_reference", release_id=release.id
    )
    assert manifest_path.exists()
    assert read_release_manifest(manifest_path).manifest_id == release.manifest_id


def test_build_release_reuses_identical_release_without_rebuilding(
    tmp_path: Path, qdrant_client
) -> None:
    _setup_source_manifest(tmp_path)
    catalog_cfg = _catalog_config()
    build_config = _build_config(catalog_cfg)
    kwargs = dict(
        kb_id="pytorch_reference",
        build_config=build_config,
        catalog_digest="sha256:" + "a" * 64,
        catalog_cfg=catalog_cfg,
        rag_data_root=tmp_path,
        collection_manager_factory=_manager_factory(qdrant_client),
        embedding_client=_EmbeddingClient(),
        adapter_registry=_registry(),
    )

    first = build_release(**kwargs)
    second = build_release(**kwargs)

    assert first.id == second.id
    assert first.manifest_id == second.manifest_id


def test_build_release_different_chunking_produces_different_release_and_isolated_nodes(
    tmp_path: Path, qdrant_client
) -> None:
    _setup_source_manifest(tmp_path)

    small_catalog = _catalog_config(chunk_size=512, chunk_overlap=64)
    large_catalog = _catalog_config(chunk_size=256, chunk_overlap=32)

    small_release = build_release(
        kb_id="pytorch_reference",
        build_config=_build_config(small_catalog),
        catalog_digest="sha256:" + "a" * 64,
        catalog_cfg=small_catalog,
        rag_data_root=tmp_path,
        collection_manager_factory=_manager_factory(qdrant_client),
        embedding_client=_EmbeddingClient(),
        adapter_registry=_registry(),
    )
    large_release = build_release(
        kb_id="pytorch_reference",
        build_config=_build_config(large_catalog),
        catalog_digest="sha256:" + "a" * 64,
        catalog_cfg=large_catalog,
        rag_data_root=tmp_path,
        collection_manager_factory=_manager_factory(qdrant_client),
        embedding_client=_EmbeddingClient(),
        adapter_registry=_registry(),
    )

    assert small_release.id != large_release.id
    assert small_release.collection_name != large_release.collection_name

    chunks_root = tmp_path / "source_instances" / "pytorch_reference.docs" / "chunks"
    digest_dirs = [p for p in chunks_root.iterdir() if p.is_dir()]
    assert len(digest_dirs) == 2


def test_build_release_hybrid_requires_sparse_client(tmp_path: Path, qdrant_client) -> None:
    _setup_source_manifest(tmp_path)
    text = dedent(
        """
        schema_version = 4

        [[knowledge_bases]]
        id = "pytorch_reference"
        description = "PyTorch docs"
        default_alias = "challenger"

        [knowledge_bases.aliases.challenger.build.chunking]
        strategy = "sentence"
        chunk_size = 512
        chunk_overlap = 64

        [knowledge_bases.aliases.challenger.build.dense_encoder]
        model = "test-embedding"
        dimension = 3

        [knowledge_bases.aliases.challenger.build.sparse_encoder]
        model = "test-sparse"

        [knowledge_bases.aliases.challenger.retrieve]
        strategy = "hybrid"
        top_k = 5
        score_threshold = 0.01
        reranker_multiplier = 1

        [[source_adapters]]
        id = "generic.http_html"
        version = "1"
        description = "Fetches HTTP HTML pages."
        factory = "rag.adapters.sources:make_http_html_adapter"

        [[source_instances]]
        id = "pytorch_reference.docs"
        description = "Official docs."
        role = "corpus"
        knowledge_base = "pytorch_reference"
        adapter = { id = "generic.http_html", version = "1" }
        """
    ).strip()
    catalog_cfg = CatalogConfig(**tomllib.loads(text))
    build_config = catalog_cfg.knowledge_bases[0].aliases["challenger"].build

    with pytest.raises(ValueError, match="sparse_encoder_client"):
        build_release(
            kb_id="pytorch_reference",
            build_config=build_config,
            catalog_digest="sha256:" + "a" * 64,
            catalog_cfg=catalog_cfg,
            rag_data_root=tmp_path,
            collection_manager_factory=_manager_factory(qdrant_client),
            embedding_client=_EmbeddingClient(),
            adapter_registry=_registry(),
        )

    releases_dir = tmp_path / "knowledge_bases" / "pytorch_reference" / "releases"
    assert not releases_dir.exists() or not list(releases_dir.glob("*.json"))


def test_build_release_hybrid_with_sparse_client_succeeds(tmp_path: Path, qdrant_client) -> None:
    _setup_source_manifest(tmp_path)
    text = dedent(
        """
        schema_version = 4

        [[knowledge_bases]]
        id = "pytorch_reference"
        description = "PyTorch docs"
        default_alias = "challenger"

        [knowledge_bases.aliases.challenger.build.chunking]
        strategy = "sentence"
        chunk_size = 512
        chunk_overlap = 64

        [knowledge_bases.aliases.challenger.build.dense_encoder]
        model = "test-embedding"
        dimension = 3

        [knowledge_bases.aliases.challenger.build.sparse_encoder]
        model = "test-sparse"

        [knowledge_bases.aliases.challenger.retrieve]
        strategy = "hybrid"
        top_k = 5
        score_threshold = 0.01
        reranker_multiplier = 1

        [[source_adapters]]
        id = "generic.http_html"
        version = "1"
        description = "Fetches HTTP HTML pages."
        factory = "rag.adapters.sources:make_http_html_adapter"

        [[source_instances]]
        id = "pytorch_reference.docs"
        description = "Official docs."
        role = "corpus"
        knowledge_base = "pytorch_reference"
        adapter = { id = "generic.http_html", version = "1" }
        """
    ).strip()
    catalog_cfg = CatalogConfig(**tomllib.loads(text))
    build_config = catalog_cfg.knowledge_bases[0].aliases["challenger"].build

    release = build_release(
        kb_id="pytorch_reference",
        build_config=build_config,
        catalog_digest="sha256:" + "a" * 64,
        catalog_cfg=catalog_cfg,
        rag_data_root=tmp_path,
        collection_manager_factory=_manager_factory(qdrant_client),
        embedding_client=_EmbeddingClient(),
        sparse_encoder_client=_SparseClient(),
        adapter_registry=_registry(),
    )

    assert release.build_config.sparse_encoder is not None
    info = qdrant_client.get_collection(release.collection_name)
    assert set(info.config.params.sparse_vectors or {}) == {"sparse"}
