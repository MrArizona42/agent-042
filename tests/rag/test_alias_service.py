"""Tests for rag.control_plane.alias_service.AliasService.

Exercises the full diff/apply reconciliation table from the declarative
alias workflow plan against the phase 3 fake repositories and a real
in-memory Qdrant client (so release builds actually materialize, just like
tests/rag/test_release_builder.py). Embedding/sparse/reranker providers are
fakes since no live provider service exists in this test environment.
"""

from __future__ import annotations

import tomllib
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

import httpx
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from app_config.catalog.schema import CatalogConfig
from rag.adapters import ManifestSourceAdapter, SourceAdapterRegistry
from rag.control_plane.alias_service import (
    AliasApplyError,
    AliasApplyRequest,
    AliasDiffRequest,
    AliasService,
)
from rag.indexing.llamaindex_qdrant import QdrantCollectionManager
from rag.indexing.materialize import qdrant_alias_name
from rag.sources.extractors import HtmlDocsExtractor
from rag.sources.fetchers import HtmlDocsFetcher
from tests.rag.control_plane_fakes import (
    FakeAliasDeploymentRepository,
    FakeReleaseBuildRepository,
    FakeReleaseRepository,
)


class _EmbeddingClient:
    model = "test-embedding"
    dimension = 3

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [1.0, 0.0, 0.0] if "tensor" in text.lower() else [0.0, 1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class _SparseClient:
    model = "test-sparse"

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        return [SparseVector(indices=[1], values=[1.0]) for _ in texts]


class _RerankerClient:
    def __init__(self, model: str) -> None:
        self.model = model


def _html_client() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        page_id = request.url.path.rsplit("/", 1)[-1].removesuffix(".html")
        content = (
            f"<html><body><h1>{page_id.title()}</h1>"
            f"<p>{page_id} body text. More useful text for chunking.</p></body></html>"
        ).encode("utf-8")
        return httpx.Response(
            200, content=content, headers={"content-type": "text/html"}, request=request
        )

    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def _registry() -> SourceAdapterRegistry:
    registry = SourceAdapterRegistry()
    registry.register(
        ManifestSourceAdapter(
            adapter_id="generic.http_html",
            version="1",
            default_uri_prefix="http_html",
            _fetcher_factory=lambda: HtmlDocsFetcher(client=_html_client()),
            _extractor_factory=HtmlDocsExtractor,
        )
    )
    return registry


def _setup_source_manifest(rag_data_root: Path) -> None:
    manifest_dir = rag_data_root / "source_instances" / "pytorch_reference.docs"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "manifest.toml").write_text(
        dedent(
            """
            schema_version = 1
            [[documents]]
            id = "tensors"
            title = "Tensors"
            url = "https://docs.test/tensors.html"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _catalog_config(
    *,
    challenger_strategy: str = "hybrid",
    challenger_reranker: str | None = "cross-encoder/x",
) -> CatalogConfig:
    reranker_line = f'reranker = "{challenger_reranker}"\n' if challenger_reranker else ""
    reranker_multiplier = 4 if challenger_reranker else 1
    text = dedent(
        """
        schema_version = 4

        [[knowledge_bases]]
        id = "pytorch_reference"
        description = "PyTorch docs"
        default_alias = "champion"

        [knowledge_bases.aliases.champion.build.chunking]
        strategy = "sentence"
        chunk_size = 512
        chunk_overlap = 64

        [knowledge_bases.aliases.champion.build.dense_encoder]
        model = "test-embedding"
        dimension = 3

        [knowledge_bases.aliases.champion.retrieve]
        strategy = "dense"
        top_k = 5
        score_threshold = 0.35
        reranker_multiplier = 1

        [knowledge_bases.aliases.challenger.build.chunking]
        strategy = "sentence"
        chunk_size = 512
        chunk_overlap = 64

        [knowledge_bases.aliases.challenger.build.dense_encoder]
        model = "test-embedding"
        dimension = 3
        """
    ).strip()
    if challenger_strategy in ("hybrid", "sparse"):
        text += dedent(
            """

            [knowledge_bases.aliases.challenger.build.sparse_encoder]
            model = "test-sparse"
            """
        )
    text += dedent(
        f"""

        [knowledge_bases.aliases.challenger.retrieve]
        strategy = "{challenger_strategy}"
        top_k = 5
        score_threshold = 0.01
        {reranker_line}reranker_multiplier = {reranker_multiplier}

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
    )
    return CatalogConfig(**tomllib.loads(text))


@pytest.fixture()
def qdrant_client() -> QdrantClient:
    return QdrantClient(":memory:")


class _NonClosingCollectionManager(QdrantCollectionManager):
    """Test-only manager that doesn't close the shared in-memory client.

    Several `AliasService` operations construct and close their own manager
    per call (each owns its connection in production); tests share one
    `:memory:` QdrantClient across the whole test for speed, so closing it
    on every call would break every subsequent operation.
    """

    def close(self) -> None:
        return None


def _service(
    *,
    tmp_path: Path,
    qdrant_client: QdrantClient,
    catalog_cfg: CatalogConfig,
    evaluation_coverage_checker=None,
) -> AliasService:
    def _manager_factory(collection_name: str) -> QdrantCollectionManager:
        return _NonClosingCollectionManager(client=qdrant_client, collection_name=collection_name)

    def _alias_updater(kb_id: str, alias: str, collection_name: str) -> None:
        manager = _manager_factory(collection_name)
        manager.update_alias(qdrant_alias_name(kb_id=kb_id, alias=alias), collection_name)

    return AliasService(
        catalog_cfg=catalog_cfg,
        rag_data_root=tmp_path,
        release_build_repo=FakeReleaseBuildRepository(),
        release_repo=FakeReleaseRepository(),
        deployment_repo=FakeAliasDeploymentRepository(),
        collection_manager_factory=_manager_factory,
        qdrant_alias_updater=_alias_updater,
        embedding_client_factory=_EmbeddingClient,
        sparse_encoder_client_factory=_SparseClient,
        reranker_client_factory=_RerankerClient,
        adapter_registry=_registry(),
        evaluation_coverage_checker=evaluation_coverage_checker,
        clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


class TestBootstrapApply:
    def test_apply_non_default_alias_builds_and_activates(self, tmp_path, qdrant_client):
        _setup_source_manifest(tmp_path)
        service = _service(
            tmp_path=tmp_path, qdrant_client=qdrant_client, catalog_cfg=_catalog_config()
        )

        result = service.apply(AliasApplyRequest(kb_id="pytorch_reference", alias="challenger"))

        assert result.action == "built_release"
        assert result.deployment.status == "active"
        assert result.release.build_config.sparse_encoder is not None
        assert qdrant_client.collection_exists(result.release.collection_name)

    def test_default_alias_bootstrap_refuses_without_override(self, tmp_path, qdrant_client):
        _setup_source_manifest(tmp_path)
        service = _service(
            tmp_path=tmp_path, qdrant_client=qdrant_client, catalog_cfg=_catalog_config()
        )

        with pytest.raises(AliasApplyError, match="allow_build_default"):
            service.apply(AliasApplyRequest(kb_id="pytorch_reference", alias="champion"))

    def test_default_alias_bootstrap_with_override_still_needs_evaluation_override(
        self, tmp_path, qdrant_client
    ):
        _setup_source_manifest(tmp_path)
        service = _service(
            tmp_path=tmp_path, qdrant_client=qdrant_client, catalog_cfg=_catalog_config()
        )

        with pytest.raises(AliasApplyError, match="evaluation coverage"):
            service.apply(
                AliasApplyRequest(
                    kb_id="pytorch_reference", alias="champion", allow_build_default=True
                )
            )

    def test_default_alias_bootstrap_with_both_overrides_succeeds(self, tmp_path, qdrant_client):
        _setup_source_manifest(tmp_path)
        service = _service(
            tmp_path=tmp_path, qdrant_client=qdrant_client, catalog_cfg=_catalog_config()
        )

        result = service.apply(
            AliasApplyRequest(
                kb_id="pytorch_reference",
                alias="champion",
                allow_build_default=True,
                allow_unevaluated=True,
            )
        )

        assert result.action == "built_release"
        assert result.deployment.status == "active"
        assert result.deployment.details == {
            "overrides": {"allow_unevaluated": True, "allow_build_default": True}
        }
        attempts = service._release_build_repo.list_for_kb("pytorch_reference")
        assert attempts[0].details == {
            "overrides": {"allow_unevaluated": True, "allow_build_default": True}
        }


class TestNoDriftAndRetrievalOnly:
    def test_no_drift_apply_is_idempotent(self, tmp_path, qdrant_client):
        _setup_source_manifest(tmp_path)
        service = _service(
            tmp_path=tmp_path, qdrant_client=qdrant_client, catalog_cfg=_catalog_config()
        )
        first = service.apply(AliasApplyRequest(kb_id="pytorch_reference", alias="challenger"))

        second = service.apply(AliasApplyRequest(kb_id="pytorch_reference", alias="challenger"))

        assert second.action == "no_drift"
        assert second.deployment.id == first.deployment.id

    def test_refresh_sources_re_fetches_even_without_drift(self, tmp_path, qdrant_client):
        _setup_source_manifest(tmp_path)
        fetch_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            fetch_count["n"] += 1
            page_id = request.url.path.rsplit("/", 1)[-1].removesuffix(".html")
            content = (
                f"<html><body><h1>{page_id.title()}</h1>"
                f"<p>{page_id} body text. More useful text for chunking.</p></body></html>"
            ).encode("utf-8")
            return httpx.Response(
                200, content=content, headers={"content-type": "text/html"}, request=request
            )

        registry = SourceAdapterRegistry()
        registry.register(
            ManifestSourceAdapter(
                adapter_id="generic.http_html",
                version="1",
                default_uri_prefix="http_html",
                _fetcher_factory=lambda: HtmlDocsFetcher(
                    client=httpx.Client(
                        transport=httpx.MockTransport(handler), follow_redirects=True
                    )
                ),
                _extractor_factory=HtmlDocsExtractor,
            )
        )
        service = _service(
            tmp_path=tmp_path, qdrant_client=qdrant_client, catalog_cfg=_catalog_config()
        )
        service._adapter_registry = registry
        service.apply(AliasApplyRequest(kb_id="pytorch_reference", alias="challenger"))
        fetched_after_first_apply = fetch_count["n"]
        assert fetched_after_first_apply > 0

        result = service.apply(
            AliasApplyRequest(kb_id="pytorch_reference", alias="challenger", refresh_sources=True)
        )

        assert fetch_count["n"] > fetched_after_first_apply
        # Content didn't actually change, so build_release() resolves the
        # same release by fingerprint and _activate() finds the deployment
        # already matches it exactly -- "no_drift" is the accurate label,
        # not a sign refresh_sources had no effect (the assertion above is
        # what actually proves the re-fetch happened).
        assert result.action == "no_drift"

    def test_diff_reports_no_drift_after_apply(self, tmp_path, qdrant_client):
        _setup_source_manifest(tmp_path)
        service = _service(
            tmp_path=tmp_path, qdrant_client=qdrant_client, catalog_cfg=_catalog_config()
        )
        service.apply(AliasApplyRequest(kb_id="pytorch_reference", alias="challenger"))

        diff = service.diff(AliasDiffRequest(kb_id="pytorch_reference", alias="challenger"))

        assert diff.build_drift is False
        assert diff.retrieval_drift is False
        assert diff.source_declaration_drift is False

    def test_retrieval_only_drift_reuses_release_without_rebuilding(self, tmp_path, qdrant_client):
        _setup_source_manifest(tmp_path)
        catalog_cfg = _catalog_config(challenger_strategy="hybrid", challenger_reranker=None)
        service = _service(tmp_path=tmp_path, qdrant_client=qdrant_client, catalog_cfg=catalog_cfg)
        first = service.apply(AliasApplyRequest(kb_id="pytorch_reference", alias="challenger"))

        # Same build; only top_k changes in the alias's desired retrieve block.
        kb = service._catalog_cfg.knowledge_bases[0]
        challenger = kb.aliases["challenger"]
        kb.aliases["challenger"] = challenger.model_copy(
            update={"retrieve": challenger.retrieve.model_copy(update={"top_k": 10})}
        )

        result = service.apply(AliasApplyRequest(kb_id="pytorch_reference", alias="challenger"))

        assert result.action == "retrieval_only"
        assert result.release.id == first.release.id
        assert result.deployment.retrieval_config.top_k == 10


class TestBuildSourceDriftReuse:
    def test_reuses_existing_release_for_a_second_alias_with_identical_build(
        self, tmp_path, qdrant_client
    ):
        _setup_source_manifest(tmp_path)
        # champion and challenger here intentionally share an identical dense-only
        # build (challenger configured dense-only) so the second apply should
        # reuse rather than rebuild.
        catalog_cfg = _catalog_config(challenger_strategy="dense", challenger_reranker=None)
        service = _service(tmp_path=tmp_path, qdrant_client=qdrant_client, catalog_cfg=catalog_cfg)

        challenger_result = service.apply(
            AliasApplyRequest(kb_id="pytorch_reference", alias="challenger")
        )
        champion_result = service.apply(
            AliasApplyRequest(
                kb_id="pytorch_reference",
                alias="champion",
                allow_build_default=True,
                allow_unevaluated=True,
            )
        )

        assert champion_result.action == "reused_release"
        assert champion_result.release.id == challenger_result.release.id

    def test_ambiguous_reusable_releases_refuses_without_explicit_release(
        self, tmp_path, qdrant_client
    ):
        _setup_source_manifest(tmp_path)
        catalog_cfg = _catalog_config(challenger_strategy="dense", challenger_reranker=None)
        service = _service(tmp_path=tmp_path, qdrant_client=qdrant_client, catalog_cfg=catalog_cfg)
        first = service.apply(AliasApplyRequest(kb_id="pytorch_reference", alias="challenger"))

        # Fabricate a second release with the same build/source identity but a
        # different (simulated drifted) content snapshot, registered directly.
        duplicate = first.release.model_copy(
            update={
                "id": "ragrel_pytorch_reference_dup000000",
                "collection_name": "rag__pytorch_reference__dup000000",
                "manifest_id": "sha256:" + "1" * 64,
                "release_fingerprint": "sha256:" + "2" * 64,
                "source_snapshot_id": "sha256:" + "3" * 64,
            }
        )
        service._release_repo.insert(duplicate, manifest_path="dup.json")

        with pytest.raises(AliasApplyError, match="multiple releases"):
            service.apply(
                AliasApplyRequest(
                    kb_id="pytorch_reference",
                    alias="champion",
                    allow_build_default=True,
                    allow_unevaluated=True,
                )
            )

    def test_explicit_release_disambiguates(self, tmp_path, qdrant_client):
        _setup_source_manifest(tmp_path)
        catalog_cfg = _catalog_config(challenger_strategy="dense", challenger_reranker=None)
        service = _service(tmp_path=tmp_path, qdrant_client=qdrant_client, catalog_cfg=catalog_cfg)
        first = service.apply(AliasApplyRequest(kb_id="pytorch_reference", alias="challenger"))
        duplicate = first.release.model_copy(
            update={
                "id": "ragrel_pytorch_reference_dup000000",
                "collection_name": "rag__pytorch_reference__dup000000",
                "manifest_id": "sha256:" + "1" * 64,
                "release_fingerprint": "sha256:" + "2" * 64,
                "source_snapshot_id": "sha256:" + "3" * 64,
            }
        )
        service._release_repo.insert(duplicate, manifest_path="dup.json")

        result = service.apply(
            AliasApplyRequest(
                kb_id="pytorch_reference",
                alias="champion",
                release_id=first.release.id,
                allow_build_default=True,
                allow_unevaluated=True,
            )
        )

        assert result.release.id == first.release.id

    def test_explicit_release_mismatching_build_is_refused(self, tmp_path, qdrant_client):
        _setup_source_manifest(tmp_path)
        catalog_cfg = _catalog_config(challenger_strategy="dense", challenger_reranker=None)
        service = _service(tmp_path=tmp_path, qdrant_client=qdrant_client, catalog_cfg=catalog_cfg)
        challenger = service.apply(AliasApplyRequest(kb_id="pytorch_reference", alias="challenger"))
        mismatched = challenger.release.model_copy(
            update={
                "id": "ragrel_pytorch_reference_other00000",
                "collection_name": "rag__pytorch_reference__other00000",
                "manifest_id": "sha256:" + "4" * 64,
                "release_fingerprint": "sha256:" + "5" * 64,
                "build_config_digest": "sha256:" + "6" * 64,
            }
        )
        service._release_repo.insert(mismatched, manifest_path="other.json")

        with pytest.raises(AliasApplyError, match="does not match desired build"):
            service.apply(
                AliasApplyRequest(
                    kb_id="pytorch_reference",
                    alias="champion",
                    release_id=mismatched.id,
                    allow_build_default=True,
                    allow_unevaluated=True,
                )
            )


class TestRetrievalStrategyCompatibility:
    def test_sparse_strategy_incompatible_with_dense_only_release_is_refused(
        self, tmp_path, qdrant_client
    ):
        _setup_source_manifest(tmp_path)
        catalog_cfg = _catalog_config(challenger_strategy="dense", challenger_reranker=None)
        service = _service(tmp_path=tmp_path, qdrant_client=qdrant_client, catalog_cfg=catalog_cfg)
        dense_release = service.apply(
            AliasApplyRequest(kb_id="pytorch_reference", alias="challenger")
        ).release

        # Champion normally wants dense (matching dense_release); force it to
        # want sparse instead so the explicit dense-only release is rejected
        # for retrieval-strategy/encoder incompatibility, not a build mismatch.
        kb = service._catalog_cfg.knowledge_bases[0]
        champion = kb.aliases["champion"]
        kb.aliases["champion"] = champion.model_copy(
            update={"retrieve": champion.retrieve.model_copy(update={"strategy": "sparse"})}
        )

        with pytest.raises(AliasApplyError, match="incompatible"):
            service.apply(
                AliasApplyRequest(
                    kb_id="pytorch_reference",
                    alias="champion",
                    release_id=dense_release.id,
                    allow_build_default=True,
                    allow_unevaluated=True,
                )
            )


class TestEvaluationGate:
    def test_default_alias_retrieval_only_drift_requires_matching_evaluation(
        self, tmp_path, qdrant_client
    ):
        _setup_source_manifest(tmp_path)
        catalog_cfg = _catalog_config(challenger_strategy="dense", challenger_reranker=None)
        service = _service(
            tmp_path=tmp_path,
            qdrant_client=qdrant_client,
            catalog_cfg=catalog_cfg,
            evaluation_coverage_checker=lambda kb_id, release_id, retrieval_digest: False,
        )
        first = service.apply(
            AliasApplyRequest(
                kb_id="pytorch_reference",
                alias="champion",
                allow_build_default=True,
                allow_unevaluated=True,
            )
        )
        kb = service._catalog_cfg.knowledge_bases[0]
        champion = kb.aliases["champion"]
        kb.aliases["champion"] = champion.model_copy(
            update={"retrieve": champion.retrieve.model_copy(update={"top_k": 10})}
        )

        with pytest.raises(AliasApplyError, match="evaluation coverage"):
            service.apply(AliasApplyRequest(kb_id="pytorch_reference", alias="champion"))

        active = service._deployment_repo.get_active(
            kb_id="pytorch_reference", alias="champion"
        )
        assert active.id == first.deployment.id
        assert active.retrieval_config.top_k == 5

    def test_default_alias_retrieval_only_drift_accepts_explicit_override(
        self, tmp_path, qdrant_client
    ):
        _setup_source_manifest(tmp_path)
        catalog_cfg = _catalog_config(challenger_strategy="dense", challenger_reranker=None)
        service = _service(
            tmp_path=tmp_path,
            qdrant_client=qdrant_client,
            catalog_cfg=catalog_cfg,
            evaluation_coverage_checker=lambda kb_id, release_id, retrieval_digest: False,
        )
        first = service.apply(
            AliasApplyRequest(
                kb_id="pytorch_reference",
                alias="champion",
                allow_build_default=True,
                allow_unevaluated=True,
            )
        )
        kb = service._catalog_cfg.knowledge_bases[0]
        champion = kb.aliases["champion"]
        kb.aliases["champion"] = champion.model_copy(
            update={"retrieve": champion.retrieve.model_copy(update={"top_k": 10})}
        )

        result = service.apply(
            AliasApplyRequest(
                kb_id="pytorch_reference", alias="champion", allow_unevaluated=True
            )
        )

        assert result.action == "retrieval_only"
        assert result.release.id == first.release.id
        assert result.deployment.retrieval_config.top_k == 10

    def test_default_alias_apply_succeeds_when_evaluation_coverage_exists(
        self, tmp_path, qdrant_client
    ):
        _setup_source_manifest(tmp_path)
        catalog_cfg = _catalog_config(challenger_strategy="dense", challenger_reranker=None)
        service = _service(
            tmp_path=tmp_path,
            qdrant_client=qdrant_client,
            catalog_cfg=catalog_cfg,
            evaluation_coverage_checker=lambda kb_id, release_id, retrieval_digest: True,
        )
        challenger = service.apply(AliasApplyRequest(kb_id="pytorch_reference", alias="challenger"))

        result = service.apply(
            AliasApplyRequest(kb_id="pytorch_reference", alias="champion", allow_build_default=True)
        )

        assert result.release.id == challenger.release.id
        assert result.deployment.status == "active"

    def test_default_alias_apply_refused_when_evaluation_coverage_missing(
        self, tmp_path, qdrant_client
    ):
        _setup_source_manifest(tmp_path)
        catalog_cfg = _catalog_config(challenger_strategy="dense", challenger_reranker=None)
        service = _service(
            tmp_path=tmp_path,
            qdrant_client=qdrant_client,
            catalog_cfg=catalog_cfg,
            evaluation_coverage_checker=lambda kb_id, release_id, retrieval_digest: False,
        )
        service.apply(AliasApplyRequest(kb_id="pytorch_reference", alias="challenger"))

        with pytest.raises(AliasApplyError, match="evaluation coverage"):
            service.apply(
                AliasApplyRequest(
                    kb_id="pytorch_reference", alias="champion", allow_build_default=True
                )
            )


class TestProviderIdentityMismatch:
    def test_apply_refuses_on_dense_encoder_mismatch(self, tmp_path, qdrant_client):
        _setup_source_manifest(tmp_path)
        catalog_cfg = _catalog_config(challenger_strategy="dense", challenger_reranker=None)

        def _bad_embedding_client():
            client = _EmbeddingClient()
            client.model = "wrong-model"
            return client

        def _manager_factory(collection_name: str) -> QdrantCollectionManager:
            return _NonClosingCollectionManager(
                client=qdrant_client, collection_name=collection_name
            )

        service = AliasService(
            catalog_cfg=catalog_cfg,
            rag_data_root=tmp_path,
            release_build_repo=FakeReleaseBuildRepository(),
            release_repo=FakeReleaseRepository(),
            deployment_repo=FakeAliasDeploymentRepository(),
            collection_manager_factory=_manager_factory,
            qdrant_alias_updater=lambda *a: None,
            embedding_client_factory=_bad_embedding_client,
            adapter_registry=_registry(),
        )

        diff = service.diff(AliasDiffRequest(kb_id="pytorch_reference", alias="challenger"))
        assert diff.provider_mismatches

        with pytest.raises(AliasApplyError, match="provider identity mismatch"):
            service.apply(AliasApplyRequest(kb_id="pytorch_reference", alias="challenger"))


class TestQdrantMirrorFailureLeavesActiveDeploymentIntact:
    def test_failed_mirror_update_does_not_disturb_active_deployment(self, tmp_path, qdrant_client):
        _setup_source_manifest(tmp_path)
        catalog_cfg = _catalog_config(challenger_strategy="dense", challenger_reranker=None)
        service = _service(tmp_path=tmp_path, qdrant_client=qdrant_client, catalog_cfg=catalog_cfg)
        first = service.apply(AliasApplyRequest(kb_id="pytorch_reference", alias="challenger"))

        def _failing_updater(kb_id: str, alias: str, collection_name: str) -> None:
            raise RuntimeError("qdrant unavailable")

        service._qdrant_alias_updater = _failing_updater
        # Force drift by editing the in-memory catalog's retrieve block.
        kb = service._catalog_cfg.knowledge_bases[0]
        new_retrieve = kb.aliases["challenger"].retrieve.model_copy(update={"top_k": 99})
        kb.aliases["challenger"] = kb.aliases["challenger"].model_copy(
            update={"retrieve": new_retrieve}
        )

        with pytest.raises(AliasApplyError, match="Qdrant mirror"):
            service.apply(AliasApplyRequest(kb_id="pytorch_reference", alias="challenger"))

        active = service._deployment_repo.get_active(kb_id="pytorch_reference", alias="challenger")
        assert active.id == first.deployment.id
        assert active.retrieval_config.top_k == 5
