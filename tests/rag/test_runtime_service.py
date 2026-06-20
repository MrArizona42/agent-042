from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from llama_index.core.llms import MockLLM
from llama_index.core.schema import TextNode
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from app_config.catalog import AliasConfig, KBConfig, TaskConfig, catalog_override
from app_config.catalog.schema import AliasBuildConfig, AliasRetrievalConfig
from app_config.runtime import load_settings
from rag.contracts.metadata import node_id_for_chunk
from rag.control_plane.models import AliasDeployment, RagRelease
from rag.evaluation.models import GenerationObservation
from rag.indexing.llamaindex_qdrant import QdrantCollectionManager
from rag.indexing.materialize import materialize_release_collection
from rag.runtime import RagRuntime, RagRuntimeSource
from rag.sources.bundles import SourceNodeBundle
from tests.rag.control_plane_fakes import FakeAliasDeploymentRepository, FakeReleaseRepository


class _Embedding:
    model = "test-embedding"
    dimension = 3

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        if "tensor" in lowered or "module" in lowered:
            return [1.0, 0.0, 0.0]
        if "torch" in lowered:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class _Sparse:
    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        return [
            SparseVector(
                indices=[1] if "tensor" in text.lower() or "module" in text.lower() else [2],
                values=[1.0],
            )
            for text in texts
        ]


def _alias(*, strategy: str) -> AliasConfig:
    return AliasConfig(
        top_k=2,
        score_threshold=0.1,
        retrieval_strategy=strategy,  # type: ignore[arg-type]
        reranker_multiplier=1,
    )


def _catalog() -> dict[str, TaskConfig]:
    kb = KBConfig(
        name="pytorch_reference",
        default_alias="champion",
        aliases={
            "champion": _alias(strategy="dense"),
            "challenger": _alias(strategy="hybrid"),
        },
        description="PyTorch docs",
    )
    return {
        "code": TaskConfig(
            task="code",
            description="Code help",
            knowledge_bases=[kb],
        )
    }


def _settings():
    return load_settings(
        overrides={
            "vllm": {"model": "test-model"},
            "platform": {
                "qdrant_host": "localhost",
                "qdrant_port": 6333,
                "embeddings_url": "http://embeddings:8100",
                "vllm_base_url": "http://vllm:8000",
            },
            "gateway": {"embeddings_timeout": 10.0, "vllm_timeout": 10.0},
            "rag": {
                "embedding_model": "test-embedding",
                "embedding_device": "cpu",
                "build": {"embedding_batch_size": 32, "qdrant_upsert_batch_size": 2},
            },
        }
    )


def _node(*, document_id: str, text: str) -> TextNode:
    chunk_id = f"{document_id}:chunk:0000"
    metadata = {
        "kb_id": "pytorch_reference",
        "source_instance_id": "pytorch_reference.docs",
        "source_document_id": document_id,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "title": document_id,
        "source_uri": f"https://docs.test/{document_id}",
        "section_title": "Overview",
        "section_ordinal": 0,
        "section_level": 1,
        "ordinal": 0,
        "token_count": len(text.split()),
        "adapter_id": "generic.http_html",
        "adapter_version": "1",
    }
    return TextNode(
        id_=node_id_for_chunk(chunk_id),
        text=text,
        metadata=metadata,
        excluded_embed_metadata_keys=list(metadata),
    )


def _bundle() -> SourceNodeBundle:
    nodes = [
        _node(document_id="torch.nn.Module", text="torch.nn.Module is the base class."),
        _node(document_id="torch.Tensor", text="A tensor is a multidimensional array."),
    ]
    return SourceNodeBundle(
        kb_id="pytorch_reference",
        source_instance_id="pytorch_reference.docs",
        node_artifact_paths=["chunks/test.json"],
        node_artifact_checksums={"chunks/test.json": "sha256:test"},
        nodes=nodes,
        document_count=2,
        node_count=2,
    )


def _build_release(
    *,
    client: QdrantClient,
    root: Path,
    collection_name: str,
    capability: str,
) -> RagRelease:
    """Materialize a release directly from a fixed bundle (no real fetch/chunk)."""
    manager = QdrantCollectionManager(client=client, collection_name=collection_name)
    build_config = AliasBuildConfig(
        chunking={"strategy": "sentence", "chunk_size": 512, "chunk_overlap": 64},
        dense_encoder={"model": "test-embedding", "dimension": 3},
        sparse_encoder={"model": "test-sparse"} if capability == "hybrid" else None,
    )
    return materialize_release_collection(
        kb_id="pytorch_reference",
        release_id=f"ragrel_pytorch_reference_{collection_name[-12:]}",
        collection_name=collection_name,
        release_fingerprint=f"sha256:fingerprint-{collection_name}",
        catalog_digest="sha256:catalog",
        build_config_digest="sha256:build",
        source_declaration_digest="sha256:source",
        source_snapshot_id="sha256:snapshot",
        build_config=build_config,
        bundles=[_bundle()],
        collection_manager=manager,
        embedding_client=_Embedding(),
        sparse_encoder_client=_Sparse() if capability == "hybrid" else None,
        rag_data_root=root,
        source_adapter_versions={},
        source_manifest_digests={},
    )


def _activate(
    *,
    release: RagRelease,
    alias: str,
    deployment_repo: FakeAliasDeploymentRepository,
    strategy: str,
) -> None:
    deployment = AliasDeployment(
        id=uuid4(),
        kb_id="pytorch_reference",
        alias=alias,
        release_id=release.id,
        collection_name=release.collection_name,
        catalog_digest="sha256:catalog",
        build_config_digest=release.build_config_digest,
        retrieval_config_digest="sha256:retrieval",
        retrieval_config=AliasRetrievalConfig(strategy=strategy, top_k=2, score_threshold=0.1),
        status="pending",
    )
    deployment_repo.create_pending(deployment)
    deployment_repo.activate(deployment.id, applied_at=datetime.now(timezone.utc))


def _runtime(
    client: QdrantClient,
    *,
    deployment_repo: FakeAliasDeploymentRepository,
    release_repo: FakeReleaseRepository,
) -> RagRuntime:
    return RagRuntime(
        settings=_settings(),
        embedding_service=_Embedding(),
        qdrant_client=client,
        sparse_encoder_factory=_Sparse,
        deployment_repo=deployment_repo,
        release_repo=release_repo,
    )


def _deployed(
    *,
    client: QdrantClient,
    root: Path,
    collection_name: str,
    capability: str,
    alias: str,
    strategy: str,
) -> tuple[RagRelease, FakeAliasDeploymentRepository, FakeReleaseRepository]:
    release = _build_release(
        client=client, root=root, collection_name=collection_name, capability=capability
    )
    release_repo = FakeReleaseRepository()
    release_repo.insert(release, manifest_path="unused.json")
    deployment_repo = FakeAliasDeploymentRepository()
    _activate(release=release, alias=alias, deployment_repo=deployment_repo, strategy=strategy)
    return release, deployment_repo, release_repo


def test_runtime_uses_default_alias_and_returns_native_nodes(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    release, deployment_repo, release_repo = _deployed(
        client=client,
        root=tmp_path,
        collection_name="rag__pytorch_reference__dense",
        capability="dense",
        alias="champion",
        strategy="dense",
    )

    with catalog_override(_catalog()):
        runtime = _runtime(client, deployment_repo=deployment_repo, release_repo=release_repo)
        result = runtime.retrieve(
            query="How do I define a module?",
            sources=[RagRuntimeSource(knowledge_base="pytorch_reference")],
        )

    assert len(result.nodes) == 2
    assert result.nodes[0].node.metadata["qdrant_alias"] == "rag__pytorch_reference__champion"
    assert result.nodes[0].node.metadata["adapter_id"] == "generic.http_html"
    assert result.provenance[0]["collection_name"] == release.collection_name
    assert result.provenance[0]["manifest_id"].startswith("sha256:")
    assert result.provenance[0]["release_id"] == release.id
    assert result.diagnostics["no_hit"] is False


def test_runtime_rejects_live_embedding_provider_model_mismatch(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    _, deployment_repo, release_repo = _deployed(
        client=client,
        root=tmp_path,
        collection_name="rag__pytorch_reference__dense_mismatch",
        capability="dense",
        alias="champion",
        strategy="dense",
    )
    embedding = _Embedding()
    embedding.model = "provider-drifted-model"

    with catalog_override(_catalog()):
        runtime = RagRuntime(
            settings=_settings(),
            embedding_service=embedding,
            qdrant_client=client,
            sparse_encoder_factory=_Sparse,
            deployment_repo=deployment_repo,
            release_repo=release_repo,
        )
        try:
            with pytest.raises(RuntimeError, match="Embedding model mismatch"):
                runtime.resolve_alias_profile(
                    kb_id="pytorch_reference",
                    alias="champion",
                )
        finally:
            runtime.close()
def test_runtime_allows_dense_alias_on_hybrid_collection(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    _, deployment_repo, release_repo = _deployed(
        client=client,
        root=tmp_path,
        collection_name="rag__pytorch_reference__hybrid",
        capability="hybrid",
        alias="champion",
        strategy="dense",
    )

    with catalog_override(_catalog()):
        runtime = _runtime(client, deployment_repo=deployment_repo, release_repo=release_repo)
        result = runtime.retrieve(
            query="module",
            sources=[RagRuntimeSource(knowledge_base="pytorch_reference", alias="champion")],
        )

    assert result.skipped_sources == []
    assert result.nodes


def test_runtime_rejects_hybrid_alias_on_dense_collection(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    _, deployment_repo, release_repo = _deployed(
        client=client,
        root=tmp_path,
        collection_name="rag__pytorch_reference__dense",
        capability="dense",
        alias="challenger",
        strategy="hybrid",
    )

    with catalog_override(_catalog()):
        runtime = _runtime(client, deployment_repo=deployment_repo, release_repo=release_repo)
        result = runtime.retrieve(
            query="module",
            sources=[RagRuntimeSource(knowledge_base="pytorch_reference", alias="challenger")],
        )

    assert result.nodes == []
    assert "hybrid" in result.skipped_sources[0].reason


def test_runtime_uses_explicit_hybrid_alias(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    _, deployment_repo, release_repo = _deployed(
        client=client,
        root=tmp_path,
        collection_name="rag__pytorch_reference__hybrid",
        capability="hybrid",
        alias="challenger",
        strategy="hybrid",
    )

    with catalog_override(_catalog()):
        runtime = _runtime(client, deployment_repo=deployment_repo, release_repo=release_repo)
        result = runtime.retrieve(
            query="module",
            sources=[RagRuntimeSource(knowledge_base="pytorch_reference", alias="challenger")],
        )

    assert result.skipped_sources == []
    assert result.provenance[0]["retrieval_strategy"] == "hybrid"
    assert result.nodes


def test_runtime_marks_resolved_source_with_no_hits(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    _, deployment_repo, release_repo = _deployed(
        client=client,
        root=tmp_path,
        collection_name="rag__pytorch_reference__dense",
        capability="dense",
        alias="champion",
        strategy="dense",
    )

    with catalog_override(_catalog()):
        runtime = _runtime(client, deployment_repo=deployment_repo, release_repo=release_repo)
        result = runtime.retrieve(
            query="unrelated question",
            sources=[RagRuntimeSource(knowledge_base="pytorch_reference")],
        )

    assert result.nodes == []
    assert result.skipped_sources == []
    assert result.provenance[0]["no_hit"] is True
    assert result.diagnostics["no_hit"] is True


def test_runtime_query_engine_returns_answer_sources_and_prompt_identity(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    release, deployment_repo, release_repo = _deployed(
        client=client,
        root=tmp_path,
        collection_name="rag__pytorch_reference__dense",
        capability="dense",
        alias="champion",
        strategy="dense",
    )

    with catalog_override(_catalog()):
        runtime = _runtime(client, deployment_repo=deployment_repo, release_repo=release_repo)
        result = runtime.query(
            query="How do I define a module?",
            source=RagRuntimeSource(knowledge_base="pytorch_reference"),
            llm=MockLLM(max_tokens=32),
        )

    assert result.answer
    assert result.source_nodes
    assert {node.node.metadata["document_id"] for node in result.source_nodes} == {
        "torch.nn.Module",
        "torch.Tensor",
    }
    assert result.prompt_identity.prompt_id == "rag.query.default"
    assert result.prompt_identity.prompt_digest.startswith("sha256:")
    assert result.provenance["collection_name"] == release.collection_name

    observation = GenerationObservation(**result.prompt_identity.model_dump())
    assert observation.prompt_id == "rag.query.default"
    assert observation.prompt_version == "1"
    assert observation.prompt_params == {"response_mode": "compact"}


def test_runtime_reports_empty_query_diagnostics() -> None:
    client = QdrantClient(":memory:")
    deployment_repo = FakeAliasDeploymentRepository()
    release_repo = FakeReleaseRepository()
    with catalog_override(_catalog()):
        runtime = _runtime(client, deployment_repo=deployment_repo, release_repo=release_repo)
        result = runtime.retrieve(
            query=" ",
            sources=[RagRuntimeSource(knowledge_base="pytorch_reference")],
        )

    assert result.nodes == []
    assert result.diagnostics == {
        "requested_source_count": 1,
        "resolved_source_count": 0,
        "skipped_source_count": 0,
        "hit_count": 0,
        "no_hit": True,
    }
