"""Tests for rag.evaluation.target.materialize_benchmark_target.

This benchmark-corpus mirroring path had no test coverage before phase 5;
phase 5's applied-state runtime change broke its profile-manifest lookup
(parameter_state.collection_name stopped having a corresponding old-style
IndexManifest on disk) and its provenance field access (parameter_state.
attestation is now None on the deployment-resolved path). This test proves
the phase 5 fix -- reading the chunking profile from the resolved release's
build_config instead of an IndexManifest file -- actually works. Release-
aware benchmark *building* (not just profile mirroring) is phase 6.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient

from app_config.catalog import AliasConfig, KBConfig, TaskConfig, catalog_override
from app_config.catalog.schema import AliasBuildConfig, AliasRetrievalConfig
from app_config.runtime import load_settings
from rag.control_plane.models import AliasDeployment
from rag.evaluation.models import BenchmarkPreparedArtifacts
from rag.evaluation.target import materialize_benchmark_target
from rag.indexing.llamaindex_qdrant import QdrantCollectionManager
from rag.indexing.materialize import materialize_release_collection
from rag.runtime import RagRuntime
from rag.sources.bundles import SourceNodeBundle
from tests.rag.control_plane_fakes import FakeAliasDeploymentRepository, FakeReleaseRepository
from tests.rag.test_runtime_service import _Embedding, _node, _Sparse


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


def _catalog_kb() -> KBConfig:
    return KBConfig(
        name="pytorch_reference",
        default_alias="champion",
        aliases={
            "champion": AliasConfig(
                top_k=2, score_threshold=0.1, retrieval_strategy="dense", reranker_multiplier=1
            )
        },
        description="PyTorch docs",
    )


def test_materialize_benchmark_target_with_no_documents_mirrors_release_chunking(
    tmp_path: Path,
) -> None:
    client = QdrantClient(":memory:")
    manager = QdrantCollectionManager(client=client, collection_name="rag__pytorch_reference__c1")
    bundle = SourceNodeBundle(
        kb_id="pytorch_reference",
        source_instance_id="pytorch_reference.docs",
        node_artifact_paths=["chunks/test.json"],
        node_artifact_checksums={"chunks/test.json": "sha256:test"},
        nodes=[_node(document_id="torch.nn.Module", text="torch.nn.Module is the base class.")],
        document_count=1,
        node_count=1,
    )
    release = materialize_release_collection(
        kb_id="pytorch_reference",
        release_id="ragrel_pytorch_reference_c1",
        collection_name=manager.collection_name,
        release_fingerprint="sha256:fp-c1",
        catalog_digest="sha256:catalog",
        build_config_digest="sha256:build",
        source_declaration_digest="sha256:source",
        source_snapshot_id="sha256:snapshot",
        build_config=AliasBuildConfig(
            chunking={"strategy": "sentence", "chunk_size": 256, "chunk_overlap": 32},
            dense_encoder={"model": "test-embedding", "dimension": 3},
        ),
        bundles=[bundle],
        collection_manager=manager,
        embedding_client=_Embedding(),
        rag_data_root=tmp_path,
        source_adapter_versions={},
        source_manifest_digests={},
    )
    release_repo = FakeReleaseRepository()
    release_repo.insert(release, manifest_path="unused.json")
    deployment_repo = FakeAliasDeploymentRepository()
    deployment = AliasDeployment(
        id=uuid4(),
        kb_id="pytorch_reference",
        alias="champion",
        release_id=release.id,
        collection_name=release.collection_name,
        catalog_digest="sha256:catalog",
        build_config_digest=release.build_config_digest,
        retrieval_config_digest="sha256:retrieval",
        retrieval_config=AliasRetrievalConfig(strategy="dense", top_k=2, score_threshold=0.1),
        status="pending",
    )
    deployment_repo.create_pending(deployment)
    deployment_repo.activate(deployment.id, applied_at=datetime.now(timezone.utc))

    runtime = RagRuntime(
        settings=_settings(),
        embedding_service=_Embedding(),
        qdrant_client=client,
        sparse_encoder_factory=_Sparse,
        deployment_repo=deployment_repo,
        release_repo=release_repo,
    )
    kb = _catalog_kb()
    catalog = {
        "code": TaskConfig(task="code", description="Coding help", knowledge_bases=[kb]),
    }

    with catalog_override(catalog):
        target = materialize_benchmark_target(
            runtime=runtime,
            source_instance_id="pytorch_reference.qa_benchmark",
            kb=kb,
            alias="champion",
            artifacts=BenchmarkPreparedArtifacts(),
            rag_data_root=tmp_path,
        )

    assert target.build_profile == {"strategy": "sentence", "chunk_size": 256, "chunk_overlap": 32}
    assert target.parameter_state.manifest_id == release.manifest_id
    assert target.state.collection_name == release.collection_name
