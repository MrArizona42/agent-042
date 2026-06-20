"""Build an immutable, content-identified RagRelease for one KB alias.

This orchestrates fetch/extract/chunk (scoped by transformation digest),
bundling, fingerprinting, and Qdrant materialization. It has no Postgres
dependency and no advisory locking: those land in phase 4 once the control-
plane registry exists. Idempotent reuse here is filesystem-only -- if a
release manifest for the computed release id already exists, it is returned
without rebuilding.
"""

from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, ContextManager

from app_config.catalog.schema import AliasBuildConfig, CatalogConfig
from app_config.catalog.source_instances import (
    SourceInstanceIndex,
    build_source_instance_index,
    conventional_manifest_path,
)
from rag.adapters import SourceAdapterRegistry
from rag.contracts.manifests import read_release_manifest, release_manifest_path
from rag.control_plane import fingerprints as fp
from rag.control_plane.models import RagRelease
from rag.indexing.llamaindex_qdrant import QdrantCollectionManager
from rag.indexing.materialize import (
    EmbeddingClient,
    SparseEmbeddingClient,
    materialize_release_collection,
)
from rag.sources.build import resolve_adapter_ref
from rag.sources.bundles import collect_source_nodes
from rag.sources.cache import sha256_bytes
from rag.sources.chunks import ChunkingConfig, chunk_source_instance
from rag.sources.processing import process_source_instance


def _internal_chunking_config(build_config: AliasBuildConfig) -> ChunkingConfig:
    return ChunkingConfig(
        chunk_size=build_config.chunking.chunk_size,
        chunk_overlap=build_config.chunking.chunk_overlap,
    )


def source_declaration(
    *,
    source_instance_ids: list[str],
    source_index: SourceInstanceIndex,
    rag_data_root: Path | str,
) -> tuple[str, dict[str, str], dict[str, str]]:
    """Return (source_declaration_digest, manifest_digests, adapter_versions)."""
    entries: list[tuple[str, str, str, str]] = []
    manifest_digests: dict[str, str] = {}
    adapter_versions: dict[str, str] = {}
    for source_instance_id in source_instance_ids:
        instance = source_index.get(source_instance_id)
        manifest_path = conventional_manifest_path(rag_data_root, source_instance_id)
        digest = (
            sha256_bytes(manifest_path.read_bytes())
            if manifest_path.exists()
            else "sha256:" + "0" * 64
        )
        manifest_digests[source_instance_id] = digest
        adapter_versions[source_instance_id] = f"{instance.adapter.id}@{instance.adapter.version}"
        entries.append((source_instance_id, digest, instance.adapter.id, instance.adapter.version))
    digest = fp.source_declaration_digest(entries)
    return digest, manifest_digests, adapter_versions


def build_release(
    *,
    kb_id: str,
    build_config: AliasBuildConfig,
    catalog_digest: str,
    catalog_cfg: CatalogConfig,
    rag_data_root: Path | str,
    collection_manager_factory,
    embedding_client: EmbeddingClient,
    sparse_encoder_client: SparseEmbeddingClient | None = None,
    adapter_registry: SourceAdapterRegistry | None = None,
    source_instance_ids: list[str] | None = None,
    document_ids: list[str] | None = None,
    limit: int | None = None,
    force_fetch: bool = False,
    force_extract: bool = False,
    force_chunk: bool = False,
    qdrant_upsert_batch_size: int = 128,
    created_at: datetime | None = None,
    release_lock_factory: Callable[[str], ContextManager[None]] | None = None,
    register_release: Callable[[RagRelease], RagRelease] | None = None,
) -> RagRelease:
    """Resolve sources, materialize, and return the immutable release for *kb_id*.

    *collection_manager_factory* takes a content-addressed collection name and
    returns a connected `QdrantCollectionManager`-compatible object; the
    caller owns its lifecycle (close it after this returns).
    """
    source_index = build_source_instance_index(catalog_cfg)
    resolved_source_ids = source_instance_ids or [
        instance.id for instance in source_index.corpus_for_kb(kb_id)
    ]
    if not resolved_source_ids:
        raise ValueError(f"KB '{kb_id}' has no corpus source instances to build a release from")

    source_declaration_digest, manifest_digests, adapter_versions = source_declaration(
        source_instance_ids=resolved_source_ids,
        source_index=source_index,
        rag_data_root=rag_data_root,
    )

    transformation_digest = fp.transformation_digest(build_config.chunking)
    internal_chunking = _internal_chunking_config(build_config)
    for source_instance_id in resolved_source_ids:
        instance = source_index.get(source_instance_id)
        source_adapter = resolve_adapter_ref(
            catalog_cfg,
            adapter_id=instance.adapter.id,
            version=instance.adapter.version,
            adapter_registry=adapter_registry,
        )
        process_source_instance(
            kb_id=kb_id,
            source_instance_id=source_instance_id,
            manifest_path=conventional_manifest_path(rag_data_root, source_instance_id),
            rag_data_root=rag_data_root,
            source_adapter=source_adapter,
            document_ids=document_ids,
            limit=limit,
            force_fetch=force_fetch,
            force_extract=force_extract,
        )
        chunk_source_instance(
            rag_data_root=rag_data_root,
            kb_id=kb_id,
            source_instance_id=source_instance_id,
            document_ids=document_ids,
            limit=limit,
            config=internal_chunking,
            force=force_chunk,
            transformation_digest=transformation_digest,
        )

    bundles = [
        collect_source_nodes(
            rag_data_root=rag_data_root,
            kb_id=kb_id,
            source_instance_id=source_instance_id,
            document_ids=document_ids,
            limit=limit,
            transformation_digest=transformation_digest,
        )
        for source_instance_id in resolved_source_ids
    ]

    snapshot_entries = [
        (bundle.source_instance_id, checksum)
        for bundle in bundles
        for checksum in bundle.node_artifact_checksums.values()
    ]
    snapshot_id = fp.source_snapshot_id(snapshot_entries)
    build_digest = fp.build_config_digest(build_config)
    fingerprint = fp.release_fingerprint(
        kb_id=kb_id,
        build_config_digest=build_digest,
        source_declaration_digest=source_declaration_digest,
        source_snapshot_id=snapshot_id,
    )
    release_id = fp.release_id(kb_id, fingerprint)
    collection_name = fp.collection_name(kb_id, fingerprint)

    lock = release_lock_factory(fingerprint) if release_lock_factory else nullcontext()
    with lock:
        existing_path = release_manifest_path(
            rag_data_root=rag_data_root, kb_id=kb_id, release_id=release_id
        )
        if existing_path.exists():
            release = read_release_manifest(existing_path)
        else:
            collection_manager = collection_manager_factory(collection_name)
            try:
                release = materialize_release_collection(
                    kb_id=kb_id,
                    release_id=release_id,
                    collection_name=collection_name,
                    release_fingerprint=fingerprint,
                    catalog_digest=catalog_digest,
                    build_config_digest=build_digest,
                    source_declaration_digest=source_declaration_digest,
                    source_snapshot_id=snapshot_id,
                    build_config=build_config,
                    bundles=bundles,
                    collection_manager=collection_manager,
                    embedding_client=embedding_client,
                    sparse_encoder_client=sparse_encoder_client,
                    rag_data_root=rag_data_root,
                    source_adapter_versions=adapter_versions,
                    source_manifest_digests=manifest_digests,
                    qdrant_upsert_batch_size=qdrant_upsert_batch_size,
                    created_at=created_at or datetime.now(tz=UTC),
                )
            except Exception:
                if collection_manager.collection_exists():
                    collection_manager.client.delete_collection(collection_name)
                if existing_path.exists():
                    existing_path.unlink()
                raise
        return register_release(release) if register_release else release


def qdrant_collection_manager_factory(*, host: str, port: int):
    """Return a `collection_manager_factory` bound to a Qdrant host/port."""

    def _factory(collection_name: str) -> QdrantCollectionManager:
        return QdrantCollectionManager.connect(
            host=host,
            port=port,
            collection_name=collection_name,
        )

    return _factory
