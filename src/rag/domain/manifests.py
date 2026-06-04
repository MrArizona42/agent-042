"""Manifest artifact and collection-attestation helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rag.domain.models import (
    CollectionAttestation,
    IndexManifest,
    ManifestComparison,
    RetrievalCapability,
)
from rag.ops.meta import BuildConfig, CollectionMeta


def canonical_manifest_payload(manifest: IndexManifest) -> dict[str, Any]:
    """Return the stable JSON payload used for manifest hashing."""
    return manifest.model_dump(
        mode="json",
        exclude={"manifest_id"},
        exclude_none=True,
    )


def compute_manifest_id(manifest: IndexManifest) -> str:
    """Compute a deterministic id from the manifest payload."""
    payload = canonical_manifest_payload(manifest)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def with_manifest_id(manifest: IndexManifest) -> IndexManifest:
    """Return a copy of *manifest* with its deterministic manifest id set."""
    manifest_id = compute_manifest_id(manifest)
    return manifest.model_copy(update={"manifest_id": manifest_id})


def manifest_path(
    *,
    rag_data_root: Path | str,
    kb_id: str,
    collection_name: str,
) -> Path:
    """Return the conventional artifact path for a collection manifest."""
    return Path(rag_data_root) / kb_id / "manifests" / f"{collection_name}.json"


def write_index_manifest(path: Path | str, manifest: IndexManifest) -> IndexManifest:
    """Write a manifest JSON artifact and return the id-bearing manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = with_manifest_id(manifest)
    path.write_text(
        json.dumps(
            manifest.model_dump(mode="json", exclude_none=True),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def read_index_manifest(path: Path | str) -> IndexManifest:
    """Read and validate a manifest JSON artifact."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return IndexManifest.model_validate(payload)


def compare_manifest_attestation(
    manifest: IndexManifest,
    attestation: CollectionAttestation,
) -> ManifestComparison:
    """Compare manifest artifact provenance to Qdrant runtime attestation."""
    manifest = with_manifest_id(manifest) if manifest.manifest_id is None else manifest
    expected = manifest.to_attestation()
    mismatches: dict[str, tuple[Any, Any]] = {}
    for field_name in (
        "manifest_id",
        "kb_id",
        "collection_name",
        "embedding_model",
        "sparse_encoder",
        "retrieval_capability",
        "chunk_count",
    ):
        expected_value = getattr(expected, field_name)
        actual_value = getattr(attestation, field_name)
        if expected_value != actual_value:
            mismatches[field_name] = (expected_value, actual_value)
    return ManifestComparison(matches=not mismatches, mismatches=mismatches)


def attestation_payload(attestation: CollectionAttestation) -> dict[str, Any]:
    """Return the Qdrant payload for collection metadata."""
    return {
        "metadata_kind": "collection_attestation",
        **attestation.model_dump(mode="json", exclude_none=True),
    }


def attestation_from_payload(payload: dict[str, Any]) -> CollectionAttestation:
    """Parse Qdrant collection metadata payload into an attestation."""
    payload = dict(payload)
    payload.pop("metadata_kind", None)
    payload.pop("type", None)
    return CollectionAttestation.model_validate(payload)


def attestation_from_legacy_collection_meta(
    meta: CollectionMeta,
    *,
    manifest_id: str,
    collection_name: str,
    chunk_count: int = 0,
) -> CollectionAttestation:
    """Build a compact attestation from the current Qdrant metadata model."""
    return CollectionAttestation(
        manifest_id=manifest_id,
        kb_id=meta.kb_name,
        collection_name=collection_name,
        embedding_model=meta.build_config.embedding_model,
        sparse_encoder=meta.build_config.sparse_encoder,
        retrieval_capability=RetrievalCapability(meta.build_config.retrieval_capability),
        chunk_count=chunk_count,
        created_at=datetime.fromisoformat(meta.created_at),
    )


def manifest_from_legacy_collection_meta(
    meta: CollectionMeta,
    *,
    collection_name: str,
    source_snapshot_id: str,
    source_manifest_ref: str | None = None,
    document_count: int = 0,
    chunk_count: int = 0,
    alias: str | None = None,
) -> IndexManifest:
    """Build an artifact manifest from the current collection metadata model."""
    build_config: BuildConfig = meta.build_config
    manifest = IndexManifest(
        kb_id=meta.kb_name,
        collection_name=collection_name,
        alias=alias,
        source_snapshot_id=source_snapshot_id,
        source_manifest_ref=source_manifest_ref,
        document_count=document_count,
        chunk_count=chunk_count,
        embedding_model=build_config.embedding_model,
        sparse_encoder=build_config.sparse_encoder,
        retrieval_capability=RetrievalCapability(build_config.retrieval_capability),
        chunking_config={
            "strategy": build_config.chunking_strategy,
            "chunk_size": build_config.chunk_size,
            "chunk_overlap": build_config.chunk_overlap,
        },
        extraction_config={},
        created_at=datetime.fromisoformat(meta.created_at),
    )
    return with_manifest_id(manifest)
