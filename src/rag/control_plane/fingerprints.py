"""Canonical digest and deterministic identity helpers for the RAG control plane.

All digests are SHA-256 over canonical JSON: sorted keys, compact separators,
UTF-8 encoding, prefixed `sha256:`. Two semantically equal inputs always
produce the same digest regardless of source formatting or key order.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from pydantic import BaseModel

from app_config.catalog.schema import AliasBuildConfig, AliasChunkingConfig, AliasRetrievalConfig

TRANSFORMATION_CONTRACT_VERSION = "1"


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def canonical_digest(payload: BaseModel | object) -> str:
    """Deterministic, formatting-independent SHA-256 digest of canonical JSON."""
    data = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    return "sha256:" + hashlib.sha256(_canonical_bytes(data)).hexdigest()


def build_config_digest(build: AliasBuildConfig) -> str:
    """Digest of index-time configuration only; excludes retrieval and runtime fields."""
    return canonical_digest(build)


def retrieval_config_digest(retrieve: AliasRetrievalConfig) -> str:
    """Digest of query-time configuration only."""
    return canonical_digest(retrieve)


def catalog_digest(build: AliasBuildConfig, retrieve: AliasRetrievalConfig) -> str:
    """Digest of the complete desired alias declaration (build and retrieve combined)."""
    return canonical_digest(
        {"build": build.model_dump(mode="json"), "retrieve": retrieve.model_dump(mode="json")}
    )


def source_declaration_digest(sources: Iterable[tuple[str, str, str, str]]) -> str:
    """Digest of ordered (source_instance_id, manifest_digest, adapter_id, adapter_version).

    Computable without fetching remote content.
    """
    ordered = sorted(sources, key=lambda item: item[0])
    return canonical_digest([list(item) for item in ordered])


def transformation_digest(
    chunking: AliasChunkingConfig,
    contract_version: str = TRANSFORMATION_CONTRACT_VERSION,
) -> str:
    """Digest scoping node/chunk artifacts: chunking plus the extraction contract version."""
    return canonical_digest(
        {"chunking": chunking.model_dump(mode="json"), "contract_version": contract_version}
    )


def source_snapshot_id(nodes: Iterable[tuple[str, str]]) -> str:
    """Digest of ordered (source_instance_id, node_artifact_checksum) pairs."""
    ordered = sorted(nodes, key=lambda item: item[0])
    return canonical_digest([list(item) for item in ordered])


def release_fingerprint(
    *,
    kb_id: str,
    build_config_digest: str,
    source_declaration_digest: str,
    source_snapshot_id: str,
) -> str:
    """Full content identity of a release. A ready release with this fingerprint is reused."""
    return canonical_digest(
        {
            "kb_id": kb_id,
            "build_config_digest": build_config_digest,
            "source_declaration_digest": source_declaration_digest,
            "source_snapshot_id": source_snapshot_id,
        }
    )


def _sanitize_kb_id(kb_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in kb_id)


def _fingerprint_hex(fingerprint: str) -> str:
    return fingerprint.removeprefix("sha256:")


def release_id(kb_id: str, fingerprint: str) -> str:
    return f"ragrel_{_sanitize_kb_id(kb_id)}_{_fingerprint_hex(fingerprint)[:16]}"


def collection_name(kb_id: str, fingerprint: str) -> str:
    return f"rag__{_sanitize_kb_id(kb_id)}__{_fingerprint_hex(fingerprint)[:16]}"
