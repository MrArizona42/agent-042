"""Project identity metadata carried by LlamaIndex documents and nodes."""

from __future__ import annotations

import uuid
from typing import Any

from llama_index.core import Document
from llama_index.core.schema import TextNode

PROJECT_QDRANT_POINT_NAMESPACE = uuid.UUID("ec97877f-8718-59e6-a940-3bfcf5f88d18")

DOCUMENT_METADATA_KEYS = frozenset(
    {
        "kb_id",
        "source_instance_id",
        "source_document_id",
        "document_id",
        "title",
        "source_uri",
        "adapter_id",
        "adapter_version",
        "manifest_digest",
    }
)

NODE_METADATA_KEYS = frozenset(
    {
        "kb_id",
        "source_instance_id",
        "source_document_id",
        "document_id",
        "chunk_id",
        "title",
        "source_uri",
        "section_title",
        "section_ordinal",
        "section_level",
        "ordinal",
        "token_count",
        "adapter_id",
        "adapter_version",
    }
)


def source_document_id(source_instance_id: str, local_document_id: str) -> str:
    """Return a globally unique, human-readable source document id."""
    return f"{source_instance_id}:{local_document_id}"


def source_document(
    *,
    local_document_id: str,
    title: str,
    source_uri: str,
    kb_id: str,
    source_instance_id: str,
    adapter_id: str,
    adapter_version: str,
    manifest_digest: str,
    metadata: dict[str, Any] | None = None,
) -> Document:
    """Build a source descriptor as a LlamaIndex document."""
    document_id = source_document_id(source_instance_id, local_document_id)
    return Document(
        text="",
        id_=document_id,
        metadata={
            **(metadata or {}),
            "kb_id": kb_id,
            "source_instance_id": source_instance_id,
            "source_document_id": document_id,
            "document_id": document_id,
            "local_document_id": local_document_id,
            "title": title,
            "source_uri": source_uri,
            "adapter_id": adapter_id,
            "adapter_version": adapter_version,
            "manifest_digest": manifest_digest,
        },
    )


def require_metadata(metadata: dict[str, Any], required: frozenset[str], *, kind: str) -> None:
    """Reject objects missing project identity metadata."""
    missing = sorted(key for key in required if key not in metadata)
    if missing:
        raise ValueError(f"{kind} is missing required metadata keys: {missing}")


def require_document_metadata(document: Document) -> None:
    require_metadata(document.metadata, DOCUMENT_METADATA_KEYS, kind="Document")


def node_id_for_chunk(chunk_id: str) -> str:
    """Map a readable chunk id to a stable Qdrant-compatible UUID."""
    return str(uuid.uuid5(PROJECT_QDRANT_POINT_NAMESPACE, chunk_id))


def require_node_metadata(node: TextNode) -> None:
    require_metadata(node.metadata, NODE_METADATA_KEYS, kind="TextNode")
