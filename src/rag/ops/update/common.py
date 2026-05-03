"""Shared metadata-loading helpers for update workflows."""

from __future__ import annotations

from rag.ops.meta import CollectionMeta, read_collection_meta
from rag.vector_store import QdrantVectorStore


def load_update_collection_meta(
    *,
    vector_store: QdrantVectorStore,
    alias_name: str,
    collection_name: str,
    kb_name: str,
) -> CollectionMeta:
    """Load validated `_meta` for an update target with actionable errors."""
    try:
        return read_collection_meta(vector_store, context=collection_name)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Alias '{alias_name}' points to collection '{collection_name}' without valid _meta. "
            f"Rebuild the '{kb_name}' collection for alias '{alias_name}' "
            "before rerunning the update."
        ) from exc
    except ValueError as exc:
        message = str(exc)
        if "retrieval_capability" in message or "sparse_encoder" in message:
            raise RuntimeError(
                f"Alias '{alias_name}' points to legacy collection '{collection_name}' "
                "with incompatible _meta.build_config. "
                "Collections created before the sparse_encoder/retrieval_capability "
                "contract change "
                "cannot be refreshed in place. "
                f"Rebuild the '{kb_name}' collection for alias '{alias_name}' "
                "and rerun the update."
            ) from exc
        raise RuntimeError(
            f"Alias '{alias_name}' points to collection '{collection_name}' "
            f"with invalid _meta: {message}"
        ) from exc
