"""Source fetch cache layout and immutable write helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from rag.contracts import SourceDocument


class SourceCachePaths(BaseModel):
    """Cache paths for one source document fetch."""

    model_config = ConfigDict(extra="forbid")

    root_dir: Path
    raw_path: Path
    metadata_path: Path


def safe_document_id(source_document_id: str) -> str:
    """Return a path-safe id while keeping readable source identity."""
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", source_document_id).strip("_")
    return safe_id or hashlib.sha256(source_document_id.encode("utf-8")).hexdigest()


def source_cache_paths(
    *,
    rag_data_root: Path | str,
    kb_id: str,
    source_instance_id: str,
    source_document: SourceDocument,
    raw_filename: str,
) -> SourceCachePaths:
    """Return conventional cache paths for a source document.

    Cache paths are keyed by the globally unique source instance id, not by
    `kb_id`; `kb_id` is accepted for caller symmetry with sibling functions.
    """
    document_dir_name = safe_document_id(source_document.id)
    root_dir = Path(rag_data_root) / "source_instances" / source_instance_id
    return SourceCachePaths(
        root_dir=root_dir,
        raw_path=root_dir / "raw" / document_dir_name / raw_filename,
        metadata_path=root_dir / "metadata" / f"{document_dir_name}.json",
    )


def write_bytes_immutable(path: Path, content: bytes, *, force: bool = False) -> None:
    """Write bytes once unless *force* is set."""
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def write_json_immutable(path: Path, payload: dict[str, Any], *, force: bool = False) -> None:
    """Write JSON once unless *force* is set."""
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def sha256_bytes(content: bytes) -> str:
    """Return a sha256 digest label for bytes."""
    return f"sha256:{hashlib.sha256(content).hexdigest()}"
