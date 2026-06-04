"""Source manifest loading helpers."""

from __future__ import annotations

import tomllib
from pathlib import Path

from rag.sources.models import SourceManifest, source_manifest_from_raw


def load_source_manifest(path: Path | str) -> SourceManifest:
    """Load and validate a source manifest TOML file."""
    path = Path(path)
    if path.suffix.lower() != ".toml":
        raise ValueError(f"Source manifest must be a TOML file (got '{path.name}')")
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    return source_manifest_from_raw(raw)
