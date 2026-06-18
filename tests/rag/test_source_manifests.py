from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from rag.sources import GenericSourceEntry, load_source_manifest


def _write_manifest(path: Path, content: str) -> Path:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return path


def test_source_manifest_loads_adapter_owned_entries(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "sources.toml",
        """
        schema_version = 1

        [[documents]]
        id = "1706.03762"
        title = "Attention Is All You Need"
        """,
    )

    manifest = load_source_manifest(path)

    assert isinstance(manifest.documents[0], GenericSourceEntry)
    assert manifest.documents[0].id == "1706.03762"
    assert manifest.documents[0].uri is None
    assert manifest.documents[0].metadata == {}


def test_source_manifest_normalizes_blank_urls(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "sources.toml",
        """
        schema_version = 1

        [[documents]]
        id = "1706.03762"
        title = "Attention Is All You Need"
        url = ""
        """,
    )

    assert load_source_manifest(path).documents[0].url is None


def test_source_manifest_rejects_duplicate_document_ids(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "sources.toml",
        """
        schema_version = 1

        [[documents]]
        id = "tensors"
        url = "https://pytorch.org/docs/stable/tensors.html"
        title = "Tensors"

        [[documents]]
        id = "tensors"
        url = "https://pytorch.org/docs/stable/torch.html"
        title = "torch"
        """,
    )

    with pytest.raises(ValueError, match="Duplicate source document id 'tensors'"):
        load_source_manifest(path)


def test_source_manifest_rejects_retired_source_type(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "sources.toml",
        """
        schema_version = 1
        source_type = "qasper"

        [[documents]]
        id = "paper-1"
        title = "Paper One"
        """,
    )

    with pytest.raises(ValueError, match="source_type"):
        load_source_manifest(path)
