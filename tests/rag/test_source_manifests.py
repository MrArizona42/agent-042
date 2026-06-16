from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from rag.sources import (
    GenericSourceEntry,
    SourceManifest,
    load_source_manifest,
)


def _write_manifest(path: Path, content: str) -> Path:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return path


def test_arxiv_source_manifest_loads_source_documents(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "sources.toml",
        """
        schema_version = 1
        source_type = "arxiv_paper"

        [[documents]]
        id = "1706.03762"
        title = "Attention Is All You Need"
        """,
    )

    manifest = load_source_manifest(path)
    documents = manifest.to_source_documents()

    assert manifest.source_type == "arxiv_paper"
    assert isinstance(manifest.documents[0], GenericSourceEntry)
    assert documents[0].id == "arxiv_paper:1706.03762"
    assert documents[0].uri == "arxiv_paper:1706.03762"
    assert documents[0].metadata == {}


def test_html_docs_manifest_loads_source_documents(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "sources.toml",
        """
        schema_version = 1
        source_type = "html_docs"

        [[documents]]
        id = "tensors"
        url = "https://pytorch.org/docs/stable/tensors.html"
        title = "Tensors"
        """,
    )

    manifest = load_source_manifest(path)
    documents = manifest.to_source_documents()

    assert manifest.source_type == "html_docs"
    assert isinstance(manifest.documents[0], GenericSourceEntry)
    assert documents[0].id == "html_docs:tensors"
    assert documents[0].uri == "https://pytorch.org/docs/stable/tensors.html"
    assert documents[0].metadata == {}


def test_arxiv_source_manifest_allows_blank_url_and_uses_arxiv_uri(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "sources.toml",
        """
        schema_version = 1
        source_type = "arxiv_paper"

        [[documents]]
        id = "1706.03762"
        title = "Attention Is All You Need"
        url = ""
        """,
    )

    manifest = load_source_manifest(path)
    documents = manifest.to_source_documents()

    assert manifest.documents[0].url is None
    assert documents[0].uri == "arxiv_paper:1706.03762"


def test_source_manifest_rejects_duplicate_document_ids(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "sources.toml",
        """
        schema_version = 1
        source_type = "html_docs"

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


def test_unknown_source_manifest_uses_generic_entries(tmp_path: Path) -> None:
    path = _write_manifest(
        tmp_path / "sources.toml",
        """
        schema_version = 1
        source_type = "qasper"

        [[documents]]
        id = "paper-1"
        title = "Paper One"
        uri = "s3://datasets/qasper/paper-1.json"
        metadata = { split = "train" }
        """,
    )

    manifest = load_source_manifest(path)
    documents = manifest.to_source_documents()

    assert manifest.source_type == "qasper"
    assert isinstance(manifest.documents[0], GenericSourceEntry)
    assert documents[0].id == "qasper:paper-1"
    assert documents[0].source_type == "qasper"
    assert documents[0].uri == "s3://datasets/qasper/paper-1.json"
    assert documents[0].metadata == {"split": "train"}
