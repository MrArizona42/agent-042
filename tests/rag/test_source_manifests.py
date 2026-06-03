from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from rag.sources import (
    DEFAULT_SOURCE_CONNECTORS,
    ArxivPaperEntry,
    HtmlDocsEntry,
    SourceConnectorRegistry,
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
    assert isinstance(manifest.documents[0], ArxivPaperEntry)
    assert documents[0].id == "arxiv:1706.03762"
    assert documents[0].uri == "https://arxiv.org/abs/1706.03762"
    assert documents[0].metadata["arxiv_id"] == "1706.03762"


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
    assert isinstance(manifest.documents[0], HtmlDocsEntry)
    assert documents[0].id == "html:tensors"
    assert documents[0].uri == "https://pytorch.org/docs/stable/tensors.html"
    assert documents[0].metadata["page_id"] == "tensors"


def test_arxiv_source_manifest_allows_blank_url_and_derives_uri(tmp_path: Path) -> None:
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
    assert documents[0].uri == "https://arxiv.org/abs/1706.03762"


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


def test_source_manifest_rejects_mixed_document_types() -> None:
    with pytest.raises(ValueError, match="incompatible document entry"):
        SourceManifest(
            source_type="arxiv_paper",
            documents=[
                HtmlDocsEntry(
                    id="tensors",
                    url="https://pytorch.org/docs/stable/tensors.html",
                    title="Tensors",
                )
            ],
        )


def test_default_source_connector_registry_materializes_manifest_documents() -> None:
    manifest = SourceManifest(
        source_type="html_docs",
        documents=[
            HtmlDocsEntry(
                id="tensors",
                url="https://pytorch.org/docs/stable/tensors.html",
                title="Tensors",
            )
        ],
    )

    connector = DEFAULT_SOURCE_CONNECTORS.get("html_docs")
    documents = connector.list_documents(manifest)

    assert documents[0].source_type == "html_docs"


def test_source_connector_registry_rejects_unknown_type() -> None:
    registry = SourceConnectorRegistry()

    with pytest.raises(ValueError, match="Unknown source connector"):
        registry.get("html_docs")
