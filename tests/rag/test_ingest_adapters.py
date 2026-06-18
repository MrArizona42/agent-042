"""Tests for source ingest adapter contracts."""

from __future__ import annotations

import pytest

from rag.contracts import SourceDocument


class _Manifest:
    def __init__(self, source_type: str):
        self.source_type = source_type

    def to_source_documents(self):
        return [
            SourceDocument(
                id="html:tensors",
                source_type="html_docs",
                uri="https://docs.test/tensors.html",
                title="Tensors",
            )
        ]


def test_http_html_adapter_lists_manifest_documents():
    from rag.ingest.adapters import make_http_html_adapter

    adapter = make_http_html_adapter()

    assert adapter.adapter_id == "generic.http_html"
    assert adapter.version == "1"
    assert adapter.source_type == "html_docs"
    assert [document.id for document in adapter.list_documents(_Manifest("html_docs"))] == [
        "html:tensors"
    ]


def test_http_html_adapter_rejects_wrong_manifest_type():
    from rag.ingest.adapters import make_http_html_adapter

    adapter = make_http_html_adapter()

    with pytest.raises(ValueError, match="expects source_type 'html_docs'"):
        adapter.validate_manifest(_Manifest("arxiv_paper"))
