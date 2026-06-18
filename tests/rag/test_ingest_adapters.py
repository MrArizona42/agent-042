"""Tests for source adapter contracts."""

from __future__ import annotations

from rag.adapters import SourceAdapterContext
from rag.adapters.sources import make_http_html_adapter
from rag.sources.models import GenericSourceEntry, SourceManifest


def test_http_html_adapter_emits_identity_complete_llamaindex_documents() -> None:
    adapter = make_http_html_adapter()
    manifest = SourceManifest(
        documents=[
            GenericSourceEntry(
                id="tensors",
                title="Tensors",
                url="https://docs.test/tensors.html",
            )
        ]
    )
    documents = adapter.list_documents(
        manifest,
        context=SourceAdapterContext(
            kb_id="pytorch_reference",
            source_instance_id="pytorch_reference.docs",
            manifest_digest="sha256:manifest",
        ),
    )

    assert adapter.adapter_id == "generic.http_html"
    assert documents[0].id_ == "pytorch_reference.docs:tensors"
    assert documents[0].metadata["source_uri"] == "https://docs.test/tensors.html"
    assert documents[0].metadata["adapter_id"] == adapter.adapter_id
    assert documents[0].metadata["manifest_digest"] == "sha256:manifest"
