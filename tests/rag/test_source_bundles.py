from __future__ import annotations

from pathlib import Path

import pytest

from rag.sources.bundles import collect_source_nodes
from rag.sources.chunks import ChunkingConfig, chunk_extracted_artifact
from tests.rag.test_source_chunking import _write_extracted_artifact


def _write_nodes(root: Path, source_document_id: str, text: str) -> None:
    path = _write_extracted_artifact(
        root,
        source_document_id=source_document_id,
        text=text,
    )
    chunk_extracted_artifact(
        path,
        rag_data_root=root,
        config=ChunkingConfig(chunk_size=32, chunk_overlap=4),
    )


def test_collect_source_nodes_returns_materialization_bundle(tmp_path: Path) -> None:
    _write_nodes(tmp_path, "docs:tensors", "Tensor text. More tensor text.")
    _write_nodes(tmp_path, "docs:torch", "Torch text. More torch text.")

    bundle = collect_source_nodes(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="docs",
    )

    assert bundle.document_count == 2
    assert bundle.node_count == len(bundle.nodes)
    assert len(bundle.node_artifact_paths) == 2
    assert all(
        checksum.startswith("sha256:") for checksum in bundle.node_artifact_checksums.values()
    )


def test_collect_source_nodes_filters_document_ids_and_limit(tmp_path: Path) -> None:
    _write_nodes(tmp_path, "docs:tensors", "Tensor text.")
    _write_nodes(tmp_path, "docs:torch", "Torch text.")

    bundle = collect_source_nodes(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="docs",
        document_ids=["docs:tensors", "docs:missing"],
        limit=1,
    )

    assert bundle.document_count == 1
    assert {node.metadata["source_document_id"] for node in bundle.nodes} == {"docs:tensors"}


def test_collect_source_nodes_is_strict_for_corrupt_artifacts(tmp_path: Path) -> None:
    chunk_dir = tmp_path / "source_instances" / "docs" / "chunks"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "broken.json").write_text('{"broken": true}\n', encoding="utf-8")

    with pytest.raises(ValueError):
        collect_source_nodes(
            rag_data_root=tmp_path,
            kb_id="pytorch_reference",
            source_instance_id="docs",
        )
