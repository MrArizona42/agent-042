"""Tests locking in the source-instance-centered artifact layout (Phase 3)."""

from __future__ import annotations

from pathlib import Path

from rag.contracts.manifests import manifest_path
from rag.contracts.metadata import source_document
from rag.lifecycle.commands import build_run_path
from rag.sources.artifacts import extracted_artifact_path
from rag.sources.cache import source_cache_paths
from rag.sources.chunks import chunk_artifact_path


def test_source_cache_paths_are_keyed_by_global_source_instance_id(tmp_path: Path) -> None:
    paths = source_cache_paths(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="pytorch_reference.docs",
        source_document=source_document(
            local_document_id="tensors",
            title="Tensors",
            source_uri="https://docs.test/tensors.html",
            kb_id="pytorch_reference",
            source_instance_id="pytorch_reference.docs",
            adapter_id="generic.http_html",
            adapter_version="1",
            manifest_digest="sha256:manifest",
        ),
        raw_filename="page.html",
    )

    assert paths.raw_path.as_posix().endswith(
        "source_instances/pytorch_reference.docs/raw/pytorch_reference.docs_tensors/page.html"
    )
    assert paths.metadata_path.as_posix().endswith(
        "source_instances/pytorch_reference.docs/metadata/pytorch_reference.docs_tensors.json"
    )


def test_extracted_and_chunk_artifact_paths_are_source_instance_scoped(tmp_path: Path) -> None:
    extracted = extracted_artifact_path(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="pytorch_reference.docs",
        source_document_id="pytorch_reference.docs:tensors",
    )
    chunk = chunk_artifact_path(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="pytorch_reference.docs",
        source_document_id="pytorch_reference.docs:tensors",
    )

    assert extracted.as_posix().endswith(
        "source_instances/pytorch_reference.docs/extracted/pytorch_reference.docs_tensors.json"
    )
    assert chunk.as_posix().endswith(
        "source_instances/pytorch_reference.docs/chunks/pytorch_reference.docs_tensors.json"
    )


def test_manifest_and_build_run_paths_are_knowledge_base_scoped(tmp_path: Path) -> None:
    manifest = manifest_path(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        collection_name="rag__pytorch_reference__20260101_000000",
    )
    run = build_run_path(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        run_id="rag_build_pytorch_reference_20260101_000000",
    )

    assert manifest.as_posix().endswith(
        "knowledge_bases/pytorch_reference/manifests/rag__pytorch_reference__20260101_000000.json"
    )
    assert run.as_posix().endswith(
        "knowledge_bases/pytorch_reference/metadata/build_runs/"
        "rag_build_pytorch_reference_20260101_000000.json"
    )
