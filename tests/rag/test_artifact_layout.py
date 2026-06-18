"""Tests locking in the source-instance-centered artifact layout (Phase 3)."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from rag.contracts import SourceDocument
from rag.contracts.manifests import manifest_path
from rag.lifecycle.commands import build_run_path
from rag.sources.artifacts import extracted_artifact_path
from rag.sources.cache import source_cache_paths
from rag.sources.chunks import chunk_artifact_path


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return path


def test_source_cache_paths_are_keyed_by_global_source_instance_id(tmp_path: Path) -> None:
    paths = source_cache_paths(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="pytorch_reference.docs",
        source_document=SourceDocument(
            id="html_docs:tensors",
            source_type="html_docs",
            uri="https://docs.test/tensors.html",
            title="Tensors",
        ),
        raw_filename="page.html",
    )

    assert paths.raw_path.as_posix().endswith(
        "source_instances/pytorch_reference.docs/raw/html_docs_tensors/page.html"
    )
    assert paths.metadata_path.as_posix().endswith(
        "source_instances/pytorch_reference.docs/metadata/html_docs_tensors.json"
    )


def test_extracted_and_chunk_artifact_paths_are_source_instance_scoped(tmp_path: Path) -> None:
    extracted = extracted_artifact_path(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="pytorch_reference.docs",
        source_document_id="html_docs:tensors",
    )
    chunk = chunk_artifact_path(
        rag_data_root=tmp_path,
        kb_id="pytorch_reference",
        source_instance_id="pytorch_reference.docs",
        source_document_id="html_docs:tensors",
    )

    assert extracted.as_posix().endswith(
        "source_instances/pytorch_reference.docs/extracted/html_docs_tensors.json"
    )
    assert chunk.as_posix().endswith(
        "source_instances/pytorch_reference.docs/chunks/html_docs_tensors.json"
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
        "knowledge_bases/pytorch_reference/manifests/"
        "rag__pytorch_reference__20260101_000000.json"
    )
    assert run.as_posix().endswith(
        "knowledge_bases/pytorch_reference/metadata/build_runs/"
        "rag_build_pytorch_reference_20260101_000000.json"
    )


def test_migrate_legacy_source_manifest_copies_to_conventional_path(tmp_path: Path) -> None:
    from app_config.catalog.schema import SourceConfig, SourceIngestAdapterConfig
    from app_config.catalog.source_instances import migrate_legacy_source_manifest

    catalog_path = tmp_path / "catalog.toml"
    catalog_path.write_text("schema_version = 2\n", encoding="utf-8")
    legacy_manifest = _write(
        tmp_path / "pytorch_reference" / "sources.toml",
        """
        schema_version = 1
        source_type = "html_docs"

        [[documents]]
        id = "tensors"
        title = "Tensors"
        url = "https://docs.test/tensors.html"
        """,
    )
    source = SourceConfig(
        type="html_docs",
        kb="pytorch_reference",
        id="docs",
        manifest="pytorch_reference/sources.toml",
        ingest_adapter=SourceIngestAdapterConfig(id="generic.http_html", version="1"),
    )

    destination = migrate_legacy_source_manifest(
        source=source,
        catalog_path=catalog_path,
        rag_data_root=tmp_path / "rag_data",
    )

    assert destination.as_posix().endswith(
        "rag_data/source_instances/pytorch_reference.docs/manifest.toml"
    )
    assert destination.read_text(encoding="utf-8") == legacy_manifest.read_text(encoding="utf-8")

    # Re-running without force returns the existing destination unchanged.
    again = migrate_legacy_source_manifest(
        source=source,
        catalog_path=catalog_path,
        rag_data_root=tmp_path / "rag_data",
    )
    assert again == destination
