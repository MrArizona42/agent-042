from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent

import pytest

from rag.lifecycle import (
    BuildRequest,
    build_run_path,
    create_build_run,
    plan_build,
    read_build_run,
    run_alias_promotion_stage,
    run_materialize_stage,
    run_source_build_stage,
)


class _Model:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self, **_: object) -> dict:
        return self._payload


def test_create_build_run_records_request_and_catalog_digest(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.toml"
    catalog_path.write_text("schema_version = 2\n", encoding="utf-8")

    build_run = create_build_run(
        BuildRequest(
            catalog_path=catalog_path.as_posix(),
            kb_id="pytorch_reference",
            source_ids=["docs"],
            rag_data_root=(tmp_path / "rag_data").as_posix(),
            document_ids=[" torch.Tensor ", ""],
            limit=3,
            force_fetch=True,
        ),
        run_id="manual-run",
        created_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )

    assert build_run.run_id == "manual-run"
    assert build_run.status == "planned"
    assert build_run.kb_id == "pytorch_reference"
    assert build_run.source_ids == ["docs"]
    assert build_run.catalog_digest is not None
    assert build_run.catalog_digest.startswith("sha256:")
    assert build_run.build_profile_digest is not None
    assert build_run.started_at == datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def test_create_build_run_records_source_manifest_and_adapter_attestation(
    tmp_path: Path,
) -> None:
    docs_manifest = tmp_path / "docs.sources.toml"
    tutorials_manifest = tmp_path / "tutorials.sources.toml"
    docs_manifest.write_text('[[documents]]\nid = "docs:intro"\n', encoding="utf-8")
    tutorials_manifest.write_text('[[documents]]\nid = "tutorials:intro"\n', encoding="utf-8")
    catalog_path = tmp_path / "catalog.toml"
    catalog_path.write_text(
        dedent(
            """
            schema_version = 2

            [[knowledge_bases]]
            id = "pytorch_reference"
            enabled = true
            label = "PyTorch reference"
            description = "PyTorch docs"
            selection_description = "PyTorch docs"
            update_strategy = "replace"
            default_alias = "challenger"
            aliases.challenger.top_k = 5
            aliases.challenger.score_threshold = 0.01
            aliases.challenger.retrieval_strategy = "hybrid"
            aliases.challenger.reranker = "reranker"
            aliases.challenger.reranker_multiplier = 4

            [[sources]]
            type = "html_docs"
            kb = "pytorch_reference"
            id = "docs"
            manifest = "docs.sources.toml"
            ingest_adapter = { id = "generic.http_html", version = "1" }

            [[sources]]
            type = "html_docs"
            kb = "pytorch_reference"
            id = "tutorials"
            manifest = "tutorials.sources.toml"
            ingest_adapter = { id = "generic.http_html", version = "2" }
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    build_run = create_build_run(
        BuildRequest(
            catalog_path=catalog_path.as_posix(),
            kb_id="pytorch_reference",
            source_ids=["docs"],
            rag_data_root=(tmp_path / "rag_data").as_posix(),
        ),
        run_id="manual-run",
    )

    assert set(build_run.manifest_digests) == {"docs"}
    assert build_run.manifest_digests["docs"].startswith("sha256:")
    assert build_run.adapter_versions == {"docs": "generic.http_html@1"}


def test_plan_build_validates_source_manifest_with_adapter(tmp_path: Path) -> None:
    manifest_path = tmp_path / "docs.sources.toml"
    manifest_path.write_text(
        dedent(
            """
            source_type = "html_docs"

            [[documents]]
            id = "intro"
            title = "Introduction"
            url = "https://example.test/intro"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    catalog_path = tmp_path / "catalog.toml"
    catalog_path.write_text(
        dedent(
            """
            schema_version = 2

            [[knowledge_bases]]
            id = "pytorch_reference"
            enabled = true
            label = "PyTorch reference"
            description = "PyTorch docs"
            selection_description = "PyTorch docs"
            update_strategy = "replace"
            default_alias = "challenger"
            aliases.challenger.top_k = 5
            aliases.challenger.score_threshold = 0.01
            aliases.challenger.retrieval_strategy = "hybrid"
            aliases.challenger.reranker = "reranker"
            aliases.challenger.reranker_multiplier = 4

            [[sources]]
            type = "html_docs"
            kb = "pytorch_reference"
            id = "docs"
            manifest = "docs.sources.toml"
            ingest_adapter = { id = "generic.http_html", version = "1" }
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    result = plan_build(
        BuildRequest(
            catalog_path=catalog_path.as_posix(),
            kb_id="pytorch_reference",
            rag_data_root=(tmp_path / "rag_data").as_posix(),
        )
    )

    assert result.valid is True
    assert result.sources[0].manifest_reachable is True
    assert result.sources[0].manifest_valid is True


def test_plan_build_rejects_manifest_that_adapter_would_reject(tmp_path: Path) -> None:
    manifest_path = tmp_path / "docs.sources.toml"
    manifest_path.write_text(
        dedent(
            """
            source_type = "arxiv_paper"

            [[documents]]
            id = "intro"
            title = "Introduction"
            url = "https://example.test/intro"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    catalog_path = tmp_path / "catalog.toml"
    catalog_path.write_text(
        dedent(
            """
            schema_version = 2

            [[knowledge_bases]]
            id = "pytorch_reference"
            enabled = true
            label = "PyTorch reference"
            description = "PyTorch docs"
            selection_description = "PyTorch docs"
            update_strategy = "replace"
            default_alias = "challenger"
            aliases.challenger.top_k = 5
            aliases.challenger.score_threshold = 0.01
            aliases.challenger.retrieval_strategy = "hybrid"
            aliases.challenger.reranker = "reranker"
            aliases.challenger.reranker_multiplier = 4

            [[sources]]
            type = "html_docs"
            kb = "pytorch_reference"
            id = "docs"
            manifest = "docs.sources.toml"
            ingest_adapter = { id = "generic.http_html", version = "1" }
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    result = plan_build(
        BuildRequest(
            catalog_path=catalog_path.as_posix(),
            kb_id="pytorch_reference",
            rag_data_root=(tmp_path / "rag_data").as_posix(),
        )
    )

    assert result.valid is False
    assert result.sources[0].manifest_valid is False
    assert (
        "Source manifest invalid for adapter 'generic.http_html@1'" in result.sources[0].errors[0]
    )


def test_run_source_build_stage_persists_successful_build_run(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.toml"
    rag_data_root = tmp_path / "rag_data"
    catalog_path.write_text("schema_version = 2\n", encoding="utf-8")
    calls: list[dict] = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return _Model({"status": "success", "source": kwargs["source_instance_id"]})

    result = run_source_build_stage(
        BuildRequest(
            catalog_path=catalog_path.as_posix(),
            kb_id="pytorch_reference",
            source_ids=["docs"],
            rag_data_root=rag_data_root.as_posix(),
            document_ids=["html_docs:tensors"],
            limit=1,
            force_chunk=True,
        ),
        run_id="run-1",
        build_catalog_source_fn=fake_build,
    )

    path = build_run_path(
        rag_data_root=rag_data_root,
        kb_id="pytorch_reference",
        run_id="run-1",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert result.result.model_dump() == {"status": "success", "source": "docs"}
    assert payload["status"] == "succeeded"
    assert payload["current_stage"] == "build_source"
    assert payload["stage_results"]["build_source"] == {"status": "success", "source": "docs"}
    assert calls[0]["source_instance_id"] == "docs"
    assert calls[0]["document_ids"] == ["html_docs:tensors"]
    assert calls[0]["force_chunk"] is True


def test_run_source_build_stage_persists_failed_build_run(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.toml"
    rag_data_root = tmp_path / "rag_data"
    catalog_path.write_text("schema_version = 2\n", encoding="utf-8")

    def fake_build(**_: object):
        raise ValueError("manifest disappeared")

    with pytest.raises(ValueError, match="manifest disappeared"):
        run_source_build_stage(
            BuildRequest(
                catalog_path=catalog_path.as_posix(),
                kb_id="pytorch_reference",
                source_ids=["docs"],
                rag_data_root=rag_data_root.as_posix(),
            ),
            run_id="run-2",
            build_catalog_source_fn=fake_build,
        )

    payload = json.loads(
        build_run_path(
            rag_data_root=rag_data_root,
            kb_id="pytorch_reference",
            run_id="run-2",
        ).read_text(encoding="utf-8")
    )

    assert payload["status"] == "failed"
    assert payload["errors"] == ["manifest disappeared"]


def test_run_source_build_stage_dry_run_does_not_call_stage(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.toml"
    rag_data_root = tmp_path / "rag_data"
    catalog_path.write_text("schema_version = 2\n", encoding="utf-8")

    def fake_build(**_: object):
        raise AssertionError("dry run should not execute build")

    result = run_source_build_stage(
        BuildRequest(
            catalog_path=catalog_path.as_posix(),
            kb_id="pytorch_reference",
            source_ids=["docs"],
            rag_data_root=rag_data_root.as_posix(),
            dry_run=True,
        ),
        run_id="dry-run-1",
        build_catalog_source_fn=fake_build,
    )

    payload = json.loads(
        build_run_path(
            rag_data_root=rag_data_root,
            kb_id="pytorch_reference",
            run_id="dry-run-1",
        ).read_text(encoding="utf-8")
    )

    assert result.result["dry_run"] is True
    assert result.result["stage"] == "build_source"
    assert payload["status"] == "planned"
    assert payload["stage_results"]["build_source"]["dry_run"] is True


def test_run_source_build_stage_uses_multi_source_function(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.toml"
    catalog_path.write_text("schema_version = 2\n", encoding="utf-8")
    calls: list[dict] = []

    def fake_build_all(**kwargs):
        calls.append(kwargs)
        return _Model({"source_count": len(kwargs["source_instance_ids"])})

    result = run_source_build_stage(
        BuildRequest(
            catalog_path=catalog_path.as_posix(),
            kb_id="pytorch_reference",
            source_ids=["docs", "tutorials"],
            rag_data_root=(tmp_path / "rag_data").as_posix(),
        ),
        run_id="run-3",
        build_catalog_sources_fn=fake_build_all,
        persist=False,
    )

    assert result.build_run.status == "succeeded"
    assert result.build_run.stage_results["build_source"] == {"source_count": 2}
    assert calls[0]["source_instance_ids"] == ["docs", "tutorials"]


def test_materialize_and_promote_append_existing_build_run(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.toml"
    rag_data_root = tmp_path / "rag_data"
    catalog_path.write_text("schema_version = 2\n", encoding="utf-8")
    request = BuildRequest(
        catalog_path=catalog_path.as_posix(),
        kb_id="pytorch_reference",
        source_ids=["docs"],
        rag_data_root=rag_data_root.as_posix(),
    )

    run_source_build_stage(
        request,
        run_id="run-4",
        build_catalog_source_fn=lambda **kwargs: _Model({"source": kwargs["source_instance_id"]}),
    )
    run_materialize_stage(
        BuildRequest(
            catalog_path=catalog_path.as_posix(),
            kb_id="pytorch_reference",
            source_ids=["docs"],
            rag_data_root=rag_data_root.as_posix(),
            alias_config="challenger",
            collection_name="rag__pytorch_reference__test",
        ),
        run_id="run-4",
        stage_fn=lambda: _Model({"collection": "rag__pytorch_reference__test"}),
    )
    result = run_alias_promotion_stage(
        BuildRequest(
            catalog_path=catalog_path.as_posix(),
            kb_id="pytorch_reference",
            rag_data_root=rag_data_root.as_posix(),
            alias_config="challenger",
            collection_name="rag__pytorch_reference__test",
        ),
        run_id="run-4",
        stage_fn=lambda: _Model({"alias": "rag__pytorch_reference__challenger"}),
    )

    payload = read_build_run(
        rag_data_root=rag_data_root,
        kb_id="pytorch_reference",
        run_id="run-4",
    )

    assert result.build_run.status == "promoted"
    assert payload.status == "promoted"
    assert payload.current_stage == "promote_alias"
    assert payload.alias_config == "challenger"
    assert payload.collection_name == "rag__pytorch_reference__test"
    assert payload.stage_results == {
        "build_source": {"source": "docs"},
        "materialize": {"collection": "rag__pytorch_reference__test"},
        "promote_alias": {"alias": "rag__pytorch_reference__challenger"},
    }
