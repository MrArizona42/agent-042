from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag.lifecycle import (
    BuildRequest,
    build_run_path,
    create_build_run,
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
            document_ids=["html:tensors"],
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
    assert calls[0]["document_ids"] == ["html:tensors"]
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
