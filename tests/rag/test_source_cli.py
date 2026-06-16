from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from rag.sources import cli


def _write_catalog(path: Path) -> Path:
    path.write_text(
        dedent(
            """
            schema_version = 2

            [[tasks]]
            id = "code"
            enabled = true
            label = "Code"
            routing_description = "Coding help"
            kb_refs = ["pytorch_reference"]
            adapter = { enabled = false }

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
            manifest = "assets/rag_data/pytorch_reference/sources.toml"
            ingest_adapter = { id = "generic.http_html", version = "1" }

            [[sources]]
            type = "html_docs"
            kb = "pytorch_reference"
            id = "tutorials"
            manifest = "assets/rag_data/pytorch_reference/tutorials.toml"
            ingest_adapter = { id = "generic.http_html", version = "1" }
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return path


class _Model:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self, **_: object) -> dict:
        return self._payload


class _Settings:
    class rag:
        embedding_model = "test-embedding"
        sparse_encoder_model = "Qdrant/bm25"

        class build:
            qdrant_upsert_batch_size = 128

    class platform:
        qdrant_host = "localhost"
        qdrant_port = 6333


def test_cli_build_source_wires_catalog_pair_and_force_flags(capsys) -> None:
    calls: list[dict] = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return _Model({"status": "success"})

    exit_code = cli.main(
        [
            "build-source",
            "--catalog",
            "catalog.toml",
            "--kb",
            "pytorch_reference",
            "--source",
            "docs",
            "--rag-data-root",
            "assets/rag_data",
            "--document-id",
            "html:tensors",
            "--limit",
            "1",
            "--force-fetch",
            "--force-extract",
            "--force-chunk",
            "--chunk-size",
            "128",
            "--chunk-overlap",
            "16",
        ],
        build_catalog_source_fn=fake_build,
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload == {"status": "success"}
    assert calls[0]["kb_id"] == "pytorch_reference"
    assert calls[0]["source_instance_id"] == "docs"
    assert calls[0]["document_ids"] == ["html:tensors"]
    assert calls[0]["limit"] == 1
    assert calls[0]["force_fetch"] is True
    assert calls[0]["force_extract"] is True
    assert calls[0]["force_chunk"] is True
    assert calls[0]["chunking"].chunk_size == 128
    assert calls[0]["chunking"].chunk_overlap == 16


def test_cli_collect_bundle_outputs_bundle_summary(capsys) -> None:
    def fake_collect(**kwargs):
        assert kwargs["kb_id"] == "pytorch_reference"
        assert kwargs["source_instance_id"] == "docs"
        return _Model({"chunk_count": 3})

    exit_code = cli.main(
        [
            "collect-bundle",
            "--catalog",
            "catalog.toml",
            "--kb",
            "pytorch_reference",
            "--source",
            "docs",
            "--rag-data-root",
            "assets/rag_data",
        ],
        collect_source_chunks_fn=fake_collect,
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"chunk_count": 3}


def test_cli_build_source_without_source_builds_all_catalog_sources(capsys) -> None:
    calls: list[dict] = []

    def fake_build_all(**kwargs):
        calls.append(kwargs)
        return _Model({"source_count": 2})

    exit_code = cli.main(
        [
            "build-source",
            "--catalog",
            "catalog.toml",
            "--kb",
            "pytorch_reference",
            "--rag-data-root",
            "assets/rag_data",
        ],
        build_catalog_sources_fn=fake_build_all,
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"source_count": 2}
    assert calls[0]["source_instance_ids"] is None


def test_cli_build_source_can_persist_build_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog_path = tmp_path / "catalog.toml"
    rag_data_root = tmp_path / "rag_data"
    catalog_path.write_text("schema_version = 2\n", encoding="utf-8")

    def fake_build(**kwargs):
        return _Model({"status": "success", "source": kwargs["source_instance_id"]})

    exit_code = cli.main(
        [
            "build-source",
            "--catalog",
            catalog_path.as_posix(),
            "--kb",
            "pytorch_reference",
            "--source",
            "docs",
            "--rag-data-root",
            rag_data_root.as_posix(),
            "--build-run-id",
            "manual-run",
            "--persist-build-run",
        ],
        build_catalog_source_fn=fake_build,
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"status": "success", "source": "docs"}
    build_run_payload = json.loads(
        (
            rag_data_root / "pytorch_reference" / "metadata" / "build_runs" / "manual-run.json"
        ).read_text(encoding="utf-8")
    )
    assert build_run_payload["status"] == "succeeded"
    assert build_run_payload["stage_results"]["build_source"] == {
        "status": "success",
        "source": "docs",
    }


def test_cli_collect_bundle_with_all_uses_catalog_source_set(tmp_path: Path, capsys) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.toml")
    calls: list[dict] = []

    def fake_collect_all(**kwargs):
        calls.append(kwargs)
        return [_Model({"source": source_id}) for source_id in kwargs["source_instance_ids"]]

    exit_code = cli.main(
        [
            "collect-bundle",
            "--catalog",
            catalog_path.as_posix(),
            "--kb",
            "pytorch_reference",
            "--source",
            "all",
            "--rag-data-root",
            "assets/rag_data",
        ],
        collect_source_bundles_fn=fake_collect_all,
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == [
        {"source": "docs"},
        {"source": "tutorials"},
    ]
    assert calls[0]["source_instance_ids"] == ["docs", "tutorials"]


def test_cli_materialize_derives_hybrid_capability_from_catalog(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.toml")
    calls: list[dict] = []

    class _Embedding:
        dimension = 3

    class _Sparse:
        pass

    monkeypatch.setattr(cli, "get_settings", lambda: _Settings())
    monkeypatch.setattr(cli, "EmbeddingService", _Embedding)
    monkeypatch.setattr(cli, "SparseEncoderService", _Sparse)
    monkeypatch.setattr(
        cli,
        "_vector_store",
        lambda collection_name: {"collection": collection_name},
    )

    def fake_collect(**kwargs):
        return {"bundle": kwargs["source_instance_id"]}

    def fake_materialize(**kwargs):
        calls.append(kwargs)
        return _Model({"collection": kwargs["collection_name"]})

    exit_code = cli.main(
        [
            "materialize",
            "--catalog",
            catalog_path.as_posix(),
            "--kb",
            "pytorch_reference",
            "--source",
            "docs",
            "--alias-config",
            "challenger",
            "--collection",
            "rag__pytorch_reference__test",
            "--rag-data-root",
            "assets/rag_data",
        ],
        collect_source_chunks_fn=fake_collect,
        materialize_kb_collection_fn=fake_materialize,
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"collection": "rag__pytorch_reference__test"}
    assert calls[0]["retrieval_capability"] == "hybrid"
    assert calls[0]["target_alias"] is None
    assert calls[0]["sparse_encoder_model"] == "Qdrant/bm25"
    assert isinstance(calls[0]["sparse_encoder_client"], _Sparse)
    assert calls[0]["qdrant_upsert_batch_size"] == 128


def test_cli_materialize_can_persist_build_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.toml")
    rag_data_root = tmp_path / "rag_data"

    class _Embedding:
        dimension = 3

    class _Sparse:
        pass

    monkeypatch.setattr(cli, "get_settings", lambda: _Settings())
    monkeypatch.setattr(cli, "EmbeddingService", _Embedding)
    monkeypatch.setattr(cli, "SparseEncoderService", _Sparse)
    monkeypatch.setattr(
        cli,
        "_vector_store",
        lambda collection_name: {"collection": collection_name},
    )

    def fake_collect(**kwargs):
        return {"bundle": kwargs["source_instance_id"]}

    def fake_materialize(**kwargs):
        return _Model({"collection": kwargs["collection_name"]})

    exit_code = cli.main(
        [
            "materialize",
            "--catalog",
            catalog_path.as_posix(),
            "--kb",
            "pytorch_reference",
            "--source",
            "docs",
            "--alias-config",
            "challenger",
            "--collection",
            "rag__pytorch_reference__test",
            "--rag-data-root",
            rag_data_root.as_posix(),
            "--build-run-id",
            "manual-run",
            "--persist-build-run",
        ],
        collect_source_chunks_fn=fake_collect,
        materialize_kb_collection_fn=fake_materialize,
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"collection": "rag__pytorch_reference__test"}
    build_run_payload = json.loads(
        (
            rag_data_root / "pytorch_reference" / "metadata" / "build_runs" / "manual-run.json"
        ).read_text(encoding="utf-8")
    )
    assert build_run_payload["status"] == "succeeded"
    assert build_run_payload["alias_config"] == "challenger"
    assert build_run_payload["collection_name"] == "rag__pytorch_reference__test"
    assert build_run_payload["stage_results"]["materialize"] == {
        "collection": "rag__pytorch_reference__test"
    }


def test_cli_materialize_all_sources_passes_multiple_bundles(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.toml")
    calls: list[dict] = []

    class _Embedding:
        dimension = 3

    class _Sparse:
        pass

    monkeypatch.setattr(cli, "get_settings", lambda: _Settings())
    monkeypatch.setattr(cli, "EmbeddingService", _Embedding)
    monkeypatch.setattr(cli, "SparseEncoderService", _Sparse)
    monkeypatch.setattr(
        cli,
        "_vector_store",
        lambda collection_name: {"collection": collection_name},
    )

    def fake_collect_all(**kwargs):
        return [{"bundle": source_id} for source_id in kwargs["source_instance_ids"]]

    def fake_materialize(**kwargs):
        calls.append(kwargs)
        return _Model({"bundle_count": len(kwargs["bundles"])})

    exit_code = cli.main(
        [
            "materialize",
            "--catalog",
            catalog_path.as_posix(),
            "--kb",
            "pytorch_reference",
            "--source",
            "all",
            "--alias-config",
            "challenger",
            "--collection",
            "rag__pytorch_reference__test",
            "--rag-data-root",
            "assets/rag_data",
        ],
        collect_source_bundles_fn=fake_collect_all,
        materialize_kb_collection_fn=fake_materialize,
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"bundle_count": 2}
    assert calls[0]["bundles"] == [{"bundle": "docs"}, {"bundle": "tutorials"}]


def test_cli_promote_alias_wires_collection(tmp_path: Path, capsys, monkeypatch) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.toml")

    class _Store:
        def __init__(self, collection_name: str):
            self.collection_name = collection_name

        def read_meta(self) -> dict:
            return {"retrieval_capability": "hybrid"}

    monkeypatch.setattr(
        cli,
        "_vector_store",
        lambda collection_name: _Store(collection_name),
    )

    def fake_promote(**kwargs):
        assert kwargs["kb_id"] == "pytorch_reference"
        assert kwargs["alias"] == "challenger"
        assert kwargs["collection_name"] == "rag__pytorch_reference__test"
        return _Model({"alias_name": "rag__pytorch_reference__challenger"})

    exit_code = cli.main(
        [
            "promote-alias",
            "--catalog",
            catalog_path.as_posix(),
            "--kb",
            "pytorch_reference",
            "--alias",
            "challenger",
            "--collection",
            "rag__pytorch_reference__test",
        ],
        promote_materialized_alias_fn=fake_promote,
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "alias_name": "rag__pytorch_reference__challenger"
    }


def test_cli_promote_alias_rejects_incompatible_collection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    catalog_path = _write_catalog(tmp_path / "catalog.toml")

    class _Store:
        def read_meta(self) -> dict:
            return {"retrieval_capability": "dense"}

    monkeypatch.setattr(cli, "_vector_store", lambda collection_name: _Store())

    with pytest.raises(ValueError, match="retrieval_strategy 'hybrid' is not supported"):
        cli.main(
            [
                "promote-alias",
                "--catalog",
                catalog_path.as_posix(),
                "--kb",
                "pytorch_reference",
                "--alias",
                "challenger",
                "--collection",
                "rag__pytorch_reference__test",
            ],
            promote_materialized_alias_fn=lambda **kwargs: _Model(kwargs),
        )
