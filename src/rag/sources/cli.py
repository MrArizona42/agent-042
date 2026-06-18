"""CLI entry points for source-build lifecycle commands."""

from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from app_config.catalog import (
    CatalogConfig,
    build_source_instance_index,
    load_catalog,
    materialize_catalog,
    resolve_corpus_source_instance_ids,
)
from app_config.runtime import get_settings
from rag.embeddings import EmbeddingService
from rag.indexing.materialize import (
    collection_name_for_build,
    materialize_kb_collection,
    promote_materialized_alias,
    retrieval_capability_for_strategy,
    validate_strategy_supported,
)
from rag.lifecycle import (
    BuildRequest,
    list_build_runs,
    load_or_create_build_run,
    plan_build,
    read_build_run,
    run_alias_promotion_stage,
    run_materialize_stage,
    run_source_build_stage,
)
from rag.sources.benchmark_prep import prepare_benchmark_source_instance
from rag.sources.build import build_catalog_sources
from rag.sources.bundles import collect_source_bundles, collect_source_nodes
from rag.sources.chunks import ChunkingConfig
from rag.sparse_encoder import SparseEncoderService
from rag.vector_store import QdrantVectorStore


def _json_default(value: object) -> str:
    return str(value)


def _json_payload(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, list):
        return [_json_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_json_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_payload(item) for key, item in value.items()}
    return value


def _print_model(model: Any) -> None:
    payload = _json_payload(model)
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _document_ids(values: list[str] | None) -> list[str] | None:
    return values or None


def _source_instance_ids(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    source_instance_ids: list[str] = []
    for value in values:
        source_instance_ids.extend(item.strip() for item in value.split(",") if item.strip())
    if not source_instance_ids or any(
        source_instance_id.lower() == "all" for source_instance_id in source_instance_ids
    ):
        return None
    return source_instance_ids


def _load_catalog_config(catalog_path: Path | str) -> CatalogConfig:
    path = Path(catalog_path)
    catalog = CatalogConfig(**tomllib.loads(path.read_text(encoding="utf-8")))
    materialize_catalog(catalog)
    return catalog


def _catalog_source_ids(
    *,
    catalog_path: Path | str,
    kb_id: str,
    source_ids: list[str] | None,
) -> list[str]:
    catalog = _load_catalog_config(catalog_path)
    return resolve_corpus_source_instance_ids(
        catalog,
        kb_id=kb_id,
        source_ids=source_ids,
    )


def _kb_for_source_instances(
    *,
    catalog_path: Path | str,
    source_instance_ids: list[str],
) -> str:
    catalog = _load_catalog_config(catalog_path)
    index = build_source_instance_index(catalog)
    kb_ids: set[str] = set()
    for source_instance_id in source_instance_ids:
        instance = index.get(source_instance_id)
        if instance.role != "corpus":
            raise ValueError(
                f"Source instance '{source_instance_id}' has role '{instance.role}'; "
                "build-source only accepts role 'corpus' instances."
            )
        kb_ids.add(instance.knowledge_base)
    if len(kb_ids) != 1:
        raise ValueError(
            "build-source requires source instances from exactly one KB; "
            f"got {sorted(kb_ids)}"
        )
    return next(iter(kb_ids))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag.sources.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_source_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--catalog", required=True)
        command.add_argument("--kb", required=True)
        command.add_argument(
            "--source-instance",
            action="append",
            dest="source_instance_ids",
            help=(
                "Global source instance id. Repeat for subsets, use 'all', "
                "or omit for all corpus sources in the KB."
            ),
        )
        command.add_argument("--rag-data-root", required=True)
        command.add_argument("--document-id", action="append", dest="document_ids")
        command.add_argument("--limit", type=int)

    def add_build_run_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--build-run-id")
        command.add_argument("--persist-build-run", action="store_true")
        command.add_argument("--dry-run", action="store_true")

    build_source = subparsers.add_parser("build-source")
    build_source.add_argument("--catalog", required=True)
    build_source.add_argument(
        "--source-instance",
        action="append",
        dest="source_instance_ids",
        required=True,
        help="Global source instance id (e.g. 'kb_id.local_id'). Repeat for multiple.",
    )
    build_source.add_argument("--rag-data-root", required=True)
    build_source.add_argument("--document-id", action="append", dest="document_ids")
    build_source.add_argument("--limit", type=int)
    build_source.add_argument("--force-fetch", action="store_true")
    build_source.add_argument("--force-extract", action="store_true")
    build_source.add_argument("--force-chunk", action="store_true")
    build_source.add_argument("--chunk-size", type=int)
    build_source.add_argument("--chunk-overlap", type=int)
    add_build_run_args(build_source)

    collect_bundle = subparsers.add_parser("collect-bundle")
    add_common_source_args(collect_bundle)

    materialize = subparsers.add_parser("materialize")
    add_common_source_args(materialize)
    materialize.add_argument("--alias-config", required=True)
    materialize.add_argument("--collection")
    materialize.add_argument("--force-recreate", action="store_true")
    add_build_run_args(materialize)

    promote = subparsers.add_parser("promote-alias")
    promote.add_argument(
        "--catalog",
        help="Catalog TOML path. Defaults to CONFIG__CATALOG_PATH when omitted.",
    )
    promote.add_argument("--kb", required=True)
    promote.add_argument("--alias", required=True)
    promote.add_argument("--collection", required=True)
    promote.add_argument("--rag-data-root")
    add_build_run_args(promote)

    plan = subparsers.add_parser(
        "plan", help="Validate catalog/sources/adapters without executing."
    )
    plan.add_argument("--catalog", required=True)
    plan.add_argument("--kb", required=True)
    plan.add_argument(
        "--source-instance",
        action="append",
        dest="source_instance_ids",
        help="Global source instance id. Repeat for subsets, use 'all', or omit for all.",
    )
    plan.add_argument("--rag-data-root", default=".")

    prepare_benchmark = subparsers.add_parser(
        "prepare-benchmark", help="Validate a benchmark manifest and emit cases/labels."
    )
    prepare_benchmark.add_argument("--catalog", required=True)
    prepare_benchmark.add_argument("--source-instance", required=True, dest="source_instance_id")
    prepare_benchmark.add_argument("--rag-data-root", required=True)

    status = subparsers.add_parser("status", help="List persisted build runs for a KB.")
    status.add_argument("--kb", required=True)
    status.add_argument("--rag-data-root", required=True)

    show_run = subparsers.add_parser("show-build-run", help="Print a persisted BuildRun artifact.")
    show_run.add_argument("--kb", required=True)
    show_run.add_argument("--rag-data-root", required=True)
    show_run.add_argument("--build-run-id", required=True)

    return parser


def _chunking_from_args(args: argparse.Namespace) -> ChunkingConfig | None:
    if args.chunk_size is None and args.chunk_overlap is None:
        return None
    return ChunkingConfig(
        chunk_size=args.chunk_size or 512,
        chunk_overlap=args.chunk_overlap if args.chunk_overlap is not None else 64,
    )


def _alias_strategy(*, catalog_path: Path | str, kb_id: str, alias: str) -> str:
    _, kb_index = load_catalog(catalog_path)
    kb = kb_index.get(kb_id)
    if kb is None:
        raise ValueError(f"Unknown KB '{kb_id}'")
    alias_cfg = kb.aliases.get(alias)
    if alias_cfg is None:
        raise ValueError(f"Unknown alias config '{alias}' for KB '{kb_id}'")
    return alias_cfg.retrieval_strategy


def _catalog_path_from_args(args: argparse.Namespace) -> Path | str:
    return args.catalog or get_settings().catalog.path


def _vector_store(*, collection_name: str) -> QdrantVectorStore:
    settings = get_settings()
    return QdrantVectorStore(
        host=settings.platform.qdrant_host,
        port=settings.platform.qdrant_port,
        collection_name=collection_name,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    build_catalog_sources_fn: Callable[..., Any] = build_catalog_sources,
    collect_source_nodes_fn: Callable[..., Any] = collect_source_nodes,
    collect_source_bundles_fn: Callable[..., Any] = collect_source_bundles,
    materialize_kb_collection_fn: Callable[..., Any] = materialize_kb_collection,
    promote_materialized_alias_fn: Callable[..., Any] = promote_materialized_alias,
    prepare_benchmark_source_instance_fn: Callable[..., Any] = prepare_benchmark_source_instance,
) -> int:
    """Run the source lifecycle CLI."""
    args = _parser().parse_args(argv)

    if args.command == "prepare-benchmark":
        result = prepare_benchmark_source_instance_fn(
            catalog_path=args.catalog,
            source_instance_id=args.source_instance_id,
            rag_data_root=args.rag_data_root,
        )
        _print_model(result)
        return 0

    if args.command == "build-source":
        kb_id = _kb_for_source_instances(
            catalog_path=args.catalog,
            source_instance_ids=args.source_instance_ids,
        )
        stage_result = run_source_build_stage(
            BuildRequest(
                catalog_path=args.catalog,
                kb_id=kb_id,
                source_ids=args.source_instance_ids,
                rag_data_root=args.rag_data_root,
                document_ids=_document_ids(args.document_ids),
                limit=args.limit,
                force_fetch=args.force_fetch,
                force_extract=args.force_extract,
                force_chunk=args.force_chunk,
                dry_run=args.dry_run,
            ),
            run_id=args.build_run_id,
            build_catalog_sources_fn=build_catalog_sources_fn,
            persist=args.persist_build_run,
            chunking=_chunking_from_args(args),
        )
        _print_model(stage_result.result)
        return 0

    if args.command == "collect-bundle":
        source_ids = _source_instance_ids(args.source_instance_ids)
        resolved_source_ids = _catalog_source_ids(
            catalog_path=args.catalog,
            kb_id=args.kb,
            source_ids=source_ids,
        )
        if len(resolved_source_ids) == 1:
            result = collect_source_nodes_fn(
                rag_data_root=args.rag_data_root,
                kb_id=args.kb,
                source_instance_id=resolved_source_ids[0],
                document_ids=_document_ids(args.document_ids),
                limit=args.limit,
            )
        else:
            result = collect_source_bundles_fn(
                rag_data_root=args.rag_data_root,
                kb_id=args.kb,
                source_instance_ids=resolved_source_ids,
                document_ids=_document_ids(args.document_ids),
                limit=args.limit,
            )
        _print_model(result)
        return 0

    if args.command == "materialize":
        collection_name = args.collection or collection_name_for_build(
            kb_id=args.kb,
        )
        source_ids = _source_instance_ids(args.source_instance_ids)
        request = BuildRequest(
            catalog_path=args.catalog,
            kb_id=args.kb,
            source_ids=source_ids,
            rag_data_root=args.rag_data_root,
            alias_config=args.alias_config,
            collection_name=collection_name,
            document_ids=_document_ids(args.document_ids),
            limit=args.limit,
            force_recreate=args.force_recreate,
            dry_run=args.dry_run,
        )

        build_run_for_materialize = None
        if args.build_run_id and args.rag_data_root:
            build_run_for_materialize = load_or_create_build_run(request, run_id=args.build_run_id)

        def _materialize_stage() -> Any:
            settings = get_settings()
            strategy = _alias_strategy(
                catalog_path=args.catalog,
                kb_id=args.kb,
                alias=args.alias_config,
            )
            capability = retrieval_capability_for_strategy(strategy)  # type: ignore[arg-type]
            resolved_source_ids = _catalog_source_ids(
                catalog_path=args.catalog,
                kb_id=args.kb,
                source_ids=source_ids,
            )
            if len(resolved_source_ids) == 1:
                bundles = [
                    collect_source_nodes_fn(
                        rag_data_root=args.rag_data_root,
                        kb_id=args.kb,
                        source_instance_id=resolved_source_ids[0],
                        document_ids=_document_ids(args.document_ids),
                        limit=args.limit,
                    )
                ]
            else:
                bundles = collect_source_bundles_fn(
                    rag_data_root=args.rag_data_root,
                    kb_id=args.kb,
                    source_instance_ids=resolved_source_ids,
                    document_ids=_document_ids(args.document_ids),
                    limit=args.limit,
                )
            return materialize_kb_collection_fn(
                kb_id=args.kb,
                collection_name=collection_name,
                bundles=bundles,
                vector_store=_vector_store(collection_name=collection_name),
                embedding_client=EmbeddingService(),
                embedding_model=settings.rag.embedding_model,
                retrieval_capability=capability,
                rag_data_root=args.rag_data_root,
                target_alias=None,
                sparse_encoder_model=(
                    settings.rag.sparse_encoder_model if capability == "hybrid" else None
                ),
                sparse_encoder_client=SparseEncoderService() if capability == "hybrid" else None,
                qdrant_upsert_batch_size=settings.rag.build.qdrant_upsert_batch_size,
                force_recreate=args.force_recreate,
                build_config_ref=args.catalog,
                source_adapter_versions=(
                    build_run_for_materialize.adapter_versions
                    if build_run_for_materialize is not None
                    else None
                ),
                source_manifest_digests=(
                    build_run_for_materialize.manifest_digests
                    if build_run_for_materialize is not None
                    else None
                ),
                build_config_digest=(
                    build_run_for_materialize.catalog_digest
                    if build_run_for_materialize is not None
                    else None
                ),
                build_profile_digest=(
                    build_run_for_materialize.build_profile_digest
                    if build_run_for_materialize is not None
                    else None
                ),
            )

        stage_result = run_materialize_stage(
            request,
            stage_fn=_materialize_stage,
            run_id=args.build_run_id,
            persist=args.persist_build_run,
        )
        _print_model(stage_result.result)
        return 0

    if args.command == "promote-alias":
        catalog_path = _catalog_path_from_args(args)
        if (args.persist_build_run or args.dry_run) and not args.rag_data_root:
            raise ValueError("--rag-data-root is required for --persist-build-run or --dry-run")

        def _promote_stage() -> Any:
            strategy = _alias_strategy(catalog_path=catalog_path, kb_id=args.kb, alias=args.alias)
            vector_store = _vector_store(collection_name=args.collection)
            payload = vector_store.read_meta()
            if payload is None:
                raise RuntimeError(f"Collection '{args.collection}' has no attestation metadata")
            validate_strategy_supported(
                retrieval_strategy=strategy,  # type: ignore[arg-type]
                retrieval_capability=payload.get("retrieval_capability"),  # type: ignore[arg-type]
            )
            return promote_materialized_alias_fn(
                kb_id=args.kb,
                alias=args.alias,
                collection_name=args.collection,
                vector_store=vector_store,
            )

        if args.rag_data_root:
            stage_result = run_alias_promotion_stage(
                BuildRequest(
                    catalog_path=str(catalog_path),
                    kb_id=args.kb,
                    rag_data_root=args.rag_data_root,
                    alias_config=args.alias,
                    collection_name=args.collection,
                    dry_run=args.dry_run,
                ),
                stage_fn=_promote_stage,
                run_id=args.build_run_id,
                persist=args.persist_build_run,
            )
            result = stage_result.result
        else:
            result = _promote_stage()
        _print_model(result)
        return 0

    if args.command == "plan":
        source_ids = _source_instance_ids(args.source_instance_ids)
        result = plan_build(
            BuildRequest(
                catalog_path=args.catalog,
                kb_id=args.kb,
                source_ids=source_ids,
                rag_data_root=args.rag_data_root,
            )
        )
        _print_model(result)
        return 0 if result.valid else 1

    if args.command == "status":
        runs = list_build_runs(rag_data_root=args.rag_data_root, kb_id=args.kb)
        print(json.dumps([r.to_summary() for r in runs], indent=2, sort_keys=True))
        return 0

    if args.command == "show-build-run":
        run = read_build_run(
            rag_data_root=args.rag_data_root,
            kb_id=args.kb,
            run_id=args.build_run_id,
        )
        _print_model(run)
        return 0

    raise ValueError(f"Unknown command '{args.command}'")


if __name__ == "__main__":
    raise SystemExit(main())
