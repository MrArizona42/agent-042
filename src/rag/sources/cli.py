"""CLI entry points for source-build lifecycle commands."""

from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from app_config.catalog import CatalogConfig, load_catalog
from rag.embeddings import EmbeddingService
from rag.sources.build import build_catalog_source, build_catalog_sources, resolve_catalog_sources
from rag.sources.bundles import collect_source_bundles, collect_source_chunks
from rag.sources.chunks import ChunkingConfig
from rag.sources.materialize import (
    collection_name_for_build,
    materialize_kb_collection,
    promote_materialized_alias,
    retrieval_capability_for_strategy,
    validate_strategy_supported,
)
from rag.sparse_encoder import SparseEncoderService
from rag.vector_store import QdrantVectorStore
from shared.config import get_settings


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


def _source_ids(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    source_ids: list[str] = []
    for value in values:
        source_ids.extend(item.strip() for item in value.split(",") if item.strip())
    if not source_ids or any(source_id.lower() == "all" for source_id in source_ids):
        return None
    return source_ids


def _catalog_source_ids(
    *,
    catalog_path: Path | str,
    kb_id: str,
    source_ids: list[str] | None,
) -> list[str]:
    path = Path(catalog_path)
    catalog = CatalogConfig(**tomllib.loads(path.read_text(encoding="utf-8")))
    return [
        source.id
        for source in resolve_catalog_sources(
            catalog,
            kb_id=kb_id,
            source_instance_ids=source_ids,
        )
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rag.sources.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_source_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("--catalog", required=True)
        command.add_argument("--kb", required=True)
        command.add_argument(
            "--source",
            action="append",
            dest="sources",
            help="Source instance id. Repeat for subsets, use 'all', or omit for all.",
        )
        command.add_argument("--rag-data-root", required=True)
        command.add_argument("--document-id", action="append", dest="document_ids")
        command.add_argument("--limit", type=int)

    build_source = subparsers.add_parser("build-source")
    add_common_source_args(build_source)
    build_source.add_argument("--force-fetch", action="store_true")
    build_source.add_argument("--force-extract", action="store_true")
    build_source.add_argument("--force-chunk", action="store_true")
    build_source.add_argument("--chunk-size", type=int)
    build_source.add_argument("--chunk-overlap", type=int)

    collect_bundle = subparsers.add_parser("collect-bundle")
    add_common_source_args(collect_bundle)

    materialize = subparsers.add_parser("materialize")
    add_common_source_args(materialize)
    materialize.add_argument("--alias-config", required=True)
    materialize.add_argument("--collection")
    materialize.add_argument("--force-recreate", action="store_true")

    promote = subparsers.add_parser("promote-alias")
    promote.add_argument(
        "--catalog",
        help="Catalog TOML path. Defaults to CONFIG__CATALOG_PATH when omitted.",
    )
    promote.add_argument("--kb", required=True)
    promote.add_argument("--alias", required=True)
    promote.add_argument("--collection", required=True)

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
    build_catalog_source_fn: Callable[..., Any] = build_catalog_source,
    build_catalog_sources_fn: Callable[..., Any] = build_catalog_sources,
    collect_source_chunks_fn: Callable[..., Any] = collect_source_chunks,
    collect_source_bundles_fn: Callable[..., Any] = collect_source_bundles,
    materialize_kb_collection_fn: Callable[..., Any] = materialize_kb_collection,
    promote_materialized_alias_fn: Callable[..., Any] = promote_materialized_alias,
) -> int:
    """Run the source lifecycle CLI."""
    args = _parser().parse_args(argv)

    if args.command == "build-source":
        source_ids = _source_ids(args.sources)
        if source_ids is not None and len(source_ids) == 1:
            result = build_catalog_source_fn(
                catalog_path=args.catalog,
                kb_id=args.kb,
                source_instance_id=source_ids[0],
                rag_data_root=args.rag_data_root,
                document_ids=_document_ids(args.document_ids),
                limit=args.limit,
                force_fetch=args.force_fetch,
                force_extract=args.force_extract,
                force_chunk=args.force_chunk,
                chunking=_chunking_from_args(args),
            )
        else:
            result = build_catalog_sources_fn(
                catalog_path=args.catalog,
                kb_id=args.kb,
                source_instance_ids=source_ids,
                rag_data_root=args.rag_data_root,
                document_ids=_document_ids(args.document_ids),
                limit=args.limit,
                force_fetch=args.force_fetch,
                force_extract=args.force_extract,
                force_chunk=args.force_chunk,
                chunking=_chunking_from_args(args),
            )
        _print_model(result)
        return 0

    if args.command == "collect-bundle":
        source_ids = _source_ids(args.sources)
        if source_ids is not None and len(source_ids) == 1:
            result = collect_source_chunks_fn(
                rag_data_root=args.rag_data_root,
                kb_id=args.kb,
                source_instance_id=source_ids[0],
                document_ids=_document_ids(args.document_ids),
                limit=args.limit,
            )
        else:
            result = collect_source_bundles_fn(
                rag_data_root=args.rag_data_root,
                kb_id=args.kb,
                source_instance_ids=_catalog_source_ids(
                    catalog_path=args.catalog,
                    kb_id=args.kb,
                    source_ids=source_ids,
                ),
                document_ids=_document_ids(args.document_ids),
                limit=args.limit,
            )
        _print_model(result)
        return 0

    if args.command == "materialize":
        settings = get_settings()
        strategy = _alias_strategy(
            catalog_path=args.catalog,
            kb_id=args.kb,
            alias=args.alias_config,
        )
        capability = retrieval_capability_for_strategy(strategy)  # type: ignore[arg-type]
        collection_name = args.collection or collection_name_for_build(
            kb_id=args.kb,
        )
        source_ids = _source_ids(args.sources)
        resolved_source_ids = _catalog_source_ids(
            catalog_path=args.catalog,
            kb_id=args.kb,
            source_ids=source_ids,
        )
        if len(resolved_source_ids) == 1:
            bundles = [
                collect_source_chunks_fn(
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
        result = materialize_kb_collection_fn(
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
        )
        _print_model(result)
        return 0

    if args.command == "promote-alias":
        catalog_path = _catalog_path_from_args(args)
        strategy = _alias_strategy(catalog_path=catalog_path, kb_id=args.kb, alias=args.alias)
        vector_store = _vector_store(collection_name=args.collection)
        payload = vector_store.read_meta()
        if payload is None:
            raise RuntimeError(f"Collection '{args.collection}' has no attestation metadata")
        validate_strategy_supported(
            retrieval_strategy=strategy,  # type: ignore[arg-type]
            retrieval_capability=payload.get("retrieval_capability"),  # type: ignore[arg-type]
        )
        result = promote_materialized_alias_fn(
            kb_id=args.kb,
            alias=args.alias,
            collection_name=args.collection,
            vector_store=vector_store,
        )
        _print_model(result)
        return 0

    raise ValueError(f"Unknown command '{args.command}'")


if __name__ == "__main__":
    raise SystemExit(main())
