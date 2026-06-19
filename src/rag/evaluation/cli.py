"""Command-line entry point for catalog-declared RAG benchmarks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from app_config.runtime import get_settings
from rag.evaluation.runner import run_benchmark
from rag.runtime import RagRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run-benchmark")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--source-instance", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--rag-data-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    runtime = RagRuntime(settings=settings)
    try:
        judge = runtime.judge_settings()
        result = run_benchmark(
            catalog_path=args.catalog,
            source_instance_id=args.source_instance,
            alias=args.alias,
            rag_data_root=args.rag_data_root,
            db_url=settings.auth.agent042_db_url,
            runtime=runtime,
            base_model=settings.vllm.model,
            generation_llm=runtime.generation_llm(),
            judge_llm=runtime.judge_llm(),
            judge_model=judge.model,
            judge_backend=judge.backend,
        )
    finally:
        runtime.close()
    print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
