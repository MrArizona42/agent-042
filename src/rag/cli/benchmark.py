"""`rag benchmark` command group: run, list, show."""

from __future__ import annotations

import json
from typing import Optional

import typer

from app_config.catalog import build_source_instance_index
from app_config.runtime import get_settings
from rag.cli.factories import RagContext, load_catalog_config
from rag.cli.output import EXIT_OK, EXIT_USAGE_ERROR, emit, exit_code_for
from rag.evaluation.runner import BenchmarkRunSummary, run_benchmark
from rag.runtime import RagRuntime
from rag.sources.benchmark_prep import metadata_artifact_path

app = typer.Typer(help="Release-aware benchmark execution.")


def _run_one(*, ctx: RagContext, source_instance_id: str, alias: str) -> BenchmarkRunSummary:
    settings = get_settings()
    runtime = RagRuntime(settings=settings)
    try:
        judge = runtime.judge_settings()
        return run_benchmark(
            catalog_path=ctx.catalog_path,
            source_instance_id=source_instance_id,
            alias=alias,
            rag_data_root=ctx.data_root,
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


@app.command("run")
def run(
    ctx: typer.Context,
    source_instance_id: Optional[str] = typer.Argument(
        None, help="Run only this benchmark source instance."
    ),
    kb_id: Optional[str] = typer.Option(
        None, "--kb", help="Run every benchmark source instance attached to this KB."
    ),
    alias: str = typer.Option(..., "--alias", help="Alias to benchmark (always explicit)."),
) -> None:
    """Execute one or more prepared benchmarks against one explicit alias."""
    rag_ctx: RagContext = ctx.obj
    if bool(source_instance_id) == bool(kb_id):
        emit(
            {"error": "exactly one of a benchmark source instance or --kb is required"},
            as_json=rag_ctx.as_json,
        )
        raise typer.Exit(EXIT_USAGE_ERROR)

    if source_instance_id:
        target_ids = [source_instance_id]
    else:
        catalog_cfg = load_catalog_config(rag_ctx)
        source_index = build_source_instance_index(catalog_cfg)
        target_ids = [instance.id for instance in source_index.benchmark_for_kb(kb_id)]
        if not target_ids:
            emit(
                {"error": f"KB '{kb_id}' has no benchmark source instances"},
                as_json=rag_ctx.as_json,
            )
            raise typer.Exit(EXIT_USAGE_ERROR)

    results = []
    for target_id in target_ids:
        try:
            results.append(_run_one(ctx=rag_ctx, source_instance_id=target_id, alias=alias))
        except Exception as exc:
            emit({"source_instance_id": target_id, "error": str(exc)}, as_json=rag_ctx.as_json)
            raise typer.Exit(exit_code_for(exc)) from None

    emit(results if len(results) > 1 else results[0], as_json=rag_ctx.as_json)
    raise typer.Exit(EXIT_OK)


@app.command("list")
def list_benchmarks(
    ctx: typer.Context,
    kb_id: str = typer.Option(..., "--kb", help="Knowledge base id."),
) -> None:
    """List benchmark source instances attached to a KB."""
    rag_ctx: RagContext = ctx.obj
    catalog_cfg = load_catalog_config(rag_ctx)
    source_index = build_source_instance_index(catalog_cfg)
    instances = source_index.benchmark_for_kb(kb_id)
    payload = [
        {
            "source_instance_id": instance.id,
            "knowledge_base": instance.knowledge_base,
            "adapter": f"{instance.adapter.id}@{instance.adapter.version}",
            "suites": list(instance.benchmark.suites) if instance.benchmark else [],
        }
        for instance in instances
    ]
    emit(payload, as_json=rag_ctx.as_json)
    raise typer.Exit(EXIT_OK)


@app.command("show")
def show(
    ctx: typer.Context,
    source_instance_id: str = typer.Argument(..., help="Benchmark source instance id."),
) -> None:
    """Show the prepared-artifact metadata for one benchmark source instance."""
    rag_ctx: RagContext = ctx.obj
    path = metadata_artifact_path(rag_ctx.data_root, source_instance_id)
    if not path.is_file():
        emit(
            {"error": f"Benchmark '{source_instance_id}' is not prepared"},
            as_json=rag_ctx.as_json,
        )
        raise typer.Exit(EXIT_USAGE_ERROR)

    emit(json.loads(path.read_text(encoding="utf-8")), as_json=rag_ctx.as_json)
    raise typer.Exit(EXIT_OK)
