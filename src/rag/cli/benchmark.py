"""`rag benchmark` command group: run, list, show."""

from __future__ import annotations

from typing import Optional

import typer

from app_config.catalog import build_source_instance_index
from app_config.runtime import get_settings
from rag.cli.factories import RagContext, load_catalog_config
from rag.cli.output import EXIT_OK, EXIT_USAGE_ERROR, emit, exit_code_for
from rag.evaluation.runner import BenchmarkRunSummary, run_benchmark
from rag.runtime import RagRuntime
from shared.db.eval_writer import get_evaluation_run, list_evaluation_runs

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
    """List recent persisted benchmark metric runs for a KB."""
    rag_ctx: RagContext = ctx.obj
    payload = list_evaluation_runs(
        db_url=get_settings().auth.agent042_db_url,
        knowledge_base=kb_id,
    )
    emit(payload, as_json=rag_ctx.as_json)
    raise typer.Exit(EXIT_OK)


@app.command("show")
def show(
    ctx: typer.Context,
    eval_run_id: str = typer.Argument(..., help="Evaluation run id."),
) -> None:
    """Show one persisted benchmark metric run and its samples."""
    rag_ctx: RagContext = ctx.obj
    payload = get_evaluation_run(
        db_url=get_settings().auth.agent042_db_url,
        eval_run_id=eval_run_id,
    )
    if payload is None:
        emit(
            {"error": f"Evaluation run '{eval_run_id}' not found"},
            as_json=rag_ctx.as_json,
        )
        raise typer.Exit(EXIT_USAGE_ERROR)

    emit(payload, as_json=rag_ctx.as_json)
    raise typer.Exit(EXIT_OK)
