"""`rag alias` command group: diff, apply, status."""

from __future__ import annotations

from typing import Optional

import typer

from rag.cli.factories import RagContext, build_alias_service, load_catalog_config
from rag.cli.output import EXIT_DRIFT, EXIT_INFRA_ERROR, EXIT_OK, emit, exit_code_for
from rag.control_plane.alias_service import AliasApplyRequest, AliasDiffRequest

app = typer.Typer(help="Alias diff and apply: desired vs applied state.")


@app.command("diff")
def diff(
    ctx: typer.Context,
    kb_id: str = typer.Argument(..., help="Knowledge base id."),
    alias: str = typer.Argument(..., help="Alias name."),
) -> None:
    """Compare desired (catalog) and applied (Postgres) state for one alias."""
    rag_ctx: RagContext = ctx.obj
    service = build_alias_service(rag_ctx)
    try:
        result = service.diff(AliasDiffRequest(kb_id=kb_id, alias=alias))
    except Exception as exc:
        emit({"error": str(exc)}, as_json=rag_ctx.as_json)
        raise typer.Exit(exit_code_for(exc)) from None

    emit(result, as_json=rag_ctx.as_json)
    has_drift = (
        result.build_drift
        or result.retrieval_drift
        or result.source_declaration_drift
        or bool(result.provider_mismatches)
    )
    raise typer.Exit(EXIT_DRIFT if has_drift else EXIT_OK)


@app.command("apply")
def apply(
    ctx: typer.Context,
    kb_id: str = typer.Argument(..., help="Knowledge base id."),
    alias: str = typer.Argument(..., help="Alias name."),
    release_id: Optional[str] = typer.Option(
        None, "--release", help="Disambiguate an ambiguous reusable release."
    ),
    allow_unevaluated: bool = typer.Option(
        False,
        "--allow-unevaluated",
        help=(
            "Bootstrap/emergency override: activate the default alias without evaluation coverage."
        ),
    ),
    allow_build_default: bool = typer.Option(
        False,
        "--allow-build-default",
        help="Bootstrap override: allow building a new release directly for the default alias.",
    ),
    refresh_sources: bool = typer.Option(
        False,
        "--refresh-sources",
        help=(
            "Re-fetch source content before deciding whether a rebuild is needed. "
            "Expensive; stays explicit rather than happening on every apply."
        ),
    ),
) -> None:
    """Make this KB alias match its catalog declaration."""
    rag_ctx: RagContext = ctx.obj
    service = build_alias_service(rag_ctx)
    try:
        result = service.apply(
            AliasApplyRequest(
                kb_id=kb_id,
                alias=alias,
                release_id=release_id,
                allow_unevaluated=allow_unevaluated,
                allow_build_default=allow_build_default,
                refresh_sources=refresh_sources,
            )
        )
    except Exception as exc:
        emit({"error": str(exc)}, as_json=rag_ctx.as_json)
        raise typer.Exit(exit_code_for(exc)) from None

    emit(result, as_json=rag_ctx.as_json)
    raise typer.Exit(EXIT_OK)


@app.command("status")
def status(
    ctx: typer.Context,
    kb_id: str = typer.Argument(..., help="Knowledge base id."),
) -> None:
    """Show diff for every alias declared on a KB."""
    rag_ctx: RagContext = ctx.obj
    catalog_cfg = load_catalog_config(rag_ctx)
    kb_cfg = next((kb for kb in catalog_cfg.knowledge_bases if kb.id == kb_id), None)
    if kb_cfg is None:
        emit({"error": f"Unknown KB '{kb_id}'"}, as_json=rag_ctx.as_json)
        raise typer.Exit(2)

    service = build_alias_service(rag_ctx, catalog_cfg=catalog_cfg)
    results = []
    has_drift = False
    has_errors = False
    for alias_name in kb_cfg.aliases:
        try:
            result = service.diff(AliasDiffRequest(kb_id=kb_id, alias=alias_name))
        except Exception as exc:
            results.append({"kb_id": kb_id, "alias": alias_name, "error": str(exc)})
            has_errors = True
            continue
        results.append(result)
        has_drift = has_drift or (
            result.build_drift
            or result.retrieval_drift
            or result.source_declaration_drift
            or bool(result.provider_mismatches)
        )

    emit(results, as_json=rag_ctx.as_json)
    if has_errors:
        raise typer.Exit(EXIT_INFRA_ERROR)
    raise typer.Exit(EXIT_DRIFT if has_drift else EXIT_OK)
