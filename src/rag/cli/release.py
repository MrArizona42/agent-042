"""`rag release` command group: list, show."""

from __future__ import annotations

import typer

from rag.cli.factories import RagContext, build_release_repository
from rag.cli.output import EXIT_OK, EXIT_USAGE_ERROR, emit

app = typer.Typer(help="Inspect immutable, content-identified releases.")


@app.command("list")
def list_releases(
    ctx: typer.Context,
    kb_id: str = typer.Option(..., "--kb", help="Knowledge base id."),
) -> None:
    """List releases for a KB, newest first."""
    rag_ctx: RagContext = ctx.obj
    releases = build_release_repository(rag_ctx).list_for_kb(kb_id)
    emit(releases, as_json=rag_ctx.as_json)
    raise typer.Exit(EXIT_OK)


@app.command("show")
def show(
    ctx: typer.Context,
    release_id: str = typer.Argument(..., help="Release id."),
) -> None:
    """Show one release's full provenance."""
    rag_ctx: RagContext = ctx.obj
    release = build_release_repository(rag_ctx).get(release_id)
    if release is None:
        emit({"error": f"Release '{release_id}' not found"}, as_json=rag_ctx.as_json)
        raise typer.Exit(EXIT_USAGE_ERROR)

    emit(release, as_json=rag_ctx.as_json)
    raise typer.Exit(EXIT_OK)
