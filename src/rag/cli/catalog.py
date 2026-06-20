"""`rag catalog` command group."""

from __future__ import annotations

import typer
from pydantic import BaseModel, ConfigDict

from app_config.catalog import materialize_catalog
from rag.cli.factories import RagContext, load_catalog_config
from rag.cli.output import EXIT_OK, EXIT_USAGE_ERROR, emit

app = typer.Typer(help="Validate the desired-state catalog.")


class CatalogValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    catalog_path: str
    valid: bool
    schema_version: int | None = None
    knowledge_base_count: int = 0
    error: str | None = None


@app.command("validate")
def validate(ctx: typer.Context) -> None:
    """Validate the catalog file: schema, alias compatibility, and references."""
    rag_ctx: RagContext = ctx.obj
    try:
        catalog_cfg = load_catalog_config(rag_ctx)
        materialize_catalog(catalog_cfg)
    except Exception as exc:
        result = CatalogValidationResult(
            catalog_path=str(rag_ctx.catalog_path), valid=False, error=str(exc)
        )
        emit(result, as_json=rag_ctx.as_json)
        raise typer.Exit(EXIT_USAGE_ERROR) from None

    result = CatalogValidationResult(
        catalog_path=str(rag_ctx.catalog_path),
        valid=True,
        schema_version=catalog_cfg.schema_version,
        knowledge_base_count=len(catalog_cfg.knowledge_bases),
    )
    emit(result, as_json=rag_ctx.as_json)
    raise typer.Exit(EXIT_OK)
