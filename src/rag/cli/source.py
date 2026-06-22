"""`rag source` command group: expert diagnostics and explicit cache invalidation.

Not part of the normal operator workflow -- `rag alias apply` resolves and
builds sources on its own. These exist for debugging a specific source
instance's cache state directly.
"""

from __future__ import annotations

import typer

from app_config.catalog import build_source_instance_index, conventional_manifest_path
from rag.cli.factories import RagContext, build_adapter_registry, load_catalog_config
from rag.cli.output import EXIT_OK, EXIT_USAGE_ERROR, emit
from rag.sources.build import build_source_instance_by_global_id

app = typer.Typer(help="Expert source diagnostics. Not part of the normal workflow.")


@app.command("inspect")
def inspect(
    ctx: typer.Context,
    source_instance_id: str = typer.Argument(..., help="Global source instance id."),
) -> None:
    """Show one source instance's declaration and on-disk cache state."""
    rag_ctx: RagContext = ctx.obj
    catalog_cfg = load_catalog_config(rag_ctx)
    source_index = build_source_instance_index(catalog_cfg)
    try:
        instance = source_index.get(source_instance_id)
    except ValueError as exc:
        emit({"error": str(exc)}, as_json=rag_ctx.as_json)
        raise typer.Exit(EXIT_USAGE_ERROR) from None

    manifest_path = conventional_manifest_path(rag_ctx.data_root, source_instance_id)
    base = rag_ctx.data_root / "source_instances" / source_instance_id
    payload = {
        "source_instance_id": instance.id,
        "knowledge_base": instance.knowledge_base,
        "role": instance.role,
        "adapter": f"{instance.adapter.id}@{instance.adapter.version}",
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.is_file(),
        "raw_count": len(list((base / "raw").glob("**/*"))) if (base / "raw").is_dir() else 0,
        "extracted_count": (
            len(list((base / "extracted").glob("*.json"))) if (base / "extracted").is_dir() else 0
        ),
        "chunk_digest_dirs": (
            sorted(p.name for p in (base / "chunks").iterdir() if p.is_dir())
            if (base / "chunks").is_dir()
            else []
        ),
    }
    emit(payload, as_json=rag_ctx.as_json)
    raise typer.Exit(EXIT_OK)


@app.command("rebuild")
def rebuild(
    ctx: typer.Context,
    source_instance_id: str = typer.Argument(..., help="Global source instance id."),
) -> None:
    """Force-refetch, re-extract, and re-chunk one source instance.

    Explicit cache invalidation only; this does not build a release or touch
    any alias. Run `rag alias apply` afterward to build a release from the
    refreshed cache.
    """
    rag_ctx: RagContext = ctx.obj
    catalog_cfg = load_catalog_config(rag_ctx)
    adapter_registry = build_adapter_registry(catalog_cfg)
    result = build_source_instance_by_global_id(
        catalog_path=rag_ctx.catalog_path,
        source_instance_id=source_instance_id,
        rag_data_root=rag_ctx.data_root,
        force_fetch=True,
        force_extract=True,
        force_chunk=True,
        adapter_registry=adapter_registry,
    )
    emit(result, as_json=rag_ctx.as_json)
    raise typer.Exit(EXIT_OK)
