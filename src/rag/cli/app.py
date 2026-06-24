"""Root Typer app and global options for the `rag` CLI.

Global overrides are diagnostic/development tools, not part of the normal
operator runbook: normal commands resolve the catalog path and artifact
root from runtime settings (CONFIG__CATALOG_PATH, settings.rag.data_root).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from rag.cli import alias as alias_cli
from rag.cli import benchmark as benchmark_cli
from rag.cli import catalog as catalog_cli
from rag.cli import release as release_cli
from rag.cli import source as source_cli
from rag.cli.factories import RagContext

app = typer.Typer(
    name="rag",
    help="Declarative control plane for RAG KB aliases: diff, apply, release, benchmark.",
    no_args_is_help=True,
)
app.add_typer(catalog_cli.app, name="catalog")
app.add_typer(alias_cli.app, name="alias")
app.add_typer(release_cli.app, name="release")
app.add_typer(benchmark_cli.app, name="benchmark")
app.add_typer(source_cli.app, name="source")


def _configure_stderr_logging() -> None:
    """Send logs to stderr, never stdout.

    `clients.observability.logging.configure_logging()` deliberately binds to stdout (the
    log sink convention for gateway/airflow-worker containers); this CLI
    instead reserves stdout for result data, per the plan's explicit
    "emit logs to stderr and result data to stdout" rule.

    Unconditionally replaces root handlers (rather than skipping when one
    already exists) because `logging.StreamHandler` captures a `sys.stderr`
    reference at construction time: under `typer.testing.CliRunner`, each
    invocation rebinds `sys.stderr` to a fresh capture buffer, so a handler
    built on an earlier invocation would silently write to a stale, already-
    discarded buffer instead of the current one.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(logging.StreamHandler(sys.stderr))
    root.setLevel(logging.INFO)


@app.callback()
def global_options(
    ctx: typer.Context,
    catalog: Optional[Path] = typer.Option(
        None, "--catalog", help="Catalog TOML path override (diagnostic/development only)."
    ),
    data_root: Optional[Path] = typer.Option(
        None, "--data-root", help="RAG artifact root override (diagnostic/development only)."
    ),
    output: Optional[str] = typer.Option(
        None, "--output", help="Output format: 'json' or 'table'. Defaults to json when not a TTY."
    ),
) -> None:
    """Store global options, resolved lazily by RagContext on first access.

    This callback itself must never require a connected settings
    environment: Typer/Click invoke a parent group's callback even when the
    actual request is `rag <group> <command> --help`.
    """
    _configure_stderr_logging()
    as_json = (output == "json") if output is not None else not sys.stdout.isatty()
    ctx.obj = RagContext(
        catalog_path_override=catalog,
        data_root_override=data_root,
        as_json=as_json,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
