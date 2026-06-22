"""JSON/Rich table rendering and exit-code mapping for the `rag` CLI.

Typer command functions render through `emit()` and translate domain
exceptions to one of these exit codes via `exit_code_for()`; they contain no
rendering or error-mapping logic of their own.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

EXIT_OK = 0
EXIT_DRIFT = 1
EXIT_USAGE_ERROR = 2
EXIT_CONFLICT = 3
EXIT_INFRA_ERROR = 4

# Substrings from rag.control_plane.alias_service.AliasApplyError messages
# that this project's own code controls, used to assign a more specific exit
# code than the generic "apply was refused". Conflict is checked before
# usage since both can mention an alias/KB name.
_CONFLICT_MARKERS = (
    "multiple releases match",
    "does not match desired",
    "incompatible with",
    "evaluation coverage",
    "allow_build_default",
    "allow_unevaluated",
)
_USAGE_MARKERS = ("Unknown KB", "Unknown alias", "not found")


def exit_code_for(exc: Exception) -> int:
    """Map a raised exception to a stable CLI exit code."""
    message = str(exc)
    if any(marker in message for marker in _CONFLICT_MARKERS):
        return EXIT_CONFLICT
    if any(marker in message for marker in _USAGE_MARKERS):
        return EXIT_USAGE_ERROR
    return EXIT_INFRA_ERROR


def to_json_payload(value: Any) -> Any:
    """Recursively convert Pydantic models (and containers of them) to plain JSON."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: to_json_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_payload(item) for item in value]
    return value


def _json_default(value: Any) -> str:
    return str(value)


def emit(value: Any, *, as_json: bool) -> None:
    """Render *value* to stdout: JSON when as_json, a Rich table/panel otherwise.

    Logs go to stderr elsewhere in the CLI; this only ever writes result data
    to stdout, so JSON output is always machine-parseable.
    """
    if as_json:
        print(json.dumps(to_json_payload(value), indent=2, sort_keys=True, default=_json_default))
        return
    _emit_human(value)


def _emit_human(value: Any) -> None:
    from rich.console import Console

    console = Console()
    if isinstance(value, BaseModel):
        _emit_table(console, [value])
        return
    if isinstance(value, list) and value and isinstance(value[0], BaseModel):
        _emit_table(console, value)
        return
    if isinstance(value, list) and not value:
        console.print("(no results)")
        return
    console.print(value)


def _emit_table(console: Any, models: list[BaseModel]) -> None:
    from rich.table import Table

    payloads = [model.model_dump(mode="json") for model in models]
    table = Table(*payloads[0].keys())
    for payload in payloads:
        table.add_row(*(_cell(value) for value in payload.values()))
    console.print(table)


def _cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=_json_default)
    return str(value)
