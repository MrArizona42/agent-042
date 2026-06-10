"""Structured logging helpers for long-running services."""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

_LOG_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "agent042_log_context",
    default={},
)
_BASE_RECORD_ATTRS = frozenset(logging.makeLogRecord({}).__dict__)
_PREVIOUS_RECORD_FACTORY = logging.getLogRecordFactory()


def bind_log_context(**fields: Any) -> contextvars.Token[dict[str, Any]]:
    """Bind structured fields to logs emitted in the current context."""

    current = dict(_LOG_CONTEXT.get())
    current.update({key: value for key, value in fields.items() if value is not None})
    return _LOG_CONTEXT.set(current)


def reset_log_context(token: contextvars.Token[dict[str, Any]]) -> None:
    """Reset context fields to a previous binding token."""

    _LOG_CONTEXT.reset(token)


def clear_log_context() -> None:
    """Clear all structured fields in the current context."""

    _LOG_CONTEXT.set({})


def _record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
    record = _PREVIOUS_RECORD_FACTORY(*args, **kwargs)
    for key, value in _LOG_CONTEXT.get().items():
        if not hasattr(record, key):
            setattr(record, key, value)
    return record


class JsonLogFormatter(logging.Formatter):
    """Format log records as single-line JSON for Docker/Loki collection."""

    def __init__(self, *, service: str) -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self._service,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_current_trace_fields())
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        for key, value in record.__dict__.items():
            if key in _BASE_RECORD_ATTRS or key in payload:
                continue
            payload[key] = _json_safe(value)

        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def configure_logging(
    *,
    service: str,
    level: str | int | None = None,
    json_logs: bool = True,
) -> None:
    """Configure root logging for one service process."""

    resolved_level = _resolve_level(level or os.getenv("LOG_LEVEL", "INFO"))
    handler = logging.StreamHandler(sys.stdout)
    if json_logs:
        handler.setFormatter(JsonLogFormatter(service=service))
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(service)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        handler.addFilter(_ServiceFilter(service))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved_level)
    logging.setLogRecordFactory(_record_factory)


class _ServiceFilter(logging.Filter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self._service = service

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "service"):
            record.service = self._service
        return True


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    parsed = logging.getLevelName(level.upper())
    return parsed if isinstance(parsed, int) else logging.INFO


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _current_trace_fields() -> dict[str, str]:
    try:
        from opentelemetry import trace
    except ImportError:
        return {}

    span_context = trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return {}
    return {
        "trace_id": format(span_context.trace_id, "032x"),
        "span_id": format(span_context.span_id, "016x"),
    }
