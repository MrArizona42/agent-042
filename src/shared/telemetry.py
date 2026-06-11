"""OpenTelemetry bootstrap helpers for service processes."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_TRACING_CONFIGURED = False
_FASTAPI_INSTRUMENTED_APPS: set[int] = set()
_HTTPX_INSTRUMENTED = False
_REDIS_INSTRUMENTED = False
_CELERY_INSTRUMENTED = False


def configure_tracing(*, service: str) -> bool:
    """Configure OTLP tracing when the server deployment enables it."""

    global _TRACING_CONFIGURED
    if _TRACING_CONFIGURED:
        return True
    if not _tracing_enabled():
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning("OpenTelemetry tracing requested but dependencies are unavailable")
        return False

    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _TRACING_CONFIGURED = True
    logger.info("OpenTelemetry tracing configured", extra={"event": "otel.tracing.configured"})
    return True


def instrument_fastapi_app(app: Any, *, service: str) -> None:
    """Attach FastAPI instrumentation to an app when tracing is enabled."""

    if id(app) in _FASTAPI_INSTRUMENTED_APPS or not configure_tracing(service=service):
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        logger.warning("FastAPI OpenTelemetry instrumentation is unavailable")
        return

    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="/health,/metrics",
    )
    _FASTAPI_INSTRUMENTED_APPS.add(id(app))


def instrument_httpx(*, service: str) -> None:
    """Instrument httpx clients once per process."""

    global _HTTPX_INSTRUMENTED
    if _HTTPX_INSTRUMENTED or not configure_tracing(service=service):
        return
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    except ImportError:
        logger.warning("httpx OpenTelemetry instrumentation is unavailable")
        return

    HTTPXClientInstrumentor().instrument()
    _HTTPX_INSTRUMENTED = True


def instrument_redis(*, service: str) -> None:
    """Instrument redis clients once per process."""

    global _REDIS_INSTRUMENTED
    if _REDIS_INSTRUMENTED or not configure_tracing(service=service):
        return
    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor
    except ImportError:
        logger.warning("Redis OpenTelemetry instrumentation is unavailable")
        return

    RedisInstrumentor().instrument()
    _REDIS_INSTRUMENTED = True


def instrument_celery(*, service: str) -> None:
    """Instrument Celery producer and worker hooks once per process."""

    global _CELERY_INSTRUMENTED
    if _CELERY_INSTRUMENTED or not configure_tracing(service=service):
        return
    try:
        from opentelemetry.instrumentation.celery import CeleryInstrumentor
    except ImportError:
        logger.warning("Celery OpenTelemetry instrumentation is unavailable")
        return

    CeleryInstrumentor().instrument()
    _CELERY_INSTRUMENTED = True


def get_tracer(name: str):
    """Return an OpenTelemetry tracer or a no-op tracer when OTel is absent."""

    try:
        from opentelemetry import trace
    except ImportError:
        return _NoopTracer()
    return trace.get_tracer(name)


def current_trace_context() -> dict[str, str]:
    """Return current trace identifiers as strings when a span is active."""

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


def _tracing_enabled() -> bool:
    disabled = os.getenv("OTEL_SDK_DISABLED", "").lower()
    if disabled in {"1", "true", "yes"}:
        return False
    return bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))


class _NoopTracer:
    def start_as_current_span(self, _name: str, **_kwargs: Any):
        return _NoopSpanContext()


class _NoopSpanContext:
    def __enter__(self):
        return _NoopSpan()

    def __exit__(self, *_args: Any) -> None:
        return None


class _NoopSpan:
    def set_attribute(self, _key: str, _value: Any) -> None:
        return None
