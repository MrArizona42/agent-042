"""Celery application configuration."""

from __future__ import annotations

from celery import Celery

from shared.config import get_settings
from shared.logging import configure_logging
from shared.telemetry import instrument_celery, instrument_httpx, instrument_redis

configure_logging(service="worker")
instrument_celery(service="worker")
instrument_httpx(service="worker")
instrument_redis(service="worker")

settings = get_settings()
platform = settings.platform
worker = settings.worker

if not platform.celery_broker_url:
    raise RuntimeError(
        "RabbitMQ broker URL could not be derived. Check RABBITMQ_DEFAULT_USER, "
        "RABBITMQ_DEFAULT_PASS, and NETWORK__RABBITMQ_AMQP__*."
    )

celery_app = Celery(
    "worker",
    broker=platform.celery_broker_url,
    # No result backend needed - we use Redis Pub/Sub for streaming
    include=["worker.tasks"],
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Worker settings
    worker_prefetch_multiplier=1,  # Keep queue fairness when eval and UI share one broker queue
    worker_concurrency=worker.concurrency,
    worker_pool=worker.pool,
    worker_send_task_events=worker.send_task_events,
    task_track_started=True,
    task_send_sent_event=True,
    worker_cancel_long_running_tasks_on_connection_loss=(
        worker.cancel_long_running_tasks_on_connection_loss
    ),
    # Task acknowledgment
    task_acks_late=True,  # Ack after completion (allows retry on crash)
    task_reject_on_worker_lost=True,
    # Retry settings
    task_default_retry_delay=worker.retry_delay,
    task_max_retries=worker.max_retries,
    # Timeouts
    task_soft_time_limit=worker.default_timeout - 10,
    task_time_limit=worker.default_timeout,
    # Keep the broker connection alive for the full task duration.
    broker_heartbeat=worker.default_timeout,
    broker_transport_options={"heartbeat": worker.default_timeout},
)
