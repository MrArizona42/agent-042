"""Celery application configuration."""

from __future__ import annotations

from pathlib import Path

from celery import Celery

from shared.config import bootstrap_local_settings_env
from worker.config import get_worker_settings

bootstrap_local_settings_env(repo_root=Path(__file__).resolve().parents[2])

settings = get_worker_settings()

celery_app = Celery(
    "worker",
    broker=settings.celery_broker_url,
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
    worker_concurrency=settings.worker_concurrency,
    worker_pool=settings.worker_pool,
    worker_send_task_events=settings.worker_send_task_events,
    task_track_started=True,
    task_send_sent_event=True,
    worker_cancel_long_running_tasks_on_connection_loss=(
        settings.worker_cancel_long_running_tasks_on_connection_loss
    ),
    # Task acknowledgment
    task_acks_late=True,  # Ack after completion (allows retry on crash)
    task_reject_on_worker_lost=True,
    # Retry settings
    task_default_retry_delay=settings.task_retry_delay,
    task_max_retries=settings.task_max_retries,
    # Timeouts
    task_soft_time_limit=settings.task_default_timeout - 10,
    task_time_limit=settings.task_default_timeout,
    # Keep the broker connection alive for the full task duration.
    broker_heartbeat=settings.task_default_timeout,
    broker_transport_options={"heartbeat": settings.task_default_timeout},
)
