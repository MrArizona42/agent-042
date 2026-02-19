"""Celery application configuration."""

from __future__ import annotations

from celery import Celery

from worker.config import get_worker_settings

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
    worker_prefetch_multiplier=1,  # One task at a time for GPU workloads
    worker_concurrency=1,  # Single worker process (GPU bound)
    # Task acknowledgment
    task_acks_late=True,  # Ack after completion (allows retry on crash)
    task_reject_on_worker_lost=True,
    # Retry settings
    task_default_retry_delay=settings.task_retry_delay,
    task_max_retries=settings.task_max_retries,
    # Timeouts
    task_soft_time_limit=settings.task_default_timeout - 10,
    task_time_limit=settings.task_default_timeout,
)
