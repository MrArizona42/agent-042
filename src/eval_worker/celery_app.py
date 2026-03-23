"""Celery application for the eval-worker."""

from __future__ import annotations

from celery import Celery

from eval_worker.config import get_eval_worker_settings

settings = get_eval_worker_settings()

celery_app = Celery(
    "eval_worker",
    broker=settings.celery_broker_url,
    backend="rpc://",
    include=["eval_worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    worker_concurrency=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_soft_time_limit=1790,
    task_time_limit=1800,
)
