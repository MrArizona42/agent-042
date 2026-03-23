"""Celery eval-worker module for metric computation (bert-score, rouge, etc.)."""

from eval_worker.celery_app import celery_app
from eval_worker.tasks import calculate_metrics_task

__all__ = ["celery_app", "calculate_metrics_task"]
