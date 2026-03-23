"""Celery tasks executed by the eval-worker."""

from __future__ import annotations

import logging
from typing import Any

from eval_worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="eval_worker.tasks.calculate_metrics_task")
def calculate_metrics_task(
    self,
    metric: str,
    prediction_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compute a single metric on pre-fetched predictions.

    This task runs inside the eval-worker container, which has
    ``bert-score`` / ``torch`` (CPU) installed.

    Returns:
        List of metric row dicts.
    """
    logger.info("eval-worker task %s: metric=%s", self.request.id, metric)

    from experiments.scripts.eval.runner import calculate_metrics

    rows = calculate_metrics(metric=metric, prediction_data=prediction_data)

    logger.info("eval-worker task %s: %d metric rows computed", self.request.id, len(rows))
    return rows
