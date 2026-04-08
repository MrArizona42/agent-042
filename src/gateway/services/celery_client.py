"""Celery client for enqueueing LLM tasks from the Gateway.

This module provides a client interface for sending tasks to
the Celery worker without importing the worker's Celery app directly.
"""

from __future__ import annotations

import logging
from typing import Any

from celery import Celery

logger = logging.getLogger(__name__)


class CeleryClient:
    """Client for enqueueing Celery tasks from the Gateway."""

    def __init__(self, broker_url: str):
        """Initialize Celery client.

        Args:
            broker_url: RabbitMQ broker URL
        """
        self._broker_url = broker_url
        self._app: Celery | None = None

    def _get_app(self) -> Celery:
        """Get or create Celery app for sending tasks."""
        if self._app is None:
            self._app = Celery(
                "gateway_client",
                broker=self._broker_url,
            )
            self._app.conf.update(
                task_serializer="json",
                accept_content=["json"],
            )
        return self._app

    def close(self) -> None:
        """Close the Celery app and its broker connection."""
        if self._app is not None:
            self._app.close()
            self._app = None
            logger.info("Celery client closed")

    def enqueue_generate_response(
        self,
        conversation_id: str,
        generation_payload: dict[str, Any],
        budget_meta: dict[str, Any],
    ) -> str:
        """Enqueue a generate_response task.

        Args:
            conversation_id: Unique conversation identifier
            generation_payload: Chat completion payload without final max_tokens
            budget_meta: Exact-budget metadata for worker-side preflight

        Returns:
            Task ID
        """
        app = self._get_app()

        # Send task by name (worker.tasks.generate_response)
        task = app.send_task(
            "worker.tasks.generate_response",
            kwargs={
                "conversation_id": conversation_id,
                "generation_payload": generation_payload,
                "budget_meta": budget_meta,
            },
        )

        logger.info(f"Enqueued task {task.id} for conversation {conversation_id}")

        return task.id

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Get the status of a task.

        Args:
            task_id: Task ID to check

        Returns:
            Dict with task status info
        """
        app = self._get_app()
        result = app.AsyncResult(task_id)

        return {
            "task_id": task_id,
            "status": result.status,
            "ready": result.ready(),
            "successful": result.successful() if result.ready() else None,
            "result": result.result if result.ready() else None,
        }
