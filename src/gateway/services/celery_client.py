"""Celery client for enqueueing LLM tasks from the Gateway.

This module provides a client interface for sending tasks to
the Celery worker without importing the worker's Celery app directly.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from celery import Celery
from kombu.exceptions import OperationalError

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
                task_publish_retry=True,
                task_publish_retry_policy={
                    "max_retries": 3,
                    "interval_start": 0,
                    "interval_step": 1,
                    "interval_max": 2,
                },
            )
        return self._app

    def close(self) -> None:
        """Close Celery client resources."""
        if self._app is not None:
            self._app.close()
            self._app = None

    def enqueue_generate_response(
        self,
        conversation_id: str,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Enqueue a generate_response task.

        Args:
            conversation_id: Unique conversation identifier
            messages: List of chat messages in OpenAI format
            model: Model to use (optional)
            temperature: Sampling temperature (optional)
            top_p: Top-p sampling (optional)
            max_tokens: Maximum tokens (optional)

        Returns:
            Task ID
        """
        app = self._get_app()

        try:
            # Send task by name (worker.tasks.generate_response)
            task = app.send_task(
                "worker.tasks.generate_response",
                kwargs={
                    "conversation_id": conversation_id,
                    "messages": messages,
                    "model": model,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                },
            )
        except (OperationalError, ConnectionError, OSError):
            logger.warning("Celery broker connection failed, recreating client and retrying once")
            self.close()
            task = self._get_app().send_task(
                "worker.tasks.generate_response",
                kwargs={
                    "conversation_id": conversation_id,
                    "messages": messages,
                    "model": model,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
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


# Global client instance (lazy initialization)
_celery_client: CeleryClient | None = None


def get_celery_client() -> CeleryClient:
    """Get or create the global Celery client.

    Uses CELERY_BROKER_URL environment variable for configuration.

    Raises:
        RuntimeError: If CELERY_BROKER_URL is not set.
    """
    global _celery_client
    if _celery_client is None:
        broker_url = os.environ.get("CELERY_BROKER_URL")
        if not broker_url:
            raise RuntimeError(
                "CELERY_BROKER_URL environment variable is required but not set. "
                "Example: amqp://user:password@rabbitmq:5672//"
            )
        _celery_client = CeleryClient(broker_url)
    return _celery_client


def close_celery_client() -> None:
    """Close and reset global Celery client."""
    global _celery_client
    if _celery_client is not None:
        _celery_client.close()
        _celery_client = None
