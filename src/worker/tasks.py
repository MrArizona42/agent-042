"""Celery tasks for LLM inference."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import redis

from worker.celery_app import celery_app
from worker.config import get_worker_settings

logger = logging.getLogger(__name__)

# Redis event types
EVENT_TOKEN = "token"
EVENT_DONE = "done"
EVENT_ERROR = "error"


def get_redis_client() -> redis.Redis:
    """Create a Redis client for publishing events."""
    settings = get_worker_settings()
    return redis.from_url(settings.redis_url, decode_responses=True)


def publish_event(
    redis_client: redis.Redis,
    conversation_id: str,
    event_type: str,
    data: dict[str, Any],
) -> None:
    """Publish an event to the Redis channel for a conversation."""
    channel = f"tokens:{conversation_id}"
    message = json.dumps({"type": event_type, **data})
    redis_client.publish(channel, message)


@celery_app.task(
    bind=True,
    autoretry_for=(httpx.HTTPStatusError, httpx.ConnectError),
    retry_backoff=True,
    retry_backoff_max=60,
    max_retries=3,
)
def generate_response(
    self,
    conversation_id: str,
    messages: list[dict[str, Any]],
    model: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """
    Generate LLM response asynchronously.

    Streams tokens to Redis Pub/Sub and returns the full response.

    Args:
        conversation_id: Unique conversation identifier for the Redis channel
        messages: List of chat messages in OpenAI format
        model: Model to use (defaults to worker config)
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate

    Returns:
        Dict with full response content and metadata
    """
    settings = get_worker_settings()
    redis_client = get_redis_client()

    # Build request payload
    payload: dict[str, Any] = {
        "model": model or settings.vllm_model,
        "messages": messages,
        "stream": True,  # Always stream from vLLM
    }

    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    logger.info(f"Task {self.request.id}: Starting generation for conversation {conversation_id}")

    full_content = ""
    finish_reason = None

    try:
        with httpx.Client(timeout=None) as client:
            with client.stream(
                "POST",
                f"{settings.vllm_base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    # SSE format: "data: {...}"
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix

                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])

                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                finish_reason = choices[0].get("finish_reason")

                                if content:
                                    full_content += content
                                    # Publish token to Redis
                                    publish_event(
                                        redis_client,
                                        conversation_id,
                                        EVENT_TOKEN,
                                        {"content": content},
                                    )

                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse SSE chunk: {data_str}")
                            continue

        # Publish completion event
        publish_event(
            redis_client,
            conversation_id,
            EVENT_DONE,
            {
                "content": full_content,
                "finish_reason": finish_reason or "stop",
                "task_id": self.request.id,
            },
        )

        logger.info(
            f"Task {self.request.id}: Completed generation for conversation {conversation_id}"
        )

        return {
            "conversation_id": conversation_id,
            "content": full_content,
            "finish_reason": finish_reason or "stop",
            "task_id": self.request.id,
        }

    except Exception as e:
        logger.error(f"Task {self.request.id}: Error during generation: {e}")

        # Publish error event
        publish_event(
            redis_client,
            conversation_id,
            EVENT_ERROR,
            {
                "error": str(e),
                "task_id": self.request.id,
            },
        )

        raise
