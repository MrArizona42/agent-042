"""Celery tasks for LLM inference."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import redis

from shared.config import get_settings
from shared.vllm_payloads import (
    ResponseBudgetExceededError,
    apply_response_token_budget,
    extract_tokenize_payload,
)
from worker.celery_app import celery_app

logger = logging.getLogger(__name__)

# Redis event types
EVENT_TOKEN = "token"
EVENT_DONE = "done"
EVENT_ERROR = "error"


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def get_redis_client() -> redis.Redis:
    """Create a Redis client for publishing events."""
    settings = get_settings()
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
    generation_payload: dict[str, Any],
    budget_meta: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate LLM response asynchronously.

    Streams tokens to Redis Pub/Sub and returns the full response.

    Args:
        conversation_id: Unique conversation identifier for the Redis channel
        generation_payload: Chat completion payload without final max_tokens
        budget_meta: Exact-budget metadata for worker-side preflight

    Returns:
        Dict with full response content and metadata
    """
    shared = get_settings()
    redis_client = get_redis_client()

    logger.info(f"Task {self.request.id}: Starting generation for conversation {conversation_id}")

    full_content = ""
    finish_reason = None
    usage: dict[str, Any] = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }

    try:
        headers = _headers(shared.api_key)
        with httpx.Client(timeout=None) as client:
            tokenize_payload = extract_tokenize_payload(generation_payload)
            tokenize_response = client.post(
                f"{shared.vllm_base_url}/tokenize",
                json=tokenize_payload,
                headers=headers,
            )
            tokenize_response.raise_for_status()
            usage["prompt_tokens"] = int(tokenize_response.json()["count"])

            payload, final_max_tokens = apply_response_token_budget(
                generation_payload,
                prompt_tokens=usage["prompt_tokens"],
                budget_meta=budget_meta,
                stream=True,
            )
            logger.info(
                "Task %s: exact prompt_tokens=%s final_max_tokens=%s",
                self.request.id,
                usage["prompt_tokens"],
                final_max_tokens,
            )

            with client.stream(
                "POST",
                f"{shared.vllm_base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
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
                            chunk_usage = chunk.get("usage")
                            if isinstance(chunk_usage, dict):
                                usage = {
                                    "prompt_tokens": chunk_usage.get(
                                        "prompt_tokens", usage.get("prompt_tokens")
                                    ),
                                    "completion_tokens": chunk_usage.get("completion_tokens"),
                                    "total_tokens": chunk_usage.get("total_tokens"),
                                }
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
                "usage": usage,
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
            "usage": usage,
        }

    except ResponseBudgetExceededError as e:
        logger.error(f"Task {self.request.id}: Exact budget rejected: {e}")

        publish_event(
            redis_client,
            conversation_id,
            EVENT_ERROR,
            {
                "error": str(e),
                "error_type": "budget_exceeded",
                "task_id": self.request.id,
            },
        )

        raise

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
