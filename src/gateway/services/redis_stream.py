"""Redis Pub/Sub service for token streaming.

This module provides async subscription to Redis channels for
receiving tokens streamed from Celery workers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# Event types (must match worker/tasks.py)
EVENT_THINKING_TOKEN = "thinking_token"
EVENT_ANSWER_TOKEN = "answer_token"
EVENT_DONE = "done"
EVENT_ERROR = "error"


def _monotonic_time() -> float:
    """Return the event loop monotonic clock for timeout bookkeeping."""
    return asyncio.get_running_loop().time()


class RedisStreamService:
    """Service for subscribing to Redis Pub/Sub channels for token streaming."""

    def __init__(self, redis_url: str):
        """Initialize the Redis stream service.

        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379/0)
        """
        self._redis_url = redis_url
        self._redis: aioredis.Redis | None = None

    async def _get_redis(self) -> aioredis.Redis:
        """Get or create Redis connection."""
        if self._redis is None:
            self._redis = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None

    async def store_prompt_preview(
        self,
        request_id: str,
        preview: dict[str, Any],
        *,
        ttl_seconds: int = 900,
    ) -> None:
        redis = await self._get_redis()
        await redis.setex(
            f"prompt_preview:{request_id}",
            ttl_seconds,
            json.dumps(preview),
        )

    async def get_prompt_preview(self, request_id: str) -> dict[str, Any] | None:
        redis = await self._get_redis()
        raw = await redis.get(f"prompt_preview:{request_id}")
        if raw is None:
            return None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to decode prompt preview for request_id=%s", request_id)
            return None
        return value if isinstance(value, dict) else None

    async def subscribe(
        self,
        conversation_id: str,
        timeout: float = 300.0,
    ) -> AsyncIterator[dict[str, Any]]:
        """Subscribe to token events for a conversation.

        Yields events until a 'done' or 'error' event is received,
        or the idle timeout is reached.

        Args:
            conversation_id: Conversation ID to subscribe to
            timeout: Maximum idle time to wait for the next event in seconds

        Yields:
            Event dictionaries with 'type' and additional data
        """
        channel_name = f"tokens:{conversation_id}"
        redis = await self._get_redis()
        pubsub = redis.pubsub()

        try:
            await pubsub.subscribe(channel_name)
            logger.info(f"Subscribed to channel: {channel_name}")

            last_event_at = _monotonic_time()

            while True:
                # Treat timeout as idle time between events, not total stream duration.
                idle_elapsed = _monotonic_time() - last_event_at
                if idle_elapsed >= timeout:
                    logger.warning(f"Idle timeout waiting for events on {channel_name}")
                    yield {"type": EVENT_ERROR, "error": "Timeout waiting for response"}
                    break

                # Get message with timeout
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=min(1.0, max(0.05, timeout - idle_elapsed)),
                )

                if message is None:
                    # No message, continue polling
                    await asyncio.sleep(0.01)
                    continue

                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        event_type = data.get("type")
                        last_event_at = _monotonic_time()

                        yield data

                        # Stop on done or error
                        if event_type in (EVENT_DONE, EVENT_ERROR):
                            logger.info(f"Received {event_type} event, closing subscription")
                            break

                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse message: {message['data']}")
                        continue

        finally:
            await pubsub.unsubscribe(channel_name)
            await pubsub.close()
            logger.info(f"Unsubscribed from channel: {channel_name}")
