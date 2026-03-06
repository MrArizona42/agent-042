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
EVENT_TOKEN = "token"
EVENT_DONE = "done"
EVENT_ERROR = "error"


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

    async def subscribe(
        self,
        conversation_id: str,
        timeout: float = 300.0,
    ) -> AsyncIterator[dict[str, Any]]:
        """Subscribe to token events for a conversation.

        Yields events until a 'done' or 'error' event is received,
        or the timeout is reached.

        Args:
            conversation_id: Conversation ID to subscribe to
            timeout: Maximum time to wait for events in seconds

        Yields:
            Event dictionaries with 'type' and additional data
        """
        channel_name = f"tokens:{conversation_id}"
        redis = await self._get_redis()
        pubsub = redis.pubsub()

        try:
            await pubsub.subscribe(channel_name)
            logger.info(f"Subscribed to channel: {channel_name}")

            start_time = asyncio.get_event_loop().time()

            while True:
                # Check timeout
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    logger.warning(f"Timeout waiting for events on {channel_name}")
                    yield {"type": EVENT_ERROR, "error": "Timeout waiting for response"}
                    break

                # Get message with timeout
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )

                if message is None:
                    # No message, continue polling
                    await asyncio.sleep(0.01)
                    continue

                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        event_type = data.get("type")

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

    async def subscribe_sse(
        self,
        conversation_id: str,
        timeout: float = 300.0,
    ) -> AsyncIterator[bytes]:
        """Subscribe and yield SSE-formatted events.

        This is a convenience method that formats events as Server-Sent Events
        for direct streaming to HTTP clients.

        Args:
            conversation_id: Conversation ID to subscribe to
            timeout: Maximum time to wait for events in seconds

        Yields:
            SSE-formatted bytes
        """
        full_content = ""

        async for event in self.subscribe(conversation_id, timeout):
            event_type = event.get("type")

            if event_type == EVENT_TOKEN:
                content = event.get("content", "")
                full_content += content
                # Format as OpenAI-compatible SSE chunk
                chunk = {
                    "id": f"chatcmpl-{conversation_id}",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode()

            elif event_type == EVENT_DONE:
                # Final chunk with finish_reason
                chunk = {
                    "id": f"chatcmpl-{conversation_id}",
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": event.get("finish_reason", "stop"),
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode()
                yield b"data: [DONE]\n\n"

            elif event_type == EVENT_ERROR:
                error_msg = event.get("error", "Unknown error")
                error_chunk = {
                    "error": {
                        "message": error_msg,
                        "type": "server_error",
                    }
                }
                yield f"data: {json.dumps(error_chunk)}\n\n".encode()
