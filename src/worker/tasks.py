"""Celery tasks for LLM inference."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
import redis

from shared.config import get_settings
from shared.vllm_payloads import (
    ResponseBudgetExceededError,
    apply_response_token_budget,
    canonicalize_assistant_content,
    extract_tokenize_payload,
)
from worker.celery_app import celery_app

logger = logging.getLogger(__name__)

# Redis event types
EVENT_THINKING_TOKEN = "thinking_token"
EVENT_ANSWER_TOKEN = "answer_token"
EVENT_DONE = "done"
EVENT_ERROR = "error"

_THINK_OPEN_TAG = "<think>"
_THINK_CLOSE_TAG = "</think>"
_REPETITIVE_CHAR_RUN_RE = re.compile(r"([^\s])\1{255,}")
_REPETITION_NOTICE = "\n\n[Response truncated because the model entered a repetitive output loop.]"


def _coerce_text_fragment(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_coerce_text_fragment(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            text = value.get(key)
            if isinstance(text, str):
                return text
    return ""


def _extract_explicit_thinking_delta(delta: dict[str, Any]) -> str:
    for key in ("reasoning_content", "reasoning", "thinking"):
        text = _coerce_text_fragment(delta.get(key))
        if text:
            return text
    return ""


def _detect_repetitive_answer_run(answer_content: str) -> str | None:
    match = _REPETITIVE_CHAR_RUN_RE.search(answer_content[-1024:])
    if match:
        return match.group(1)
    return None


def _usage_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _merge_usage(existing: dict[str, Any], incoming: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(existing)
    if isinstance(incoming, dict):
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = _usage_int(incoming.get(field))
            if value is not None:
                merged[field] = value

    prompt_tokens = _usage_int(merged.get("prompt_tokens"))
    completion_tokens = _usage_int(merged.get("completion_tokens"))
    total_tokens = _usage_int(merged.get("total_tokens"))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        merged["total_tokens"] = prompt_tokens + completion_tokens

    return merged


def _build_done_event(
    *,
    request_id: str,
    thinking_content: str,
    answer_content: str,
    finish_reason: str | None,
    task_id: str,
    usage: dict[str, Any],
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "thinking_content": thinking_content,
        "answer_content": answer_content,
        "content": canonicalize_assistant_content(thinking_content, answer_content),
        "finish_reason": finish_reason or "stop",
        "task_id": task_id,
        "usage": _merge_usage(usage, None),
    }


class _ThinkTagStreamParser:
    def __init__(self) -> None:
        self._buffer = ""
        self._in_thinking = False

    def feed(self, content: str) -> list[tuple[str, str]]:
        self._buffer += content
        return self._drain(final=False)

    def flush(self) -> list[tuple[str, str]]:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        while self._buffer:
            marker = _THINK_CLOSE_TAG if self._in_thinking else _THINK_OPEN_TAG
            event_type = EVENT_THINKING_TOKEN if self._in_thinking else EVENT_ANSWER_TOKEN
            marker_index = self._buffer.find(marker)

            if marker_index >= 0:
                if marker_index > 0:
                    events.append((event_type, self._buffer[:marker_index]))
                self._buffer = self._buffer[marker_index + len(marker) :]
                self._in_thinking = not self._in_thinking
                continue

            if final:
                events.append((event_type, self._buffer))
                self._buffer = ""
                break

            safe_length = max(0, len(self._buffer) - len(marker) + 1)
            if safe_length == 0:
                break
            events.append((event_type, self._buffer[:safe_length]))
            self._buffer = self._buffer[safe_length:]

        return [(event_type, fragment) for event_type, fragment in events if fragment]


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


def _maybe_truncate_repetitive_output(
    *,
    redis_client: redis.Redis,
    conversation_id: str,
    request_id: str,
    answer_content: str,
    task_id: str,
) -> tuple[bool, str]:
    repeated_char = _detect_repetitive_answer_run(answer_content)
    if repeated_char is None:
        return False, answer_content

    if not answer_content.endswith(_REPETITION_NOTICE):
        answer_content += _REPETITION_NOTICE
        publish_event(
            redis_client,
            conversation_id,
            EVENT_ANSWER_TOKEN,
            {"request_id": request_id, "content": _REPETITION_NOTICE},
        )

    logger.warning(
        "Task %s: Truncated repetitive output for conversation %s after repeated character %r",
        task_id,
        conversation_id,
        repeated_char,
    )
    return True, answer_content


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
    request_id: str,
    generation_payload: dict[str, Any],
    budget_meta: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate LLM response asynchronously.

    Streams tokens to Redis Pub/Sub and returns the full response.

    Args:
        conversation_id: Unique conversation identifier for the Redis channel
        request_id: Gateway-generated request identifier for the response lifecycle
        generation_payload: Chat completion payload without final max_tokens
        budget_meta: Exact-budget metadata for worker-side preflight

    Returns:
        Dict with full response content and metadata
    """
    shared = get_settings()
    redis_client = get_redis_client()

    logger.info(f"Task {self.request.id}: Starting generation for conversation {conversation_id}")

    thinking_content = ""
    answer_content = ""
    finish_reason = None
    usage: dict[str, Any] = {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }
    saw_explicit_thinking = False
    think_tag_parser = _ThinkTagStreamParser()
    repetition_guard_triggered = False

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
                include_usage=True,
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
                            usage = _merge_usage(usage, chunk_usage)
                            choices = chunk.get("choices", [])

                            if choices:
                                delta = choices[0].get("delta", {})
                                finish_reason = choices[0].get("finish_reason") or finish_reason
                                explicit_thinking = _extract_explicit_thinking_delta(delta)
                                content = _coerce_text_fragment(delta.get("content"))

                                if explicit_thinking:
                                    if not saw_explicit_thinking:
                                        for event_type, fragment in think_tag_parser.flush():
                                            if event_type == EVENT_THINKING_TOKEN:
                                                thinking_content += fragment
                                            else:
                                                answer_content += fragment
                                            publish_event(
                                                redis_client,
                                                conversation_id,
                                                event_type,
                                                {"request_id": request_id, "content": fragment},
                                            )
                                    saw_explicit_thinking = True
                                    thinking_content += explicit_thinking
                                    publish_event(
                                        redis_client,
                                        conversation_id,
                                        EVENT_THINKING_TOKEN,
                                        {"request_id": request_id, "content": explicit_thinking},
                                    )

                                if saw_explicit_thinking:
                                    if content:
                                        answer_content += content
                                        publish_event(
                                            redis_client,
                                            conversation_id,
                                            EVENT_ANSWER_TOKEN,
                                            {"request_id": request_id, "content": content},
                                        )
                                        repetition_guard_triggered, answer_content = (
                                            _maybe_truncate_repetitive_output(
                                                redis_client=redis_client,
                                                conversation_id=conversation_id,
                                                request_id=request_id,
                                                answer_content=answer_content,
                                                task_id=self.request.id,
                                            )
                                        )
                                        if repetition_guard_triggered:
                                            finish_reason = "stop"
                                            break
                                    continue

                                if content:
                                    for event_type, fragment in think_tag_parser.feed(content):
                                        if event_type == EVENT_THINKING_TOKEN:
                                            thinking_content += fragment
                                        else:
                                            answer_content += fragment
                                        publish_event(
                                            redis_client,
                                            conversation_id,
                                            event_type,
                                            {"request_id": request_id, "content": fragment},
                                        )
                                        if event_type == EVENT_ANSWER_TOKEN:
                                            repetition_guard_triggered, answer_content = (
                                                _maybe_truncate_repetitive_output(
                                                    redis_client=redis_client,
                                                    conversation_id=conversation_id,
                                                    request_id=request_id,
                                                    answer_content=answer_content,
                                                    task_id=self.request.id,
                                                )
                                            )
                                            if repetition_guard_triggered:
                                                finish_reason = "stop"
                                                break
                                    if repetition_guard_triggered:
                                        break

                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse SSE chunk: {data_str}")
                            continue

        if not saw_explicit_thinking and not repetition_guard_triggered:
            for event_type, fragment in think_tag_parser.flush():
                if event_type == EVENT_THINKING_TOKEN:
                    thinking_content += fragment
                else:
                    answer_content += fragment
                publish_event(
                    redis_client,
                    conversation_id,
                    event_type,
                    {"request_id": request_id, "content": fragment},
                )

        done_event = _build_done_event(
            request_id=request_id,
            thinking_content=thinking_content,
            answer_content=answer_content,
            finish_reason=finish_reason,
            task_id=self.request.id,
            usage=usage,
        )

        # Publish completion event
        publish_event(
            redis_client,
            conversation_id,
            EVENT_DONE,
            done_event,
        )

        logger.info(
            f"Task {self.request.id}: Completed generation for conversation {conversation_id}"
        )

        return {
            "conversation_id": conversation_id,
            **done_event,
        }

    except ResponseBudgetExceededError as e:
        logger.error(f"Task {self.request.id}: Exact budget rejected: {e}")

        publish_event(
            redis_client,
            conversation_id,
            EVENT_ERROR,
            {
                "request_id": request_id,
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
                "request_id": request_id,
                "error": str(e),
                "task_id": self.request.id,
            },
        )

        raise
