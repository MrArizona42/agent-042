"""Schema for durable inference lifecycle events."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.telemetry import current_trace_context

InferenceEventType = Literal[
    "chat.request.accepted",
    "chat.request.rejected",
    "rag.context.selected",
    "celery.task.enqueued",
    "worker.generation.started",
    "worker.vllm.tokenized",
    "worker.generation.completed",
    "worker.generation.failed",
    "chat.response.completed",
    "chat.persistence.completed",
]

FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "access_token",
        "answer_content",
        "api_key",
        "content",
        "cookie",
        "cookies",
        "messages",
        "oauth_payload",
        "prompt",
        "prompt_messages",
        "response",
        "token",
        "thinking_content",
    }
)


class InferenceEvent(BaseModel):
    """Replayable event for inference analytics and downstream consumers."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    schema_version: int = 1
    event_type: InferenceEventType
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    service: str
    request_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    user_id: str | None = None
    chat_session_id: str | None = None
    celery_task_id: str | None = None
    conversation_id: str | None = None
    model: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("payload")
    @classmethod
    def _reject_sensitive_payload_keys(cls, value: dict[str, Any]) -> dict[str, Any]:
        _check_payload_keys(value)
        return value

    @classmethod
    def build(
        cls,
        *,
        event_type: InferenceEventType,
        service: str,
        request_id: str | None = None,
        user_id: str | None = None,
        chat_session_id: str | None = None,
        celery_task_id: str | None = None,
        conversation_id: str | None = None,
        model: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> "InferenceEvent":
        trace_context = current_trace_context()
        return cls(
            event_type=event_type,
            service=service,
            request_id=request_id,
            trace_id=trace_context.get("trace_id"),
            span_id=trace_context.get("span_id"),
            user_id=user_id,
            chat_session_id=chat_session_id,
            celery_task_id=celery_task_id,
            conversation_id=conversation_id,
            model=model,
            payload=payload or {},
        )

    def to_json_bytes(self) -> bytes:
        return self.model_dump_json(exclude_none=True).encode("utf-8")


def _check_payload_keys(value: Any, *, path: str = "payload") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_PAYLOAD_KEYS:
                raise ValueError(f"{path}.{key} is not allowed in inference event payloads")
            _check_payload_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_payload_keys(child, path=f"{path}[{index}]")
