from __future__ import annotations

import json

import pytest

from clients.events import InferenceEvent, InferenceEventProducer


class FakeKafkaProducer:
    def __init__(self) -> None:
        self.records: list[tuple[str, str | None, bytes]] = []
        self.poll_calls = 0
        self.flush_calls = 0

    def produce(self, topic: str, *, key: str | None, value: bytes, on_delivery) -> None:
        self.records.append((topic, key, value))
        on_delivery(None, None)

    def poll(self, timeout: float) -> int:
        self.poll_calls += 1
        return 0

    def flush(self, timeout: float) -> int:
        self.flush_calls += 1
        return 0


def test_inference_event_schema_excludes_empty_fields_and_keeps_raw_user_id() -> None:
    event = InferenceEvent.build(
        event_type="chat.request.accepted",
        service="gateway",
        request_id="req-1",
        user_id="raw-user-id",
        chat_session_id="session-1",
        model="qwen",
        payload={"message_count": 2, "rag_sources_count": 0},
    )

    payload = json.loads(event.to_json_bytes())

    assert payload["schema_version"] == 1
    assert payload["event_type"] == "chat.request.accepted"
    assert payload["service"] == "gateway"
    assert payload["request_id"] == "req-1"
    assert payload["user_id"] == "raw-user-id"
    assert payload["payload"] == {"message_count": 2, "rag_sources_count": 0}
    assert "environment" not in payload


def test_inference_event_rejects_prompt_or_response_payloads() -> None:
    with pytest.raises(ValueError):
        InferenceEvent.build(
            event_type="chat.response.completed",
            service="gateway",
            payload={"content": "model output"},
        )

    with pytest.raises(ValueError):
        InferenceEvent.build(
            event_type="chat.request.accepted",
            service="gateway",
            payload={"nested": {"messages": ["user prompt"]}},
        )


def test_inference_event_producer_publishes_json_to_configured_topic() -> None:
    fake = FakeKafkaProducer()
    producer = InferenceEventProducer(
        service="gateway",
        topic="inference.events.v1",
        bootstrap_servers=None,
        producer=fake,
    )

    producer.publish(
        "celery.task.enqueued",
        request_id="req-1",
        celery_task_id="task-1",
        conversation_id="conversation-1",
        model="qwen",
    )

    assert fake.poll_calls == 1
    assert len(fake.records) == 1
    topic, key, value = fake.records[0]
    payload = json.loads(value)
    assert topic == "inference.events.v1"
    assert key == "req-1"
    assert payload["event_type"] == "celery.task.enqueued"
    assert payload["celery_task_id"] == "task-1"
    assert payload["conversation_id"] == "conversation-1"


def test_inference_event_producer_without_bootstrap_is_noop() -> None:
    producer = InferenceEventProducer(
        service="gateway",
        topic="inference.events.v1",
        bootstrap_servers=None,
    )

    producer.publish("chat.request.accepted", request_id="req-1")
