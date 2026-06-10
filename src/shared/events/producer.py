"""Kafka-compatible producer for durable inference events."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from shared.config import Settings
from shared.events.inference import InferenceEvent, InferenceEventType

logger = logging.getLogger(__name__)


class _KafkaProducer(Protocol):
    def produce(self, topic: str, *, key: str | None, value: bytes, on_delivery: Any) -> None: ...

    def poll(self, timeout: float) -> int: ...

    def flush(self, timeout: float) -> int: ...


class InferenceEventProducer:
    """Publish inference events without making Kafka a hard request dependency."""

    def __init__(
        self,
        *,
        service: str,
        topic: str,
        bootstrap_servers: str | None,
        producer: _KafkaProducer | None = None,
    ) -> None:
        self._service = service
        self._topic = topic
        self._bootstrap_servers = bootstrap_servers
        self._producer = producer
        self._initialization_attempted = producer is not None

    @property
    def configured(self) -> bool:
        return bool(self._bootstrap_servers or self._producer)

    def publish(
        self,
        event_type: InferenceEventType,
        *,
        request_id: str | None = None,
        user_id: str | None = None,
        chat_session_id: str | None = None,
        celery_task_id: str | None = None,
        conversation_id: str | None = None,
        model: str | None = None,
        payload: dict[str, Any] | None = None,
        key: str | None = None,
    ) -> None:
        if not self.configured:
            return

        try:
            event = InferenceEvent.build(
                event_type=event_type,
                service=self._service,
                request_id=request_id,
                user_id=user_id,
                chat_session_id=chat_session_id,
                celery_task_id=celery_task_id,
                conversation_id=conversation_id,
                model=model,
                payload=payload,
            )
        except Exception:
            logger.warning(
                "Failed to build inference event",
                extra={"event": "inference_event.build_failed", "event_type": event_type},
                exc_info=True,
            )
            return

        producer = self._get_producer()
        if producer is None:
            return

        try:
            producer.produce(
                self._topic,
                key=key or event.request_id or event.event_id,
                value=event.to_json_bytes(),
                on_delivery=self._delivery_callback,
            )
            producer.poll(0)
        except Exception:
            logger.warning(
                "Failed to publish inference event",
                extra={"event": "inference_event.publish_failed", "event_type": event_type},
                exc_info=True,
            )

    def close(self) -> None:
        if self._producer is None:
            return
        try:
            self._producer.flush(2.0)
        except Exception:
            logger.warning("Failed to flush inference event producer", exc_info=True)

    def _get_producer(self) -> _KafkaProducer | None:
        if self._producer is not None:
            return self._producer
        if self._initialization_attempted or not self._bootstrap_servers:
            return None

        self._initialization_attempted = True
        try:
            from confluent_kafka import Producer
        except ImportError:
            logger.warning("Kafka inference events requested but confluent-kafka is unavailable")
            return None

        try:
            self._producer = Producer(
                {
                    "bootstrap.servers": self._bootstrap_servers,
                    "client.id": f"agent-042-{self._service}",
                    "message.timeout.ms": 3000,
                    "socket.timeout.ms": 3000,
                    "request.timeout.ms": 3000,
                }
            )
            logger.info(
                "Inference event producer initialized",
                extra={"event": "inference_event.producer.initialized", "topic": self._topic},
            )
        except Exception:
            logger.warning("Failed to initialize inference event producer", exc_info=True)
            return None
        return self._producer

    @staticmethod
    def _delivery_callback(error: Any, message: Any) -> None:
        if error is None:
            return
        logger.warning(
            "Inference event delivery failed",
            extra={
                "event": "inference_event.delivery_failed",
                "topic": message.topic() if message is not None else None,
                "error_type": type(error).__name__,
            },
        )


def create_inference_event_producer(
    *,
    service: str,
    settings: Settings,
) -> InferenceEventProducer:
    return InferenceEventProducer(
        service=service,
        topic=settings.platform.inference_events_topic,
        bootstrap_servers=settings.platform.kafka_bootstrap_servers,
    )
