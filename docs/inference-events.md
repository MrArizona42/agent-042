# Durable Inference Events

Redpanda provides the Kafka-compatible event stream for inference lifecycle
analytics. The first topic is:

```text
inference.events.v1
```

The stream is enabled by default in the server Compose deployment through:

```text
PLATFORM__KAFKA_BOOTSTRAP_SERVERS=redpanda:9092
PLATFORM__INFERENCE_EVENTS_TOPIC=inference.events.v1
```

There is no separate `EVENTS_ENABLED` flag. If Redpanda is unavailable,
publishing failures are logged as structured warnings and the chat flow
continues.

## Event Types

Gateway publishes:

- `chat.request.accepted`
- `chat.request.rejected`
- `rag.context.selected`
- `celery.task.enqueued`
- `chat.response.completed`
- `chat.persistence.completed`

Worker publishes:

- `worker.generation.started`
- `worker.vllm.tokenized`
- `worker.generation.completed`
- `worker.generation.failed`

## Schema

All events use the shared schema in `src/shared/events/inference.py`.

Common fields:

- `event_id`
- `schema_version`
- `event_type`
- `occurred_at`
- `service`
- `request_id`
- `trace_id`
- `span_id`
- `user_id`
- `chat_session_id`
- `celery_task_id`
- `conversation_id`
- `model`
- `payload`

Payloads intentionally stay metadata-only. Full prompts, full responses,
messages, generated content, cookies, access tokens, API keys, and OAuth
payloads are rejected by the event schema. Token counts are allowed.

RAG events currently include only coarse metadata:

- `context_chunks_count`
- `context_sources_count`
- `sources`: knowledge-base and alias names

Chunk-level source details belong to the later source-citation phase.

## Inspecting Events

Redpanda Console is included in Compose and is bound to `127.0.0.1`.
Default from `.env.example`:

```text
http://localhost:8081
```

Open the `inference.events.v1` topic to inspect recent lifecycle events.

From the server shell, a quick broker check can also use Redpanda's `rpk`:

```bash
docker compose --env-file .env -f infra/compose/docker-compose.yaml exec redpanda \
  rpk topic consume inference.events.v1 --num 5
```

## Relationship To Observability

Logs and traces answer "what happened during this request right now?" Durable
events answer "what happened across many requests over time, and can we replay
that metadata into analytics storage?"

The key joins are:

- `request_id`: Gateway logs, traces, Kafka events, prompt preview.
- `trace_id`: Loki logs and Tempo traces.
- `chat_session_id`: Kafka events and PostgreSQL chat persistence.
- `celery_task_id`: Gateway enqueue event, worker logs, worker events.

ClickHouse ingestion should consume this topic later instead of scraping logs.
