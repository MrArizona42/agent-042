# Observability Workflow

This stack is designed for the single-node server deployment. Application logs,
traces, metrics, and persisted metadata are all available from Grafana.

## Backends

- Prometheus stores metrics from Gateway, vLLM, RabbitMQ, node-exporter, and
  dcgm-exporter.
- Loki stores Docker logs collected by Grafana Alloy from the Docker socket.
- Tempo stores OpenTelemetry traces received through `otel-collector`.
- PostgreSQL stores durable application records such as chat sessions, chat
  messages, evaluation runs, and evaluation samples.
- Redpanda stores replayable inference lifecycle events for analytics and later
  ClickHouse ingestion.
- ClickHouse stores production inference analytics ingested from Redpanda
  events.

Grafana provisions the core datasources from
`infra/grafana/provisioning/datasources/datasources.yml`.

## Correlation Fields

Python services emit structured JSON logs through `src/clients/observability/logging.py`.
The most useful fields are:

- `request_id`: canonical gateway-generated request identifier.
- `trace_id`: OpenTelemetry trace identifier, emitted when a log happens inside
  an active span.
- `span_id`: OpenTelemetry span identifier.
- `user_id`: raw internal user id.
- `chat_session_id`: chat session identifier.
- `celery_task_id`: worker task identifier.
- `conversation_id`: Redis Pub/Sub stream identifier for async generation.
- `event`: stable lifecycle event name.

Do not rely on Loki labels for high-cardinality analysis. Labels stay small:
`service`, `container`, and `level`. Query high-cardinality values as JSON
fields.

## Main Request Flow

For a chat completion request:

1. Gateway receives `/v1/chat/completions` and creates `request_id`.
2. Gateway builds task routing, optional RAG context, and the budgeted prompt.
3. Gateway stores prompt preview in Redis under the same `request_id`.
4. Gateway enqueues the Celery task and logs `celery_task_id`.
5. Worker tokenizes the prompt through vLLM, applies the exact response budget,
   and streams generation chunks through Redis.
6. Gateway streams the response to the UI and persists the exchange when a
   `chat_session_id` and `user_id` are present.

The trace names to start with in Tempo are:

- `POST /v1/chat/completions`
- `gateway.prepare_chat_request`
- `gateway.task_routing`
- `rag.auto_select_sources`
- `rag.retrieve_context`
- `gateway.prompt_build`
- `celery.enqueue_generate_response`
- `worker.vllm_tokenize`
- `worker.vllm_generate`
- `gateway.persist_exchange`

## Grafana Explore

Useful Loki queries:

```logql
{service="gateway"} | json | request_id="..."
{service="celery-worker"} | json | celery_task_id="..."
{service=~"gateway|celery-worker"} | json | trace_id="..."
{service="gateway", level="ERROR"} | json
```

Useful Tempo flow:

1. Open Grafana Explore.
2. Select the `Tempo` datasource.
3. Search by service name, span name, or paste a `trace_id` from Loki.
4. Use the configured trace-to-logs link to jump back to Loki.

Useful PostgreSQL checks:

```sql
select id, user_id, title, created_at, updated_at
from chat_sessions
order by updated_at desc
limit 20;

select session_id, role, prompt_tokens, completion_tokens, created_at
from chat_messages
where session_id = '<chat_session_uuid>'
order by created_at;
```

## Operator Notes

- The deployed Python services export traces only when
  `OTEL_EXPORTER_OTLP_ENDPOINT` is set by Compose.
- `OTEL_TRACES_SAMPLER_ARG=1.0` keeps all traces initially. Lower it later if
  trace volume becomes noisy.
- Logs intentionally do not include full prompts, full responses, access tokens,
  API keys, cookies, or OAuth payloads.
- Prompt preview remains a short-lived Redis debugging aid, not a Loki log.
- Durable inference event details are documented in
  `docs/analytics/inference-events.md`.
- ClickHouse analytics ingestion and first queries are documented in
  `docs/analytics/clickhouse-analytics.md`.
