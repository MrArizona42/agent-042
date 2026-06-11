# ClickHouse Analytics

ClickHouse consumes durable inference events directly from Redpanda with a
Kafka Engine table and stores them in a MergeTree archive.

## Ingestion Flow

```text
Redpanda topic: inference.events.v1
-> kafka_inference_events_stream     -- Kafka Engine stream adapter
-> mv_inference_events_raw           -- Materialized View
-> inference_events_raw              -- durable MergeTree archive
```

The Kafka Engine table is not long-term storage. It is a ClickHouse consumer
over the Redpanda topic. Query `inference_events_raw` for analytics.

## Tables

Database:

```text
agent042_analytics
```

Tables:

- `kafka_inference_events_stream`: Kafka Engine stream adapter.
- `mv_inference_events_raw`: materialized view that parses event metadata.
- `inference_events_raw`: durable event archive with raw JSON and common
  analytics columns.

The raw table keeps full event JSON in `raw` and extracts common fields:

- event identifiers and timestamps;
- `event_type`;
- service and correlation ids;
- token counts;
- finish reason and error type;
- coarse RAG context counts.

Prompt text, generated response text, access tokens, API keys, cookies, and
OAuth payloads should not appear here because the producer-side schema rejects
those fields before events reach Redpanda.

## First Queries

Recent events:

```sql
select
    occurred_at,
    service,
    event_type,
    request_id,
    model,
    prompt_tokens,
    completion_tokens,
    error_type
from inference_events_raw
order by occurred_at desc
limit 50;
```

Requests per hour:

```sql
select
    toStartOfHour(occurred_at) as hour,
    countIf(event_type = 'chat.request.accepted') as accepted,
    countIf(event_type = 'chat.response.completed') as completed,
    countIf(event_type = 'worker.generation.failed') as failed
from inference_events_raw
where occurred_at >= now() - interval 24 hour
group by hour
order by hour;
```

Token usage by model:

```sql
select
    model,
    countIf(event_type = 'worker.generation.completed') as generations,
    avg(prompt_tokens) as avg_prompt_tokens,
    avg(completion_tokens) as avg_completion_tokens,
    sum(total_tokens) as total_tokens
from inference_events_raw
where event_type = 'worker.generation.completed'
group by model
order by generations desc;
```

Requests accepted but not completed:

```sql
with
    accepted as (
        select request_id, min(occurred_at) as accepted_at
        from inference_events_raw
        where event_type = 'chat.request.accepted'
        group by request_id
    ),
    completed as (
        select distinct request_id
        from inference_events_raw
        where event_type = 'chat.response.completed'
    )
select accepted.request_id, accepted.accepted_at
from accepted
left anti join completed using request_id
order by accepted_at desc
limit 50;
```

RAG usage:

```sql
select
    toStartOfHour(occurred_at) as hour,
    count() as rag_requests,
    avg(rag_context_chunks_count) as avg_context_chunks,
    avg(rag_context_sources_count) as avg_context_sources
from inference_events_raw
where event_type = 'rag.context.selected'
group by hour
order by hour;
```

## Operations

Check the raw table:

```bash
docker compose --env-file .env -f infra/compose/docker-compose.yaml exec clickhouse \
  clickhouse-client --database agent042_analytics \
  --query "select event_type, count() from inference_events_raw group by event_type"
```

Check the Kafka consumer table exists:

```bash
docker compose --env-file .env -f infra/compose/docker-compose.yaml exec clickhouse \
  clickhouse-client --database agent042_analytics \
  --query "show tables"
```

Grafana provisions the ClickHouse datasource as `ClickHouse`.
