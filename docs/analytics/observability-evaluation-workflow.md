# Observability, Evaluation, And Analytics Workflow

This guide explains how to investigate one response and how to turn repeated
failures into evaluation or analytics work.

The project has several signal stores:

- Loki: searchable structured logs.
- Tempo: request traces.
- Prometheus: service and infrastructure metrics.
- Redpanda: durable inference lifecycle events.
- ClickHouse: production inference analytics from Redpanda events.
- Postgres: operational state, chat metadata, and offline evaluation results.
- Notebooks: deeper analysis over eval results and exported production patterns.

Use them as one loop, not as separate dashboards.

## Main Correlation Keys

- `request_id`: primary key for one chat completion request.
- `trace_id`: joins Loki logs to Tempo traces.
- `span_id`: points to a specific span inside one trace.
- `chat_session_id`: joins runtime events to persisted chat/session state.
- `celery_task_id`: joins gateway enqueue logs/events to worker execution.
- `user_id`: internal user identifier.
- `model`: requested or executed model/adapter name.

High-cardinality values such as `request_id`, `trace_id`, `user_id`,
`chat_session_id`, `celery_task_id`, KB, alias, collection, adapter, and model
are stored as JSON fields in logs/events, not as Loki labels.

## One-Request Diagnostic Path

Start with `request_id` whenever possible.

1. Search Loki logs for the request:

   ```logql
   {service=~"gateway|celery-worker"} | json | request_id="<REQUEST_ID>"
   ```

2. Copy `trace_id` from the log line or response metadata.

3. Open Tempo and search by trace id:

   ```traceql
   {trace:id = "<TRACE_ID>"}
   ```

4. Inspect the request-level event stream in ClickHouse:

   ```sql
   select
       occurred_at,
       service,
       event_type,
       request_id,
       trace_id,
       celery_task_id,
       chat_session_id,
       model,
       prompt_tokens,
       completion_tokens,
       total_tokens,
       error_type,
       raw
   from inference_events_raw
   where request_id = '<REQUEST_ID>'
   order by occurred_at;
   ```

5. If the request used RAG, inspect RAG logs and event metadata for KB, alias,
   collection, and coarse context counts. Source/chunk-level details belong to
   the later citation phase.

6. If the request corresponds to an offline eval sample, inspect Postgres eval
   data and the failure analysis notebook rather than relying only on runtime
   traces.

## When A Response Is Slow

Use Tempo first, then ClickHouse.

Tempo answers where time was spent inside one request:

- gateway authentication and validation;
- RAG retrieval;
- prompt construction;
- Celery enqueue and worker execution;
- vLLM tokenization and generation;
- persistence and streaming.

ClickHouse answers whether the case is isolated or common:

```sql
select
    model,
    event_type,
    count() as events,
    avg(total_tokens) as avg_total_tokens
from inference_events_raw
where occurred_at >= now() - interval 24 hour
group by model, event_type
order by events desc;
```

Prometheus/Grafana answers whether the slowdown matches service-level pressure:

- CPU/GPU saturation;
- queue growth;
- Redis/RabbitMQ pressure;
- vLLM latency;
- container restarts.

## When A Response Fails

Start from the lifecycle events:

```sql
select
    occurred_at,
    service,
    event_type,
    request_id,
    error_type,
    raw
from inference_events_raw
where request_id = '<REQUEST_ID>'
order by occurred_at;
```

Then use Loki to inspect the component that emitted the failure:

```logql
{service=~"gateway|celery-worker|embeddings|reranker"} | json | request_id="<REQUEST_ID>"
```

Use Tempo if the failure crossed services or the logs show a trace id.

## When A Response Is Ungrounded Or Low Quality

Runtime observability tells what happened. Evaluation tells whether the behavior
is acceptable and whether it repeats.

Use runtime signals for:

- which KB/alias/collection was selected;
- whether RAG retrieval ran;
- how many coarse context chunks/sources were selected;
- whether generation completed;
- token counts and finish reason.

Use evaluation data for:

- expected answer comparison;
- retrieval quality;
- metric verdicts;
- repeated failure categories;
- champion/challenger alias comparisons.

Phase 1 should keep this path simple:

1. Find the bad production request by `request_id`.
2. Inspect logs, trace, and ClickHouse events.
3. Decide whether the failure is retrieval, prompt construction, generation,
   infrastructure, or evaluation coverage.
4. If it is a repeatable quality issue, add it to the failure analysis notebook
   and consider turning it into a RAG eval dataset row in Phase 2.

## Which Store Answers Which Question

| Question | Use |
| --- | --- |
| What happened inside one component? | Loki logs |
| Where did one request spend time? | Tempo traces |
| Is the whole service unhealthy or overloaded? | Prometheus/Grafana |
| What lifecycle events happened for one request? | ClickHouse |
| Is this pattern common across production traffic? | ClickHouse |
| What did the offline eval runner measure? | Postgres eval tables |
| Why did a specific eval sample fail? | Failure analysis notebook |
| Which KB/alias was active? | Logs/events now; citations later |
| Did source grounding work? | Phase 2 citation metrics |
| Did users prefer the answer? | Phase 3 feedback tracking |

## Dashboard Categories

Dashboard JSON is developed separately. The workflow needs these dashboard
categories:

- service health: container status, restarts, CPU/GPU, memory, queues;
- request flow: gateway, worker, vLLM, embeddings, reranker latency;
- inference analytics: accepted/completed/failed requests, tokens, models,
  adapters, finish reasons;
- RAG analytics: RAG usage, hit/no-hit proxy, KB/alias distribution;
- evaluation: latest runs, metric trends, failed samples;
- feedback and A/B: later, after those events exist.

## Related Docs

- [Observability](observability.md)
- [Durable Inference Events](inference-events.md)
- [ClickHouse Analytics](clickhouse-analytics.md)
- [RAG Operations](../operations/rag-operations.md)
- [Improvement Plan](../planning/improvements.md)
