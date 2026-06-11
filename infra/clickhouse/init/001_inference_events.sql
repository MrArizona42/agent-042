CREATE DATABASE IF NOT EXISTS agent042_analytics;

CREATE TABLE IF NOT EXISTS agent042_analytics.inference_events_raw
(
    ingested_at DateTime64(3, 'UTC') DEFAULT now64(3),
    event_id String,
    schema_version UInt16,
    event_type LowCardinality(String),
    occurred_at DateTime64(3, 'UTC'),
    service LowCardinality(String),
    request_id String,
    trace_id String,
    span_id String,
    user_id String,
    chat_session_id String,
    celery_task_id String,
    conversation_id String,
    model String,
    finish_reason LowCardinality(String),
    error_type LowCardinality(String),
    prompt_tokens Nullable(UInt32),
    completion_tokens Nullable(UInt32),
    total_tokens Nullable(UInt32),
    rag_context_chunks_count Nullable(UInt32),
    rag_context_sources_count Nullable(UInt32),
    raw String
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(occurred_at)
ORDER BY (occurred_at, event_type, request_id, event_id)
TTL occurred_at + INTERVAL 180 DAY
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS agent042_analytics.kafka_inference_events_stream
(
    raw String
)
ENGINE = Kafka
SETTINGS
    kafka_broker_list = 'redpanda:9092',
    kafka_topic_list = 'inference.events.v1',
    kafka_group_name = 'clickhouse_inference_events_raw_v1',
    kafka_format = 'JSONAsString',
    kafka_num_consumers = 1,
    kafka_handle_error_mode = 'stream';

CREATE MATERIALIZED VIEW IF NOT EXISTS agent042_analytics.mv_inference_events_raw
TO agent042_analytics.inference_events_raw
AS
SELECT
    JSONExtractString(raw, 'event_id') AS event_id,
    toUInt16OrZero(JSONExtractRaw(raw, 'schema_version')) AS schema_version,
    JSONExtractString(raw, 'event_type') AS event_type,
    coalesce(
        parseDateTime64BestEffortOrNull(JSONExtractString(raw, 'occurred_at'), 3, 'UTC'),
        now64(3)
    ) AS occurred_at,
    JSONExtractString(raw, 'service') AS service,
    JSONExtractString(raw, 'request_id') AS request_id,
    JSONExtractString(raw, 'trace_id') AS trace_id,
    JSONExtractString(raw, 'span_id') AS span_id,
    JSONExtractString(raw, 'user_id') AS user_id,
    JSONExtractString(raw, 'chat_session_id') AS chat_session_id,
    JSONExtractString(raw, 'celery_task_id') AS celery_task_id,
    JSONExtractString(raw, 'conversation_id') AS conversation_id,
    JSONExtractString(raw, 'model') AS model,
    JSONExtractString(raw, 'payload', 'finish_reason') AS finish_reason,
    JSONExtractString(raw, 'payload', 'error_type') AS error_type,
    toUInt32OrNull(JSONExtractRaw(raw, 'payload', 'prompt_tokens')) AS prompt_tokens,
    toUInt32OrNull(JSONExtractRaw(raw, 'payload', 'completion_tokens')) AS completion_tokens,
    toUInt32OrNull(JSONExtractRaw(raw, 'payload', 'total_tokens')) AS total_tokens,
    toUInt32OrNull(JSONExtractRaw(raw, 'payload', 'context_chunks_count')) AS rag_context_chunks_count,
    toUInt32OrNull(JSONExtractRaw(raw, 'payload', 'context_sources_count')) AS rag_context_sources_count,
    raw
FROM agent042_analytics.kafka_inference_events_stream;
