# System Design: Asynchronous LLM Chat System with Streaming and Durable Storage

> **Status:** Target Architecture
> **Current state:** Synchronous flow (Streamlit → FastAPI → vLLM)
> **Target state:** Asynchronous, fault-tolerant system with durable history

---

## 1. Goals & Non-Goals

### Goals

- **Low-latency, token-level streaming** to UI
- **Asynchronous, fault-tolerant** LLM inference
- **Durable long-term storage** of chat history
- **Clear separation of concerns** between transport, execution, and persistence
- **Production-like architecture** suitable for research and extension

### Non-Goals

- Exactly-once execution semantics
- Multi-region deployment
- Online model training

---

## 2. High-Level Architecture

```
┌─────────────────────┐
│   Streamlit (UI)    │
└─────────┬───────────┘
          │ HTTP / SSE
          ▼
┌─────────────────────┐
│  FastAPI (Gateway)  │◄────────┐
│  - Orchestration    │         │
│  - SSE streaming    │         │ Redis Pub/Sub
└─────────┬───────────┘         │ (token events)
          │ enqueue job         │
          ▼                     │
┌─────────────────────┐         │
│  RabbitMQ (Broker)  │         │
└─────────┬───────────┘         │
          │                     │
          ▼                     │
┌─────────────────────┐         │
│   Celery Workers    │─────────┘
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   vLLM (Inference)  │
└─────────────────────┘
```

**Side channels:**

| Channel         | Purpose                          |
|-----------------|----------------------------------|
| Redis Pub/Sub   | Real-time token streaming        |
| ClickHouse      | Durable chat history & analytics |

---

## 3. Component Responsibilities

### Streamlit (UI)

- Collect user input
- Open SSE connection for streaming tokens
- Render tokens incrementally
- Display completed assistant messages

> Stateless with respect to conversation storage.

### FastAPI (Control Plane)

- Request validation and authentication
- Conversation/session management
- Persist chat events to ClickHouse
- Enqueue inference jobs via Celery
- Maintain SSE connections to clients
- Bridge Redis Pub/Sub events to UI

> FastAPI is the **coordination layer**, not the execution layer.

### RabbitMQ (Message Broker)

- Reliable delivery of inference jobs
- Backpressure handling
- Decoupling API responsiveness from inference latency

> Used only for **job transport**, not for streaming or storage.

### Celery (Task Execution)

- Defines task semantics (retry, timeout, failure)
- Executes LLM inference jobs asynchronously
- Handles transient failures (GPU OOM, timeouts)
- Publishes token events during execution

> Celery workers are **stateless** between tasks.

### vLLM (Inference Engine)

- High-throughput, GPU-backed text generation
- Produces tokens incrementally
- Integrated into Celery workers

### Redis Pub/Sub (Streaming Channel)

- Low-latency transport of token events
- Ephemeral, best-effort delivery
- One-to-many fan-out (worker → API)

> Redis is **not** a source of truth.

### ClickHouse (Long-Term Storage)

- Durable storage of conversation events
- Append-only event log
- High write throughput
- Optimized for analytics and evaluation

> Acts as the **authoritative memory** of the system.

---

## 4. Data Model (Chat History)

Chat history is stored as **events**, not mutable documents.

### Schema

```sql
CREATE TABLE chat_events (
    conversation_id  UUID,
    event_id         UUID,
    event_type       Enum8('user' = 1, 'assistant' = 2, 'tool' = 3, 'system' = 4),
    content          String,
    timestamp        DateTime64(3),
    metadata         String  -- JSON: model, latency, params, etc.
) ENGINE = MergeTree()
ORDER BY (conversation_id, timestamp);
```

### Enables

- Full conversation replay
- Offline evaluation
- RAG over historical chats
- Dataset generation for fine-tuning

---

## 5. Request Lifecycle

### 5.1 User Message

```
User submits message via Streamlit
            │
            ▼
        FastAPI
            ├── creates / resolves conversation_id
            ├── persists user event to ClickHouse
            ├── enqueues Celery inference task
            └── returns immediately, opens SSE stream
```

### 5.2 Inference & Streaming

```
Celery worker consumes task from RabbitMQ
            │
            ▼
        Worker calls vLLM
            │
            ▼ (as tokens are generated)
        Publish token events to Redis Pub/Sub
            │
            ▼
        FastAPI subscribes to Redis
            └── forwards tokens to UI via SSE
```

> Tokens are **not** persisted.

### 5.3 Completion

```
Worker emits done signal
            │
            ▼
        FastAPI
            ├── assembles full assistant message
            ├── persists assistant event to ClickHouse
            └── closes SSE stream
```

> ClickHouse always contains only **semantic message boundaries**, not tokens.

---

## 6. Failure Semantics

| Failure            | Outcome                                      |
|--------------------|----------------------------------------------|
| UI disconnect      | Streaming stops, history preserved           |
| FastAPI restart    | Streaming interrupted, no data loss          |
| Worker crash       | Task retried by Celery                       |
| Redis failure      | Token streaming lost, final message persisted|
| ClickHouse failure | Request fails explicitly                     |

> System prioritizes **durability of meaning** over durability of UX effects.

---

## 7. Why This Design

### Why RabbitMQ + Celery?

- Clear job semantics
- Mature retry & timeout handling
- Easy horizontal scaling
- Simpler than Kafka for execution workloads

### Why Redis Pub/Sub?

- Ultra-low latency
- Minimal operational overhead
- Suitable for transient token streams

### Why ClickHouse?

- Append-only event workloads
- Analytical queries (evaluation, metrics)
- Scales better than OLTP DBs for logs

---

## 8. Extension Paths

This architecture naturally extends to:

- **RAG** — embedding consumers reading from ClickHouse
- **Agent systems** — multi-step Celery tasks or Ray actors
- **Offline evaluation pipelines**
- **Kafka-based event backbone** — optional future replacement for streams

---

## 9. Design Principle Summary

```
Queues move work.
Pub/Sub moves signals.
Databases store truth.
```

This separation keeps the system simple, explainable, and production-aligned.
