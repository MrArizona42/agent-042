# Implementation Plan: Asynchronous Chat System

> **Reference:** [CHAT-BOT-SYSTEM-DESIGN.md](CHAT-BOT-SYSTEM-DESIGN.md)
> **Approach:** Incremental phases with validation gates
> **Created:** February 2026

---

## Overview

This document outlines the implementation path from the current synchronous architecture to the target asynchronous system with durable storage and API authorization.

### Current State

```
Streamlit → FastAPI → vLLM (direct HTTP)
                ↓
              Qdrant (RAG)
```

- Synchronous request/response
- No message persistence
- No user authorization
- Streaming via direct SSE from Gateway

### Target State

```
Streamlit → FastAPI → RabbitMQ → Celery → vLLM
                ↓           ↑
           ClickHouse    Redis Pub/Sub
```

- Asynchronous job execution
- Durable conversation history
- API key authorization
- Streaming via Redis Pub/Sub

---

## Implementation Phases

| Phase | Focus                         | New Infrastructure        | Risk Level |
|-------|-------------------------------|---------------------------|------------|
| 1     | Async Inference               | RabbitMQ, Celery, Redis   | Medium     |
| 2     | Durable History               | ClickHouse                | Low        |
| 3     | API Authorization             | —                         | Low        |
| 4     | Cleanup & Hardening           | —                         | Low        |

---

## Phase 1: Async Inference Pipeline

**Goal:** Decouple API responsiveness from LLM inference latency.

### 1.1 Infrastructure Setup

#### Add services to `docker-compose.yaml`

```yaml
# RabbitMQ - Message broker
rabbitmq:
  image: rabbitmq:3-management
  ports:
    - "${RABBITMQ_PORT:-5672}:5672"
    - "${RABBITMQ_MGMT_PORT:-15672}:15672"
  environment:
    RABBITMQ_DEFAULT_USER: ${RABBITMQ_USER:-agent}
    RABBITMQ_DEFAULT_PASS: ${RABBITMQ_PASS:-agent}
  volumes:
    - rabbitmq_data:/var/lib/rabbitmq
  networks:
    - backend_net
  healthcheck:
    test: ["CMD", "rabbitmq-diagnostics", "check_running"]
    interval: 10s
    timeout: 5s
    retries: 5
  restart: unless-stopped

# Redis - Pub/Sub for token streaming
redis:
  image: redis:7-alpine
  ports:
    - "${REDIS_PORT:-6379}:6379"
  volumes:
    - redis_data:/data
  networks:
    - backend_net
  healthcheck:
    test: ["CMD", "redis-cli", "ping"]
    interval: 10s
    timeout: 5s
    retries: 5
  restart: unless-stopped

# Celery Worker - LLM task execution
celery-worker:
  build:
    context: ${PROJECT_ROOT}
    dockerfile: infra/docker/celery/Dockerfile
  environment:
    CELERY_BROKER_URL: amqp://${RABBITMQ_USER:-agent}:${RABBITMQ_PASS:-agent}@rabbitmq:5672//
    REDIS_URL: redis://redis:6379/0
    VLLM_BASE_URL: http://vllm:8000
  depends_on:
    rabbitmq:
      condition: service_healthy
    redis:
      condition: service_healthy
    vllm:
      condition: service_healthy
  networks:
    - backend_net
  restart: unless-stopped
```

#### New volumes

```yaml
volumes:
  # ... existing volumes
  rabbitmq_data:
  redis_data:
```

### 1.2 Celery Worker Implementation

Create `src/worker/` module:

```
src/worker/
├── __init__.py
├── celery_app.py      # Celery app configuration
├── tasks.py           # LLM inference task
└── config.py          # Worker settings
```

#### Key implementation points

**`celery_app.py`**
- Configure Celery with RabbitMQ broker
- Set task acknowledgment to `late` (ack after completion)
- Configure retry policy for transient failures

**`tasks.py`**
- `generate_response` task:
  - Accepts `conversation_id`, `messages`, `params`
  - Calls vLLM via HTTP
  - Publishes tokens to Redis channel `tokens:{conversation_id}`
  - Publishes completion event with full response

### 1.3 Gateway Modifications

#### New dependencies

Add to `requirements-gateway.txt`:
```
celery[redis]>=5.3.0
redis>=5.0.0
```

#### New services

Create `src/gateway/services/`:

| File              | Responsibility                              |
|-------------------|---------------------------------------------|
| `celery_client.py`| Enqueue tasks, check task status            |
| `redis_stream.py` | Subscribe to Redis Pub/Sub, yield tokens    |

#### Modify `processing.py`

```python
# Before (synchronous)
async def chat(payload):
    response = await vllm_client.chat_completions(payload)
    return response

# After (asynchronous)
async def chat(payload):
    conversation_id = payload.conversation_id or uuid4()

    # Enqueue task
    task = generate_response.delay(
        conversation_id=str(conversation_id),
        messages=payload.messages,
        params=payload.model_dump(),
    )

    return {"conversation_id": conversation_id, "task_id": task.id}
```

#### Modify streaming endpoint

```python
async def stream_chat(payload) -> AsyncGenerator[bytes, None]:
    conversation_id = payload.conversation_id or uuid4()

    # Enqueue task
    generate_response.delay(...)

    # Subscribe to Redis and yield tokens
    async for token in redis_stream.subscribe(f"tokens:{conversation_id}"):
        yield f"data: {token}\n\n"
```

### 1.4 Validation Gate

Before proceeding to Phase 2:

- [ ] RabbitMQ management UI accessible at `:15672`
- [ ] Celery worker processes tasks from queue
- [ ] Tokens stream through Redis to UI
- [ ] Retry works on worker restart
- [ ] Latency acceptable (measure P50, P95, P99)

---

## Phase 2: Durable Conversation History

**Goal:** Persist all conversations for replay, analytics, and evaluation.

### 2.1 Infrastructure Setup

#### Add ClickHouse to `docker-compose.yaml`

```yaml
clickhouse:
  image: clickhouse/clickhouse-server:latest
  ports:
    - "${CLICKHOUSE_HTTP_PORT:-8123}:8123"
    - "${CLICKHOUSE_NATIVE_PORT:-9000}:9000"
  volumes:
    - clickhouse_data:/var/lib/clickhouse
    - ${PROJECT_ROOT}/infra/clickhouse/init:/docker-entrypoint-initdb.d
  environment:
    CLICKHOUSE_USER: ${CLICKHOUSE_USER:-default}
    CLICKHOUSE_PASSWORD: ${CLICKHOUSE_PASSWORD:-}
  networks:
    - backend_net
  healthcheck:
    test: ["CMD", "clickhouse-client", "--query", "SELECT 1"]
    interval: 10s
    timeout: 5s
    retries: 5
  restart: unless-stopped
```

### 2.2 Database Schema

Create `infra/clickhouse/init/001_chat_events.sql`:

```sql
CREATE TABLE IF NOT EXISTS chat_events (
    conversation_id  UUID,
    event_id         UUID,
    event_type       Enum8('user' = 1, 'assistant' = 2, 'tool' = 3, 'system' = 4),
    content          String,
    timestamp        DateTime64(3),
    metadata         String  -- JSON
) ENGINE = MergeTree()
ORDER BY (conversation_id, timestamp);

-- Index for conversation retrieval
CREATE INDEX idx_conversation_id ON chat_events (conversation_id) TYPE bloom_filter GRANULARITY 1;
```

### 2.3 Gateway Integration

#### New dependencies

```
clickhouse-connect>=0.7.0
```

#### New service

Create `src/gateway/services/history.py`:

```python
class ChatHistoryService:
    async def save_event(self, event: ChatEvent) -> None:
        """Persist a chat event to ClickHouse."""

    async def get_conversation(self, conversation_id: UUID) -> list[ChatEvent]:
        """Retrieve full conversation history."""

    async def list_conversations(self, limit: int = 50) -> list[ConversationSummary]:
        """List recent conversations."""
```

#### Modify request lifecycle

1. **On user message:** Save user event before enqueueing task
2. **On completion:** Save assistant event after receiving done signal
3. **On conversation load:** Fetch history from ClickHouse

### 2.4 API Endpoints

Add to Gateway:

| Endpoint                              | Method | Description                    |
|---------------------------------------|--------|--------------------------------|
| `/v1/conversations`                   | GET    | List conversations             |
| `/v1/conversations/{id}`              | GET    | Get conversation history       |
| `/v1/conversations/{id}`              | DELETE | Delete conversation            |

### 2.5 UI Updates

Modify Streamlit to:

- Load conversation history on page load (if `conversation_id` in URL)
- Display conversation list in sidebar
- Allow switching between conversations

### 2.6 Validation Gate

Before proceeding to Phase 3:

- [ ] User events persist to ClickHouse
- [ ] Assistant events persist after completion
- [ ] Conversation replay works
- [ ] UI displays conversation list
- [ ] No data loss on component restarts

---

## Phase 3: API Authorization

**Goal:** Protect API with simple API key authentication.

### 3.1 Design

| Aspect           | Decision                                      |
|------------------|-----------------------------------------------|
| Auth method      | API key in `Authorization: Bearer <key>` header |
| Key storage      | Environment variable (single key) or ClickHouse (multiple) |
| Scope            | Gateway API only (UI uses Gateway internally) |
| Rate limiting    | Optional, can add with Redis                  |

### 3.2 Implementation

#### Single API key (simplest)

Add to Gateway config:

```python
class Settings(BaseSettings):
    api_key: str | None = None  # If set, API requires auth
```

#### Middleware

Create `src/gateway/middleware/auth.py`:

```python
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str | None):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        # Skip auth for health endpoints
        if request.url.path in ["/health", "/ready"]:
            return await call_next(request)

        # Skip if no API key configured
        if not self.api_key:
            return await call_next(request)

        # Validate API key
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            raise HTTPException(401, "Missing API key")

        token = auth[7:]
        if token != self.api_key:
            raise HTTPException(403, "Invalid API key")

        return await call_next(request)
```

#### Register middleware

In `main.py`:

```python
app.add_middleware(APIKeyMiddleware, api_key=settings.api_key)
```

#### UI configuration

Add API key to UI config to pass to Gateway client:

```python
class Settings(BaseSettings):
    gateway_api_key: str | None = None
```

### 3.3 Multi-key support (optional)

If you need multiple API keys with different permissions:

1. Store keys in ClickHouse:
   ```sql
   CREATE TABLE api_keys (
       key_hash    String,
       name        String,
       created_at  DateTime64(3),
       expires_at  DateTime64(3) NULL,
       permissions Array(String)
   ) ENGINE = MergeTree()
   ORDER BY key_hash;
   ```

2. Validate against table in middleware
3. Add key management endpoints (create, revoke, list)

### 3.4 Validation Gate

Before Phase 4:

- [ ] Unauthenticated requests rejected with 401
- [ ] Valid API key grants access
- [ ] UI works with configured API key
- [ ] Health endpoints remain public

---

## Phase 4: Cleanup & Hardening

**Goal:** Production readiness.

### 4.1 Remove Legacy Code

- [ ] Remove direct vLLM calls from Gateway (use Celery only)
- [ ] Remove synchronous `chat()` fallback if no longer needed

### 4.2 Observability

| Component     | Tool              | Purpose                     |
|---------------|-------------------|-----------------------------|
| Metrics       | Prometheus        | Latency, throughput, errors |
| Tracing       | OpenTelemetry     | Request flow visualization  |
| Logs          | Structured JSON   | Debugging, audit            |

#### Key metrics to add

- `chat_request_total` — counter by status
- `chat_latency_seconds` — histogram (enqueue to completion)
- `celery_task_duration_seconds` — histogram
- `redis_pubsub_messages_total` — counter

### 4.3 Configuration Audit

Review all environment variables:

| Variable              | Required | Secret | Default       |
|-----------------------|----------|--------|---------------|
| `RABBITMQ_USER`       | Yes      | No     | `agent`       |
| `RABBITMQ_PASS`       | Yes      | Yes    | —             |
| `REDIS_URL`           | Yes      | No     | —             |
| `CLICKHOUSE_PASSWORD` | No       | Yes    | —             |
| `API_KEY`             | No       | Yes    | —             |

### 4.4 Documentation

- [ ] Update `infra/README.md` with new services
- [ ] Document API endpoints (OpenAPI is auto-generated)
- [ ] Add runbook for common operations

### 4.5 Validation Gate

- [ ] All health checks pass
- [ ] Metrics exported to Prometheus
- [ ] No hardcoded secrets in code
- [ ] Documentation complete

---

## File Changes Summary

### New Files

| Path                                        | Description                    |
|---------------------------------------------|--------------------------------|
| `src/worker/__init__.py`                    | Celery worker package          |
| `src/worker/celery_app.py`                  | Celery configuration           |
| `src/worker/tasks.py`                       | LLM inference task             |
| `src/worker/config.py`                      | Worker settings                |
| `src/gateway/services/celery_client.py`    | Task enqueueing                |
| `src/gateway/services/redis_stream.py`     | Pub/Sub subscription           |
| `src/gateway/services/history.py`          | ClickHouse persistence         |
| `src/gateway/middleware/auth.py`           | API key validation             |
| `infra/docker/celery/Dockerfile`           | Worker container               |
| `infra/clickhouse/init/001_chat_events.sql`| Schema initialization          |

### Modified Files

| Path                                | Changes                              |
|-------------------------------------|--------------------------------------|
| `infra/compose/docker-compose.yaml` | Add RabbitMQ, Redis, ClickHouse, Worker |
| `infra/compose/.env.example`        | Add new env vars                     |
| `src/gateway/config.py`             | Add Redis, Celery, ClickHouse, API key settings |
| `src/gateway/main.py`               | Add auth middleware                  |
| `src/gateway/services/processing.py`| Async task enqueueing                |
| `src/gateway/api/v1/openai_compat.py`| Stream via Redis                    |
| `src/ui/app.py`                     | Conversation list, history loading   |
| `src/ui/client.py`                  | Add API key header                   |
| `infra/README.md`                   | Document new services                |

---

## Estimated Effort

| Phase | Effort    | Dependencies              |
|-------|-----------|---------------------------|
| 1     | 3-5 days  | —                         |
| 2     | 2-3 days  | Phase 1                   |
| 3     | 1-2 days  | —                         |
| 4     | 2-3 days  | Phases 1-3                |

**Total:** ~8-13 days of implementation work

---

## Risks & Mitigations

| Risk                              | Mitigation                                      |
|-----------------------------------|-------------------------------------------------|
| Celery/Redis adds latency         | Benchmark; fallback to direct calls if needed   |
| ClickHouse learning curve         | Start with simple schema; expand later          |
| Worker scaling complexity         | Start with 1 worker; scale horizontally later   |
| Breaking existing streaming       | Feature flag for async mode during transition   |

---

## Decision Log

| Date       | Decision                                | Rationale                          |
|------------|----------------------------------------|------------------------------------|
| 2026-02-08 | Use API keys over OAuth                | Simpler; sufficient for current needs |
| 2026-02-08 | Prioritize async inference first       | Core architectural change          |
| 2026-02-08 | Incremental phases over big bang       | Reduce risk, validate incrementally |
