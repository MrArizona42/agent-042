# Agent 042 Improvement Plan

This is the single planning document for the remaining polish work and the next
expansion stage. The plan is structured so foundational safety comes first,
then observability and analytics are built as one coherent mini-project, and
only then the stack expands into heavier platform features.

## Recommended Order

1. **Batch 1: Foundation And Safety** - make the current system safer,
   reproducible, and easier to run.
2. **Batch 2: Observability And Analytics** - make the whole app inspectable
   through Grafana, with logs, metrics, traces, and production inference
   analytics.
3. **Batch 3: Purposeful Platform Expansion** - add Spark, Kubernetes, and
   other larger features only after the operational feedback loop exists.

This order keeps the project portfolio-friendly: first prove the system is
solid, then prove it is observable, then prove it can scale and learn from
production behavior.

## Batch 1: Foundation And Safety

Batch 1 should not add major new infrastructure. It should make the existing
Postgres/Compose/FastAPI/Airflow system feel intentional and production-shaped.

### 1. Database Migrations

Current state: the gateway still bootstraps the `agent042` database with
`Base.metadata.create_all` during startup. There are also standalone SQL files
for eval/chat schema changes.

Target state:

- Add Alembic for the `agent042` Postgres application database.
- Generate an initial migration from the current SQLAlchemy ORM schema.
- Fold existing schema patches, especially chat usage columns, into managed
  migrations.
- Replace startup `create_all` with an explicit migration path.
- Document local Compose and server migration commands.

Postgres remains the operational source of truth for:

- `users`;
- `chat_sessions`;
- `chat_messages`;
- `eval_runs`;
- `eval_samples`.

ClickHouse is intentionally out of Alembic scope. It will get separate SQL
migrations in Batch 2 because it serves append-only analytics, not OLTP
application state.

Acceptance criteria:

- A fresh database can be created only through migrations.
- An existing database can be stamped or upgraded without dropping data.
- Gateway startup no longer silently mutates schema.
- Tests cover migration metadata and at least one upgrade path.

Suggested first PR:

- Add `alembic.ini`, `alembic/env.py`, and initial revision.
- Add `alembic` to the gateway/runtime dependency set.
- Update `src/gateway/main.py` to remove `Base.metadata.create_all`.
- Add `docs/database-migrations.md`.

### 2. Gateway Safety And Request Validation

Current state: request models validate basic shape, but there are no explicit
limits for message length, message count, `rag_sources`, session identifiers, or
request rate.

Target state:

- Add bounded validation to OpenAI-compatible chat requests:
  - maximum number of messages;
  - maximum message content length;
  - maximum number of `rag_sources`;
  - allowed format and length for knowledge base names and aliases;
  - bounded `chat_session_id`.
- Add rate limiting at the gateway, preferably Redis-backed so it works across
  gateway replicas later.
- Return consistent `4xx` errors before RAG retrieval or Celery enqueue.
- Add config values for all limits with production-safe defaults.

Acceptance criteria:

- Oversized requests fail before expensive work starts.
- Rate limits work per authenticated user when auth is enabled and per IP when
  auth is disabled.
- Tests cover valid requests, edge-of-limit requests, and rejected requests.
- Documentation names the limits and environment variables.

Suggested first PR:

- Add Pydantic constraints and validators in
  `src/gateway/schemas/openai_chat.py`.
- Add route-level tests around `/v1/chat/completions`.
- Add Redis-backed rate limiting middleware after validation behavior is locked
  down.

### 3. Test And CI Ergonomics

Current state: tests and Airflow tasks manually make `src/` importable in a few
places. This works, but local test execution is more implicit than it needs to
be.

Target state:

- Make `pytest` discover source imports consistently without requiring users to
  remember `PYTHONPATH=src`.
- Add focused markers for integration, compose, gpu, and slow tests.
- Keep unit tests fast by default.
- Document the common test commands in one place.

Acceptance criteria:

- `uv run pytest` works for the default unit suite from a clean checkout.
- Integration/Compose tests are opt-in.
- Airflow DAG import tests remain deterministic.

Suggested first PR:

- Add pytest config in `pyproject.toml` or `pytest.ini`.
- Add markers and update existing tests if marker warnings appear.
- Add `docs/testing.md`.

### 4. Secrets And Production Configuration Hygiene

Current state: deployment docs are detailed, but the safety checklist should be
easier to audit before exposing the stack.

Target state:

- Refresh `.env.example` coverage for gateway, auth, Redis, RabbitMQ, Postgres,
  Grafana, MLflow, Airflow, DVC/S3, and model/runtime settings.
- Mark secrets versus non-secret tuning knobs.
- Add a production preflight checklist:
  - default passwords changed;
  - OIDC configured or intentionally disabled;
  - CORS and nginx hosts restricted;
  - Airflow Fernet key set;
  - persistent volumes backed up;
  - DVC/S3 credentials scoped;
  - public endpoints reviewed.
- Explain what startup configuration summary is safe to log.

Acceptance criteria:

- A fresh deploy can be configured from `.env.example` without hunting through
  Compose.
- No real secrets are committed.
- The production checklist is short enough to actually use.

Suggested first PR:

- Audit Compose environment variables against `.env.example`.
- Add `docs/production-checklist.md`.

### 5. Compose And Service Health Smoke Tests

Current state: services have Compose definitions and runtime tests, but there is
no compact smoke suite that proves the deployment comes up coherently.

Target state:

- Add opt-in smoke tests or scripts for:
  - gateway `/health` or equivalent;
  - gateway `/metrics`;
  - Redis connectivity;
  - RabbitMQ management/API or broker connectivity;
  - Postgres connectivity;
  - Qdrant readiness;
  - Grafana and Prometheus availability.
- Keep expensive model loading out of default smoke checks unless explicitly
  requested.

Acceptance criteria:

- Smoke tests are safe to run against local Compose.
- Failures produce actionable service names and URLs.
- CI can run a lightweight subset without GPU services.

Suggested first PR:

- Add `scripts/compose_smoke.sh` or `tests/smoke/test_compose_health.py`.
- Add a README/quickstart command that runs the smoke suite.

### 6. Architecture Decision Records

Current state: the system design README explains the architecture, but the
reasoning behind key choices is not captured as small decision records.

Target state:

- Add ADRs for:
  - Docker Compose as the current deployment contract;
  - Postgres for operational state and ClickHouse for analytics;
  - RabbitMQ for task dispatch plus Redis for streams/session-adjacent state;
  - vLLM as the inference runtime;
  - Airflow for reproducible ML/RAG workflows;
  - single-repo layout for a portfolio-scale ML platform;
  - Qdrant aliases for champion/challenger RAG collection promotion.
- Keep each ADR short: context, decision, consequences, alternatives.

Acceptance criteria:

- `docs/adr/` contains numbered ADRs.
- README/system design links to the ADR index.
- ADRs explain why the current choices fit this project size and what would
  trigger revisiting them.

Suggested first PR:

- Add `docs/adr/000-index.md`.
- Add ADRs for Compose deployment, database split, and RabbitMQ plus Redis.

### 7. Portfolio Quickstart Skeleton

Current state: the repo has strong system documentation, but a reviewer still
has to infer the shortest impressive path through the stack.

Target state:

- Add a concise quickstart that demonstrates the main portfolio story:
  gateway, async inference, RAG retrieval, alias lifecycle, eval metrics,
  observability, and training/eval artifacts.
- Include copy-paste API examples for non-streaming and streaming chat.
- Link to deeper docs instead of duplicating them.
- Leave final observability screenshots until Batch 2 is done.

Acceptance criteria:

- A reader can identify the recommended demo path in under five minutes.
- The quickstart can be completed before the observability mini-project is
  finished, then refreshed with screenshots after Batch 2.

Suggested first PR:

- Add `docs/portfolio-quickstart.md`.
- Add API examples with `curl` and expected response shape.

## Batch 2: Observability And Analytics

Batch 2 is a coherent mini-project: make the whole application analyzable from
Grafana without confusing operational logs, request traces, service metrics, and
business/ML analytics.

Target architecture:

- **Prometheus** - service and infrastructure metrics.
- **Tempo** - OpenTelemetry traces.
- **Loki** - searchable container/application logs.
- **Postgres datasource** - current operational/eval metadata dashboards.
- **ClickHouse datasource** - production inference analytics.
- **Grafana** - the unified analysis surface over all of the above.

### 1. Logging Hygiene And Correlation

Current state: services use stdlib logging in many places, some modules call
`basicConfig` directly, and scripts/DAGs use `print` where convenient. Logs are
available through Docker, but they are not easy to search or correlate.

Target state:

- Add shared logging setup in `src/shared/logging.py`.
- Add configurable `LOG_LEVEL` and optional JSON log format.
- Standardize service names in logs.
- Add correlation fields where safe:
  - `request_id`;
  - `trace_id` when OpenTelemetry is enabled;
  - `celery_task_id`;
  - `chat_session_id`;
  - RAG `knowledge_base`, `alias`, and `qdrant_collection`;
  - route and status.
- Avoid logging full prompts, responses, access tokens, or secrets at info level.
- Keep `print` only for CLI JSON output and Airflow subprocess stdout where it
  improves operator ergonomics.

Acceptance criteria:

- A single request can be followed across gateway and worker logs by
  `request_id`.
- Sensitive request content is not logged by default.
- Existing tests are not made noisy by logging setup.

### 2. OpenTelemetry Tracing With Tempo

Current state: Prometheus shows aggregate metrics, but there is no request
waterfall showing where one chat completion spent time.

Target state:

- Add `otel-collector` and `tempo` to Compose.
- Export traces from Python services through OTLP.
- Add FastAPI, HTTPX, SQLAlchemy, Redis, and Celery instrumentation where useful.
- Add manual spans around ML-specific operations:
  - request validation/preparation;
  - task routing;
  - RAG retrieval;
  - reranking;
  - prompt build;
  - Celery enqueue and queue wait;
  - vLLM tokenize;
  - vLLM generation;
  - chat persistence.
- Provision Tempo as a Grafana datasource.

Acceptance criteria:

- Grafana can display a trace for one chat completion.
- Gateway and worker spans share the same trace where feasible.
- Spans include useful attributes but not raw prompt/response text by default.

### 3. Loki And Alloy Log Search

Current state: Docker logs are available but inconvenient to query across
services, time ranges, and request identifiers.

Target state:

- Add Loki to Compose.
- Add Grafana Alloy to collect Docker/container logs and push them to Loki.
- Keep Docker `json-file` logging with rotation as the local fallback.
- Provision Loki as a Grafana datasource.
- Add log labels for service/container/environment without creating high
  cardinality explosions.
- Link traces to logs through `trace_id` and/or `request_id` where possible.

Acceptance criteria:

- Grafana Explore can query logs by service and request id.
- Logs survive normal container restarts through configured volumes/retention.
- Loki is optional for local development and does not block core app startup.

### 4. Existing Dashboard Polish

Current state: Grafana dashboards exist for infrastructure, gateway/vLLM, and
RAG/eval observability. They should become proof-oriented instead of just
available.

Target state:

- Document what each existing dashboard proves:
  - request traffic and gateway latency;
  - vLLM token throughput and time-to-first-token;
  - queue depth and worker health;
  - GPU/CPU/memory/disk pressure;
  - eval score trends;
  - RAG no-hit or low-hit behavior where currently observable.
- Add missing panel descriptions and consistent panel titles.
- Add a dashboard validation checklist for demo and operations.
- Clearly separate currently implemented metrics from upcoming ClickHouse
  analytics.

Acceptance criteria:

- A reviewer can open Grafana and understand which panels matter.
- Dashboard JSON remains provisioned through `infra/grafana`.
- Docs explain the role of Prometheus, Postgres, Tempo, Loki, and ClickHouse.

Suggested first PR:

- Add `docs/observability.md`.
- Update dashboard descriptions in JSON only where low-risk.
- Link the doc from `infra/README.md` and the portfolio quickstart.

### 5. Durable Inference Events With Kafka Or Redpanda

Current state: gateway responses and usage live primarily in PostgreSQL chat
tables after generation. There is no replayable inference event log for
analytics, feedback, or downstream data jobs.

Target state:

- Gateway publishes an event after each completed generation to
  `inference-events`.
- Payload includes:
  - request id and timestamp;
  - route and status;
  - model/base model;
  - LoRA adapter and alias;
  - RAG provenance and hit counts;
  - latency and token counts;
  - user/session identifiers or privacy-preserving hashes;
  - error type when applicable.
- Reserve `feedback-events` for future explicit user feedback.
- Keep RabbitMQ as the task queue. Kafka/Redpanda is the durable, replayable
  production event log.

Acceptance criteria:

- Successful and failed generation attempts produce well-defined events.
- Event schema is documented and versioned.
- Event publication failure does not break chat completion unless strict mode is
  explicitly enabled.

Likely files:

- `src/gateway/services/event_publisher.py`
- `src/gateway/services/processing.py`
- `infra/compose/docker-compose.yaml`
- `src/shared/config.py`

### 6. ClickHouse Inference Analytics

Current state: Postgres supports eval/chat analytics and Grafana dashboards.
Production inference analytics such as latency percentiles by adapter and RAG
hit-rate trends are not stored in an OLAP-friendly shape.

Target state:

- Add ClickHouse with separate SQL migrations, for example under
  `infra/clickhouse/migrations/`.
- Ingest `inference-events` into ClickHouse.
- Create the core analytical tables:
  - `inference_log`: one row per completed/failed generation request;
  - `inference_rag_hits`: optional one row per retrieved RAG hit;
  - `feedback_events`: reserved for future user feedback;
  - `inference_daily_rollups`: optional dashboard acceleration.
- Add Grafana ClickHouse datasource.
- Add panels for:
  - latency percentiles by adapter, route, and variant;
  - token throughput;
  - RAG hit/no-hit rate;
  - error rate;
  - adapter usage;
  - session/user aggregates where privacy policy allows.

Acceptance criteria:

- Grafana can query production inference analytics from ClickHouse.
- Postgres remains the operational source of truth.
- ClickHouse schema changes are managed separately from Alembic.

Likely files:

- `infra/clickhouse/migrations/`
- Compose service and datasource provisioning
- Optional Kafka engine table/materialized view or a small consumer service

### 7. Cross-Signal Grafana Workflow

Current state: each observability surface can exist independently, but the
portfolio value comes from moving smoothly between them.

Target state:

- From a dashboard latency spike, jump to representative traces.
- From a trace, jump to Loki logs by `trace_id` or `request_id`.
- From a request id, inspect the ClickHouse inference event.
- From a RAG no-hit trend, inspect related logs/traces and source/alias
  metadata.
- Document one end-to-end demo:
  "Investigate a slow RAG chat request from Grafana dashboard to trace to logs
  to inference event."

Acceptance criteria:

- `docs/observability.md` contains the end-to-end workflow.
- The portfolio quickstart links to this workflow.
- Screenshots can be captured from a working local/server deployment.

### 8. A/B Champion/Challenger Evaluation

Current state: champion/challenger aliases exist for RAG collections and MLflow
model registry workflows, but promotion decisions are mostly offline.

Target state:

- Add `challenger_traffic_pct` config with default `0`.
- Route a controlled share of requests to challenger variants.
- Add `ab_variant` and variant metadata to inference events.
- Compare variants in ClickHouse and a notebook or script.
- Use guardrails before promotion:
  - latency;
  - error rate;
  - token budget;
  - RAG hit rate;
  - offline eval deltas.

Acceptance criteria:

- Champion/challenger comparison can use production inference data.
- Promotion recommendations include guardrail checks, not only quality deltas.
- The process is documented as an operator workflow.

Likely files:

- `src/gateway/services/task_router.py`
- `src/gateway/services/processing.py`
- `src/shared/config.py`
- `experiments/training/lora_ops.ipynb`

## Batch 3: Purposeful Platform Expansion

Batch 3 should add heavier infrastructure only when it closes a clear ML
production loop. The observability/analytics batch should make the need visible
before these features are implemented.

### 1. Spark For Data Quality And Feedback Loops

Current state: RAG source builds and training data preparation are mostly
single-process Python workflows. They are reproducible, but not yet positioned
as distributed data-quality jobs.

Target state:

- Add small, measurable Spark jobs rather than broad Spark adoption:
  - RAG source dedup/filter/chunk quality gates;
  - training data dedup/filter before LoRA training;
  - weekly KB gap detection from low-hit inference events;
  - weekly query drift detection against a baseline.
- Run Spark local/standalone from Airflow so it remains server-friendly but
  cluster-ready.
- Write gap/drift reports into ClickHouse for Grafana dashboards.

Likely files:

- `src/spark/rag_preprocessing.py`
- `src/spark/training_data_prep.py`
- `src/spark/kb_gap_detection.py`
- `src/spark/query_drift_detection.py`
- `dags/kb_gap_detection.py`
- `dags/query_drift_detection.py`
- updates to `dags/rag_lifecycle.py` and `dags/train_lora.py`

### 2. Kubernetes Later: k3s, Helm, KEDA

Current state: Docker Compose is the real deployment contract.

Target state:

- Add k3s/Helm after the production data loop exists.
- Use KEDA to autoscale Celery workers from RabbitMQ queue depth.
- Add GPU resource requests/limits for vLLM.
- Support rolling updates for model or adapter changes.

Why this is later:

- Kubernetes is impressive only when it demonstrates an operational need.
- Queue-driven scaling and zero-downtime model swaps are better demos once
  production events and analytics can prove the need.

### 3. Optional LLM Observability Products

Current state: Batch 2 should provide vendor-neutral tracing, logs, metrics, and
analytics. Specialized LLM observability tools may still be useful for prompt
and retrieval review workflows.

Target state:

- Evaluate Langfuse or Arize Phoenix only after OpenTelemetry/Tempo, Loki, and
  ClickHouse are in place.
- Capture prompt/response metadata only with explicit redaction and retention
  rules.
- Use these tools for prompt/retrieval review, not as replacements for the core
  observability stack.

## Deferred Ideas

These are valuable, but they should not interrupt the batches above:

- Function-calling agent layer on top of current task routing.
- Web search as an optional tool.
- Broader user feedback UX.
- Full multi-node Kubernetes deployment.
- More advanced cost accounting once ClickHouse inference analytics exists.

