# Agent 042 Improvement Plan

This is the single planning document for the remaining polish work and the next
expansion stage. The plan is structured so the schema foundation comes first,
then observability and analytics are built as one coherent mini-project, then
documentation is refreshed around the real workflows, and only then the stack
expands or adds lower-priority operational hardening.

## Recommended Order

1. **Batch 1: Database Migration Foundation** - make the operational Postgres
   schema explicit and reproducible.
2. **Batch 2: Observability And Analytics** - make the whole app inspectable
   through Grafana, with logs, metrics, traces, and production inference
   analytics.
3. **Batch 3: Documentation Improvements** - document the workflows after the
   major surfaces exist.
4. **Batch 4: Purposeful Platform Expansion** - add production decision loops
   such as A/B champion/challenger evaluation.
5. **Batch 5: Deferred Operational Hardening** - add operator health checks and
   gateway abuse protection after the higher-value work is complete.

This order keeps the project portfolio-friendly: first prove the system is
solid, then prove it is observable, then prove it can scale and learn from
production behavior.

## Batch 1: Database Migration Foundation

Batch 1 should stay narrow: introduce managed migrations for the existing
Postgres application schema and stop relying on implicit table creation.

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

### 4. Observability Workflow And Dashboard Specification

Current state: Grafana dashboards exist for infrastructure, gateway/vLLM, and
RAG/eval observability. Dashboard implementation is not part of this plan, but
the project needs a clean technical document describing the overall
observability workflow and the dashboard categories that should exist.

Target state:

- Add a clean technical observability doc covering:
  - what each backend stores;
  - how Grafana ties the backends together;
  - how logs, metrics, traces, Postgres metadata, and ClickHouse analytics
    should be used together;
  - which dashboard types are needed;
  - which dashboard types depend on future ClickHouse/event-stream data.
- Define dashboard categories, not final dashboard JSON:
  - service health and infrastructure;
  - gateway/API latency and errors;
  - vLLM throughput and generation latency;
  - queue and worker behavior;
  - RAG retrieval health;
  - eval/model quality trends;
  - production inference analytics.
- Leave actual Grafana dashboard development for separate dashboard-specific
  work.

Acceptance criteria:

- `docs/observability.md` explains the end-to-end observability workflow.
- The doc names dashboard categories and the data source behind each category.
- The doc clearly separates currently available data from planned ClickHouse
  analytics.
- No dashboard JSON work is required by this item.

Suggested first PR:

- Add `docs/observability.md`.
- Link the doc from `infra/README.md` and, later, the portfolio quickstart.

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

### 8. LLM Observability Product Evaluation

Current state: Batch 2 should provide vendor-neutral tracing, logs, metrics, and
analytics. Specialized LLM observability tools may still be useful for prompt
and retrieval review workflows.

Target state:

- Evaluate Langfuse or Arize Phoenix as part of the main observability design,
  not as a later afterthought.
- Decide whether either tool adds enough value on top of OpenTelemetry, Tempo,
  Loki, Prometheus, Postgres, and ClickHouse.
- Define what may be captured if an LLM observability product is adopted:
  - prompt/response metadata;
  - retrieval context metadata;
  - latency per LLM/RAG step;
  - feedback labels;
  - redacted prompt/response samples only if explicitly allowed.
- Define privacy, redaction, and retention rules before storing prompt or
  response text.
- Use these tools for prompt/retrieval review, not as replacements for the core
  observability stack.

Acceptance criteria:

- `docs/observability.md` includes a recommendation: adopt one product, defer
  adoption, or explicitly skip for now.
- The recommendation explains what problem the product solves that the base
  Grafana/OpenTelemetry stack does not.
- Any prompt/response capture plan includes explicit redaction and retention
  rules.

## Batch 3: Documentation Improvements

Batch 3 should refresh documentation after the observability and analytics work
exists, so the docs describe real operator/reviewer workflows instead of
promising future surfaces.

### 1. Portfolio Quickstart

Current state: the repo has strong system documentation, but a reviewer still
has to infer the shortest impressive path through the stack.

Target state:

- Add a concise quickstart that demonstrates the main portfolio story:
  gateway, async inference, RAG retrieval, alias lifecycle, eval metrics,
  observability, and training/eval artifacts.
- Include copy-paste API examples for non-streaming and streaming chat.
- Link to deeper docs instead of duplicating them.
- Include real screenshots or links to real Grafana/Airflow/MLflow views after
  Batch 2 is complete.

Acceptance criteria:

- A reader can identify the recommended demo path in under five minutes.
- The quickstart reflects the implemented observability workflow rather than a
  placeholder version of it.
- The quickstart links to database migration, RAG operations, observability, and
  CI reproduction docs.

Suggested first PR:

- Add `docs/portfolio-quickstart.md`.
- Add API examples with `curl` and expected response shape.
- Add screenshots after Batch 2 dashboards/traces/logs/analytics are available.

### 2. CI Reproduction Documentation

Current state: CI is already structured into pre-commit, core pytest, and
training pytest jobs. The gap is documentation: a contributor should be able to
run the same commands locally without reverse-engineering `.github/workflows/ci.yml`.

Target state:

- Document the exact CI commands and environment variables:
  - pre-commit job;
  - core pytest dependency install;
  - core pytest test paths;
  - training pytest dependency install;
  - training pytest command.
- Explain that CI uses `PROJECT_ROOT=$PWD` and `PYTHONPATH=src`.
- Avoid changing CI unless the documentation work reveals a real mismatch.

Acceptance criteria:

- A local developer can reproduce each CI job from docs.
- The documented commands match `.github/workflows/ci.yml`.
- No new markers, test restructuring, or CI changes are introduced without a
  concrete failing case.

Suggested first PR:

- Add `docs/testing.md` with copy-paste commands from CI.
- Link it from `README.md` and the portfolio quickstart.

## Batch 4: Purposeful Platform Expansion

Batch 4 should add features that directly use the production observability and
analytics foundation from Batch 2.

### 1. RAG Source Citations

Current state: RAG retrieval can provide context to the model, and runtime
observability tracks provenance internally, but user-facing answers do not yet
make source grounding explicit enough for research workflows.

Problem to solve:

- researchers need to know which documents support an answer;
- RAG answers should expose source provenance without forcing the user to inspect
  backend logs or Qdrant metadata;
- citations should make hallucinations and weak retrieval easier to spot.

Target state:

- Preserve source metadata through retrieval, prompt construction, generation,
  streaming, and persisted chat history.
- Ask the model to cite retrieved sources in answers when RAG context is used.
- Return structured citation metadata in API responses:
  - source title or document id;
  - source URI;
  - chunk id or section metadata when available;
  - rank/score where useful;
  - knowledge base and alias.
- Render citations in the UI in a compact, inspectable way.
- Track citation coverage in inference events and ClickHouse analytics.

Acceptance criteria:

- A RAG answer can be traced from visible citation to retrieved chunk metadata.
- Non-RAG answers do not invent citations.
- API and UI behavior remain useful even when some source metadata is missing.
- Tests cover citation metadata propagation and no-citation behavior.

Likely files:

- `src/rag/runtime/models.py`
- `src/gateway/services/rag_service.py`
- `src/gateway/services/prompt_builder.py`
- `src/gateway/services/processing.py`
- `src/gateway/schemas/openai_chat.py`
- `src/ui/app.py`

### 2. User Feedback Tracking

Current state: there is no simple way for users to tell the system whether an
answer was useful. Quality signals come mostly from offline evals and operator
inspection.

Problem to solve:

- offline evals do not capture whether real users found an answer helpful;
- researchers may need to flag bad grounding, missing citations, or weak
  answers quickly;
- later A/B decisions need lightweight human feedback signals.

Target state:

- Add simple feedback capture:
  - thumbs up/down on an answer;
  - optional short reason or category;
  - optional "choose between two answers" workflow for comparison tasks.
- Associate feedback with request id, chat session id, model/adapter metadata,
  RAG sources, citations, and timestamp.
- Publish feedback to `feedback-events` and store it in ClickHouse for
  analytics.
- Keep feedback UI minimal and non-blocking.
- Define privacy rules for any free-text feedback.

Acceptance criteria:

- Users can submit feedback for a completed answer.
- Feedback can be joined to inference events by request id.
- Feedback analytics can be queried in ClickHouse.
- The system supports thumbs feedback first, with pairwise answer choice as a
  later extension if needed.

Likely files:

- `src/gateway/api/v1/`
- `src/gateway/services/event_publisher.py`
- `src/ui/app.py`
- `infra/clickhouse/migrations/`

### 3. A/B Champion/Challenger Evaluation

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

## Batch 5: Deferred Operational Hardening

These items are useful, but they should come after the core observability work.
By then the project will have better logs, traces, dashboards, and analytics,
which makes these hardening tasks easier to design and validate.

### 1. Compose Health Inspection

Current state: services have Compose definitions and runtime tests, but there is
no compact operator command that summarizes whether the already-running Compose
deployment is healthy. Compose already defines many service healthchecks; this
task should not create a CI test suite that requires running services.

Problem to solve:

- after deploy or local `docker compose up`, the operator should have one
  command that answers "is the stack basically alive?";
- failures should point to the service and endpoint that failed;
- this should wrap existing healthchecks/endpoints, not duplicate application
  tests.

Target state:

- Add an operator-facing script or documented command sequence for an
  already-running Compose stack.
- Check:
  - `docker compose ps` health state;
  - gateway `/health`;
  - gateway `/metrics`;
  - Prometheus readiness;
  - Grafana health;
  - Qdrant readiness;
  - Redis/RabbitMQ/Postgres through either Compose health state or lightweight
    container exec checks.
- Keep vLLM/model generation out of the default check.
- Add optional deeper checks later if a concrete operator need appears.

Acceptance criteria:

- The check is safe to run against local or server Compose.
- Failures produce actionable service names and URLs.
- It is documented as an operator/deploy check, not as part of the normal CI
  test suite.

Suggested first PR:

- Add `scripts/compose_health.sh` or `scripts/compose_status.sh`.
- Document it in `infra/README.md` and the quickstart.

### 2. Gateway Abuse Protection

Current state: the gateway requires auth for real user workflows, but it does
not have an explicit abuse-protection layer at the edge. The goal is not to mix
business/backend request constraints with user throttling. Prompt budget and
backend-specific request shaping already belong to the prompt/budgeting code.

Problem to solve:

- unauthenticated traffic should not be able to cheaply hammer public endpoints;
- authenticated users should have a basic overuse guardrail;
- rate-limit decisions should happen before expensive RAG, Celery, or vLLM work.

Target state:

- Add Redis-backed gateway rate limiting.
- Use authenticated user id as the primary rate-limit key for protected routes.
- Use IP-based limits for unauthenticated/public routes such as health,
  discovery, auth start/callback, and static/non-mutating endpoints if exposed.
- Keep user-agent out of the primary key by default. It is easy to spoof and can
  create surprising cardinality; it can be logged for diagnostics.
- Return consistent `429` responses with useful retry metadata.
- Add config values for authenticated and unauthenticated limits.
- Keep OpenAI-compatible payload flexibility unless a separate, concrete backend
  failure mode requires a bound.

Acceptance criteria:

- Authenticated chat requests are limited per user id.
- Unauthenticated/public requests are limited per client IP.
- Rate-limit state works across multiple gateway replicas through Redis.
- Redis failure behavior is explicit and documented.
- Tests cover auth-keyed limits, IP-keyed limits, and `429` responses.

Suggested first PR:

- Add a small gateway rate-limit middleware/service.
- Add settings for authenticated and unauthenticated limit windows.
- Add route-level tests with mocked Redis/time.

## Deferred Ideas

These are valuable, but they should not interrupt the batches above:

- Function-calling agent layer on top of current task routing.
- Web search as an optional tool.
- Broader user feedback UX.
- Full multi-node Kubernetes deployment.
- More advanced cost accounting once ClickHouse inference analytics exists.

### Spark For Data Quality And Feedback Loops

Current idea:

- RAG source dedup/filter/chunk quality gates.
- Training data dedup/filter before LoRA training.
- Weekly KB gap detection from low-hit inference events.
- Weekly query drift detection against a baseline.
- Write gap/drift reports into ClickHouse for Grafana dashboards.

This should wait until ClickHouse inference analytics exists and there is enough
real or synthetic traffic to justify batch data jobs.

### Kubernetes Later: k3s, Helm, KEDA

Current idea:

- Add k3s/Helm after the production data loop exists.
- Use KEDA to autoscale Celery workers from RabbitMQ queue depth.
- Add GPU resource requests/limits for vLLM.
- Support rolling updates for model or adapter changes.

This should wait until Compose-based operations are observable enough that
Kubernetes solves a visible problem instead of adding infrastructure for its own
sake.
