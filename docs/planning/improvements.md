# Agent 042 Improvement Plan

This is the single planning document for the next project stage. The plan is
organized around the core AI / LLM / RAG system goals:

1. prove that the system can be observed, evaluated, and analyzed;
2. improve RAG answer quality and research usability;
3. expand platform functionality where it directly supports production LLM/RAG
   workflows;
4. keep lower-priority infrastructure and operational ideas deferred.

## Phase 1: Observability, Evaluation, And Analytics

Phase 1 should make the current system controllable and understandable end to
end. The goal is to answer: when a response is bad, slow, poorly grounded, or
surprising, how do we trace what happened, evaluate why, and analyze the pattern
across many requests?

### 1. OpenTelemetry, Tempo, Loki, And Structured Logs

Current state: logs and metrics exist, but request-level traces and searchable
cross-service logs are limited. Without correlated logs/traces, it is hard to
control multi-step flows across gateway, RAG, Celery, vLLM, Redis, and Postgres.

Target state:

- Add shared logging setup and correlation fields:
  - `request_id`;
  - `trace_id`;
  - `celery_task_id`;
  - `chat_session_id`;
  - RAG KB/alias/collection metadata where safe.
- Add OpenTelemetry instrumentation and manual spans for:
  - gateway request handling;
  - task routing;
  - RAG retrieval;
  - prompt build;
  - Celery enqueue/worker execution;
  - vLLM tokenize/generation;
  - persistence.
- Add `otel-collector` and Tempo.
- Add Loki and Grafana Alloy for searchable logs.
- Provision Grafana datasources.

Fixed design choices for the logging slice:

- Runtime logs are optimized for the single-node dedicated server deployment.
  Local execution is mainly mocked tests, not a separate application runtime
  environment.
- Use structured JSON logs for deployed Python services.
- Start with a small in-repo JSON formatter instead of adding a logging
  framework or external formatter dependency.
- Gateway generates the canonical `request_id`; client-supplied request IDs are
  not trusted as the primary correlation key.
- Raw internal `user_id` may be logged because this is an internal tool.
- Do not log full prompts, responses, cookies, access tokens, API keys, or OAuth
  payloads by default.
- Use Grafana Alloy to collect Docker `json-file` logs into Loki; do not use the
  Loki Docker logging driver initially.
- Loki labels should stay low-cardinality: `service`, `container`, and `level`.
  High-cardinality values such as `request_id`, `trace_id`, `user_id`,
  `chat_session_id`, `celery_task_id`, KB, alias, collection, adapter, and model
  stay as JSON fields.
- Do not introduce `environment=local/server/prod` labels unless the deployment
  model changes.

Acceptance criteria:

- One request can be followed from logs to trace to relevant persisted metadata.
- Prompt/response text is not logged by default.
- The setup remains optional enough that mocked tests and one-off scripts are
  not blocked.

Implementation notes:

- Structured JSON logs are implemented for Gateway, worker, embeddings, and
  reranker.
- OpenTelemetry is enabled for deployed services through Compose-provided OTLP
  settings and remains no-op when the endpoint is absent.
- Grafana datasources now include Postgres, Prometheus, Loki, and Tempo.
- The operator workflow is documented in `docs/analytics/observability.md`.

### 2. Durable Inference Events With Kafka Or Redpanda

Current state: gateway responses and usage live primarily in PostgreSQL chat
tables after generation. There is no replayable inference event log for
analytics, feedback, or downstream data jobs.

Target state:

- Gateway and worker publish lifecycle metadata to `inference.events.v1`.
- Payload includes:
  - request id and timestamp;
  - event type and schema version;
  - `trace_id` / `span_id` when available;
  - model;
  - RAG source counts and KB/alias names where safe;
  - token counts;
  - user/session/task/conversation identifiers;
  - finish reason or error type.
- Reserve explicit user feedback events for the later feedback step.
- Keep RabbitMQ as the task queue. Redpanda is the durable, replayable event
  log.

Fixed design choices:

- Use Redpanda as the Kafka-compatible broker for this single-node deployment.
- Include Redpanda Console immediately for topic inspection.
- Event publishing is enabled by default in server Compose via
  `PLATFORM__KAFKA_BOOTSTRAP_SERVERS`; do not add a separate
  `EVENTS__ENABLED` flag.
- Keep RAG event payloads coarse for now. Source/chunk details move to the
  source-citation phase.

Acceptance criteria:

- Successful and failed generation attempts produce well-defined events.
- Event schema is documented and versioned.
- Event publication failure does not break chat completion.
- Full prompts, responses, messages, generated content, access tokens, cookies,
  API keys, and OAuth payloads are rejected by schema validation. Token counts
  are allowed.

Implementation notes:

- Shared event schema and producer live under `src/shared/events/`.
- The first topic is `inference.events.v1`.
- Gateway and worker publish lifecycle metadata only.
- Operator workflow is documented in `docs/analytics/inference-events.md`.

### 3. ClickHouse Analytics Expansion

Current state: Postgres supports current eval/chat analytics. ClickHouse should
be introduced incrementally as the production inference analytics backend.

Target state:

- Add ClickHouse with separate SQL init/migration files under
  `infra/clickhouse/`.
- Ingest `inference.events.v1` directly from Redpanda using the ClickHouse
  Kafka Engine.
- Start with raw-first analytics tables:
  - `kafka_inference_events_stream`: Kafka Engine stream adapter, not durable
    analytics storage;
  - `mv_inference_events_raw`: materialized view that drains Kafka into
    ClickHouse;
  - `inference_events_raw`: durable MergeTree archive with raw JSON and common
    parsed columns.
- Add derived request-level tables later after the raw ingestion path is proven:
  - `inference_requests`;
  - `inference_rag_hits` when source citation work provides chunk/source ids;
  - `feedback_events`;
  - `inference_daily_rollups`.
- Add Grafana ClickHouse datasource.
- Support analytics for:
  - latency percentiles by adapter, route, and variant;
  - token throughput;
  - RAG hit/no-hit rate;
  - citation coverage;
  - feedback rates;
  - error rate;
  - adapter usage.

Acceptance criteria:

- Grafana or notebooks can query production inference analytics from ClickHouse.
- Postgres remains the operational source of truth.
- ClickHouse schema changes are managed separately from Alembic.

Fixed design choices:

- Use direct Redpanda-to-ClickHouse ingestion through ClickHouse Kafka Engine,
  not a Python consumer service for the first implementation.
- Keep the first ClickHouse layer raw and replay-friendly. Derived analytical
  tables come after ingestion is proven.

Implementation notes:

- Compose includes ClickHouse and installs the Grafana ClickHouse datasource
  plugin.
- Initial SQL lives in `infra/clickhouse/init/001_inference_events.sql`.
- Operator workflow and starter queries are documented in
  `docs/analytics/clickhouse-analytics.md`.

### 4. Observability And Evaluation Technical Workflow

Current state: the project has several observability and evaluation surfaces:
logs, traces, Prometheus/Grafana dashboards, Postgres-backed eval tables,
ClickHouse inference analytics, RAG runtime provenance, Airflow DAGs, and
notebooks. What is missing is one clean technical workflow that explains how to
use them together.

Target state:

- Add `docs/analytics/observability-evaluation-workflow.md`.
- Document the end-to-end diagnostic path for a chat/RAG response:
  - request enters gateway;
  - task routing and KB selection happen;
  - RAG retrieval runs;
  - prompt is assembled;
  - Celery/vLLM generation runs;
  - result streams back;
  - chat/eval/usage metadata is persisted;
  - inference event is stored for analytics;
  - logs/traces/dashboards/notebooks expose the result.
- Explain which signal answers which question:
  - logs: what happened inside one component;
  - traces: where one request spent time across components;
  - Prometheus/Grafana: aggregate service behavior;
  - Postgres eval tables: offline quality results;
  - ClickHouse events: production request analytics;
  - RAG provenance: which KB/alias/collection/chunks were used;
  - notebooks: deeper failure analysis and comparisons.
- Define dashboard categories needed later, but do not implement dashboard JSON
  in this phase.

Acceptance criteria:

- A reader can follow one request through logs, traces, persisted metadata, and
  analytics.
- The doc separates implemented signals from future or optional telemetry.
- The doc explains how evaluation and observability complement each other.

### 5. Failure Analysis Notebook

Current state: eval results are stored, but there is no single notebook focused
on diagnosing failures across retrieval, generation, and system behavior.

Target state:

- Add or extend a notebook for failure analysis, for example
  `experiments/eval/failure_analysis.ipynb`.
- Support workflows such as:
  - inspect failed or low-scoring eval samples;
  - compare champion vs challenger RAG aliases;
  - identify no-hit and low-score retrieval cases;
  - inspect generated answer, reference answer, retrieved context, and metric
    details together;
  - group failures by task, dataset, KB, alias, adapter, metric, and verdict;
  - export candidate questions for future RAG evaluation datasets.

Acceptance criteria:

- The notebook can load `eval_runs` and `eval_samples`.
- A failure row can be traced to model/RAG config and sample-level details.
- The notebook produces a short list of actionable failure categories.

Likely files:

- `experiments/eval/failure_analysis.ipynb`
- `experiments/eval/eval_scripts/runner.py`
- `src/shared/db/models.py`

### 6. Evaluation Result Readiness

Current state: the schema already stores aggregate eval runs and sample-level
details. Phase 1 should make sure these results are easy to query and interpret
before adding more metrics in Phase 2.

Target state:

- Review current `eval_runs` and `eval_samples` fields for analysis usability.
- Document the meaning of existing metrics and verdicts.
- Add small helper queries or notebook utilities for:
  - latest eval results;
  - metric trends by task/dataset/model/RAG alias;
  - failed samples by metric/verdict;
  - RAG-enabled vs non-RAG comparisons;
  - adapter and KB comparisons.
- Keep Postgres as the source of truth for current offline eval data.

Acceptance criteria:

- Existing eval results can be interpreted without reading the runner code.
- Common comparisons are available as documented SQL snippets or notebook cells.
- Any schema gaps needed for Phase 2 RAG metrics are explicitly listed.

### 7. Analytics Control Map

Current state: analytics are split between Postgres, Grafana, notebooks, and
service metrics. After Phase 1 infrastructure is added, the project needs one
map of where each analytical question should be answered.

Target state:

- Document analytics capabilities:
  - eval score trends;
  - chat/message usage currently stored in Postgres;
  - production inference event log;
  - token usage and throughput;
  - RAG provenance/log/trace inspection;
  - RAG hit/no-hit trends over traffic;
  - latency percentiles by adapter/variant/KB;
  - service metrics from Prometheus;
  - future user feedback and A/B production comparisons.

Acceptance criteria:

- There is no ambiguity about whether to use Postgres, ClickHouse, Prometheus,
  traces, logs, or notebooks for a given question.
- The map explains which analytics are available immediately after Phase 1 and
  which depend on later feedback/A/B work.

## Phase 2: RAG Quality Improvements

Phase 2 should improve the core RAG product: grounded answers, citations,
retrieval evaluation, and judge-based quality checks. Some evaluation groundwork
starts in Phase 1, but RAG-specific datasets and metrics belong here.

### 1. Source Citations In RAG Answers

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
- Track citation coverage for later analytics.

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

### 2. RAG Evaluation Datasets

Current state: the project has evaluation infrastructure, but RAG-specific
quality depends on having curated questions, expected sources, and expected
answer properties.

Target state:

- Add small curated RAG evaluation datasets for the main KBs.
- Include examples that test:
  - exact source lookup;
  - multi-document synthesis;
  - no-answer or insufficient-context behavior;
  - citation correctness;
  - questions that should prefer one KB over another.
- Store dataset provenance and versioning so RAG eval results are reproducible.
- Reuse Phase 1 failure analysis to turn real failures into new eval examples.

Acceptance criteria:

- Each core KB has at least a small representative eval set.
- Dataset rows include enough metadata to evaluate retrieval and citation
  quality, not only final answer text.
- Eval datasets can be run from the existing eval workflow.

Likely files:

- `assets/datasets/`
- `experiments/eval/eval_scripts/datasets.py`
- `experiments/eval/eval_scripts/runner.py`

### 3. RAG Metrics

Current state: automatic metrics exist, but RAG quality should be decomposed
into retrieval quality, citation quality, and final answer quality.

Target state:

- Add or formalize retrieval metrics:
  - Recall@k;
  - MRR;
  - nDCG where labels support it;
  - hit/no-hit rate;
  - expected-source coverage.
- Add citation metrics:
  - citation presence when RAG is used;
  - citation precision where expected sources are known;
  - unsupported citation detection;
  - answer sentences with/without cited support where feasible.
- Add answer quality metrics:
  - existing automatic metrics where appropriate;
  - LLM-as-judge relevance;
  - LLM-as-judge faithfulness/groundedness;
  - refusal/no-answer correctness for insufficient context.
- Store metric outputs in `eval_runs` / `eval_samples` with enough detail for
  the failure analysis notebook.

Acceptance criteria:

- RAG eval can show whether a failure came from retrieval, citation behavior, or
  answer generation.
- LLM-as-judge prompts are versioned and documented.
- Metrics can compare KB aliases such as champion/challenger.

Likely files:

- `experiments/eval/eval_scripts/metrics/automatic.py`
- `experiments/eval/eval_scripts/metrics/llm_judge.py`
- `experiments/eval/eval_scripts/retrieval_bench.py`
- `experiments/eval/eval_scripts/runner.py`

### 4. RAG Regression And Promotion Workflow

Current state: Qdrant aliases and eval tables support comparison, but the
promotion workflow should explicitly connect RAG builds, eval metrics, and
failure analysis.

Target state:

- Define a repeatable workflow before promoting a new KB alias:
  - build/materialize candidate collection;
  - run RAG eval dataset;
  - inspect retrieval/citation/answer metrics;
  - review failure analysis notebook;
  - promote or reject alias with a short operator note.
- Document guardrails:
  - retrieval quality must not regress;
  - citation quality must not regress;
  - answer quality must improve or stay neutral;
  - latency impact should be visible.

Acceptance criteria:

- A RAG alias promotion can be justified with eval results and failure analysis.
- The workflow is documented in RAG operations docs or a dedicated eval doc.

## Phase 3: Functionality And Platform Expansion

Phase 3 adds new platform capabilities incrementally. These should support the
Phase 1 and Phase 2 quality loops rather than distract from them.

### 1. LLM Observability Product Evaluation

Current state: OpenTelemetry/Tempo/Loki/Prometheus/Grafana can provide a strong
vendor-neutral observability stack. Specialized LLM observability tools may
still be useful for prompt and retrieval review workflows.

Target state:

- Evaluate Langfuse or Arize Phoenix as part of the platform expansion.
- Decide whether either tool adds enough value on top of the base stack.
- Define what may be captured if adopted:
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

- There is a documented recommendation: adopt one product, defer adoption, or
  explicitly skip for now.
- The recommendation explains what problem the product solves that the base
  stack does not.
- Any prompt/response capture plan includes explicit redaction and retention
  rules.

### 2. User Feedback Tracking

Current state: there is no simple way for users to tell the system whether an
answer was useful. Quality signals come mostly from offline evals and operator
inspection.

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
  - citation quality;
  - user feedback;
  - offline eval deltas.

Acceptance criteria:

- Champion/challenger comparison can use production inference data.
- Promotion recommendations include guardrail checks, not only quality deltas.
- The process is documented as an operator workflow.

## Phase 4: Future Ideas

These are valuable, but they should not interrupt the phases above.

### Operational Hardening

- Add Alembic migrations for the `agent042` Postgres database and remove
  startup `Base.metadata.create_all`.
- Add Compose health inspection for an already-running deployment.
- Add gateway abuse protection with Redis-backed rate limiting.
- Document local reproduction of CI jobs.
- Add a project quickstart once the new observability/RAG workflows exist.

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

### Product And Agent Ideas

- Function-calling agent layer on top of current task routing.
- Web search as an optional tool.
- Broader user feedback UX after simple feedback is proven useful.
- More advanced cost accounting once ClickHouse inference analytics exists.
