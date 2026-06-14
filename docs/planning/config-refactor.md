# Configuration Refactor Proposal

This document proposes a cleaner configuration contract for the current
Compose / Python settings workflow.

The target is not "more env vars". The target is one clear model:

- network and infrastructure primitives are configured once;
- derived endpoints are built from those primitives;
- application runtime settings live in TOML, not in `.env`;
- container processes receive env from Compose and never read `.env` files;
- required configuration is explicit, not hidden in Python defaults.

## Locked Design Decisions

These choices are fixed for the refactor. Implementation should not introduce
temporary compatibility aliases, fallback defaults, or alternate naming schemes.

### Canonical Env Inventory

The project keeps one `.env.example`. It contains the environment contract:
deployment topology, host paths, secrets, credentials, public routing, and
native service/integration env names.

It does not contain non-secret application behavior. That behavior lives in
`config/runtime.toml`.

Deployment mechanics:

```env
DEPLOY__PROJECT_ROOT=/home/anton-m/agent-042/current
DEPLOY__ASSETS_ROOT=/home/anton-m/agent-042/assets
DEPLOY__ARTIFACTS_ROOT=/home/anton-m/agent-042/artifacts
DEPLOY__DVC_CONFIG_LOCAL_PATH=/home/anton-m/agent-042/.dvc/config.local
DEPLOY__RUNTIME_CONFIG_PATH=/home/anton-m/agent-042/config/runtime.toml
DEPLOY__CATALOG_CONFIG_PATH=/home/anton-m/agent-042/src/shared/catalog.toml
DEPLOY__IMAGE_TAG=local
COMPOSE_PROJECT_NAME=agent-042
```

Network coordinates:

```env
NETWORK__POSTGRES__INTERNAL_HOST=postgres
NETWORK__POSTGRES__INTERNAL_PORT=5432
NETWORK__POSTGRES__HOST_PORT=5432

NETWORK__MLFLOW__INTERNAL_HOST=mlflow
NETWORK__MLFLOW__SCHEME=http
NETWORK__MLFLOW__INTERNAL_PORT=5000
NETWORK__MLFLOW__HOST_PORT=5050

NETWORK__VLLM__INTERNAL_HOST=vllm
NETWORK__VLLM__SCHEME=http
NETWORK__VLLM__INTERNAL_PORT=8000
NETWORK__VLLM__HOST_PORT=8000

NETWORK__QDRANT_HTTP__INTERNAL_HOST=qdrant
NETWORK__QDRANT_HTTP__INTERNAL_PORT=6333
NETWORK__QDRANT_HTTP__HOST_PORT=6333
NETWORK__QDRANT_GRPC__INTERNAL_HOST=qdrant
NETWORK__QDRANT_GRPC__INTERNAL_PORT=6334
NETWORK__QDRANT_GRPC__HOST_PORT=6334

NETWORK__RABBITMQ_AMQP__INTERNAL_HOST=rabbitmq
NETWORK__RABBITMQ_AMQP__INTERNAL_PORT=5672
NETWORK__RABBITMQ_AMQP__HOST_PORT=5672
NETWORK__RABBITMQ_MGMT__INTERNAL_HOST=rabbitmq
NETWORK__RABBITMQ_MGMT__SCHEME=http
NETWORK__RABBITMQ_MGMT__INTERNAL_PORT=15672
NETWORK__RABBITMQ_MGMT__HOST_PORT=15672

NETWORK__REDIS__INTERNAL_HOST=redis
NETWORK__REDIS__INTERNAL_PORT=6379
NETWORK__REDIS__HOST_PORT=6379

NETWORK__REDPANDA_KAFKA__INTERNAL_HOST=redpanda
NETWORK__REDPANDA_KAFKA__INTERNAL_PORT=9092
NETWORK__REDPANDA_KAFKA__HOST_PORT=19092
NETWORK__REDPANDA_ADMIN__INTERNAL_HOST=redpanda
NETWORK__REDPANDA_ADMIN__SCHEME=http
NETWORK__REDPANDA_ADMIN__INTERNAL_PORT=9644
NETWORK__REDPANDA_ADMIN__HOST_PORT=19644
NETWORK__REDPANDA_SCHEMA_REGISTRY__INTERNAL_HOST=redpanda
NETWORK__REDPANDA_SCHEMA_REGISTRY__SCHEME=http
NETWORK__REDPANDA_SCHEMA_REGISTRY__INTERNAL_PORT=8081
NETWORK__REDPANDA_SCHEMA_REGISTRY__HOST_PORT=18081
NETWORK__REDPANDA_PANDAPROXY__INTERNAL_HOST=redpanda
NETWORK__REDPANDA_PANDAPROXY__SCHEME=http
NETWORK__REDPANDA_PANDAPROXY__INTERNAL_PORT=8082
NETWORK__REDPANDA_PANDAPROXY__HOST_PORT=18082
NETWORK__REDPANDA_CONSOLE__INTERNAL_HOST=redpanda-console
NETWORK__REDPANDA_CONSOLE__SCHEME=http
NETWORK__REDPANDA_CONSOLE__INTERNAL_PORT=8080
NETWORK__REDPANDA_CONSOLE__HOST_PORT=8081

NETWORK__EMBEDDINGS__INTERNAL_HOST=embeddings
NETWORK__EMBEDDINGS__SCHEME=http
NETWORK__EMBEDDINGS__INTERNAL_PORT=8100
NETWORK__RERANKER__INTERNAL_HOST=reranker
NETWORK__RERANKER__SCHEME=http
NETWORK__RERANKER__INTERNAL_PORT=8101

NETWORK__CELERY_WORKER__INTERNAL_HOST=celery-worker
NETWORK__CODE_SANDBOX__INTERNAL_HOST=code-sandbox
NETWORK__CODE_SANDBOX__SCHEME=http
NETWORK__CODE_SANDBOX__INTERNAL_PORT=8200

NETWORK__GATEWAY__INTERNAL_HOST=gateway
NETWORK__GATEWAY__SCHEME=http
NETWORK__GATEWAY__INTERNAL_PORT=9000
NETWORK__GATEWAY__HOST_PORT=9001
NETWORK__UI__INTERNAL_HOST=ui
NETWORK__UI__SCHEME=http
NETWORK__UI__INTERNAL_PORT=8501
NETWORK__UI__HOST_PORT=8501

NETWORK__FLOWER__INTERNAL_HOST=flower
NETWORK__FLOWER__SCHEME=http
NETWORK__FLOWER__INTERNAL_PORT=5555
NETWORK__FLOWER__HOST_PORT=5555
NETWORK__REDISINSIGHT__INTERNAL_HOST=redisinsight
NETWORK__REDISINSIGHT__SCHEME=http
NETWORK__REDISINSIGHT__INTERNAL_PORT=5540
NETWORK__REDISINSIGHT__HOST_PORT=5540

NETWORK__PROMETHEUS__INTERNAL_HOST=prometheus
NETWORK__PROMETHEUS__SCHEME=http
NETWORK__PROMETHEUS__INTERNAL_PORT=9090
NETWORK__PROMETHEUS__HOST_PORT=9090
NETWORK__CLICKHOUSE_HTTP__INTERNAL_HOST=clickhouse
NETWORK__CLICKHOUSE_HTTP__SCHEME=http
NETWORK__CLICKHOUSE_HTTP__INTERNAL_PORT=8123
NETWORK__CLICKHOUSE_HTTP__HOST_PORT=8123
NETWORK__CLICKHOUSE_NATIVE__INTERNAL_HOST=clickhouse
NETWORK__CLICKHOUSE_NATIVE__INTERNAL_PORT=9000
NETWORK__CLICKHOUSE_NATIVE__HOST_PORT=9000
NETWORK__LOKI__INTERNAL_HOST=loki
NETWORK__LOKI__SCHEME=http
NETWORK__LOKI__INTERNAL_PORT=3100
NETWORK__LOKI__HOST_PORT=3100
NETWORK__TEMPO__INTERNAL_HOST=tempo
NETWORK__TEMPO__SCHEME=http
NETWORK__TEMPO__INTERNAL_PORT=3200
NETWORK__TEMPO__HOST_PORT=3200
NETWORK__OTEL_COLLECTOR_GRPC__INTERNAL_HOST=otel-collector
NETWORK__OTEL_COLLECTOR_GRPC__INTERNAL_PORT=4317
NETWORK__OTEL_COLLECTOR_GRPC__HOST_PORT=4317
NETWORK__OTEL_COLLECTOR_HTTP__INTERNAL_HOST=otel-collector
NETWORK__OTEL_COLLECTOR_HTTP__SCHEME=http
NETWORK__OTEL_COLLECTOR_HTTP__INTERNAL_PORT=4318
NETWORK__OTEL_COLLECTOR_HTTP__HOST_PORT=4318
NETWORK__ALLOY__INTERNAL_HOST=alloy
NETWORK__ALLOY__SCHEME=http
NETWORK__ALLOY__INTERNAL_PORT=12345
NETWORK__ALLOY__HOST_PORT=12345
NETWORK__GRAFANA__INTERNAL_HOST=grafana
NETWORK__GRAFANA__SCHEME=http
NETWORK__GRAFANA__INTERNAL_PORT=3000
NETWORK__GRAFANA__HOST_PORT=3000

NETWORK__AIRFLOW_WEBSERVER__INTERNAL_HOST=airflow-webserver
NETWORK__AIRFLOW_WEBSERVER__SCHEME=http
NETWORK__AIRFLOW_WEBSERVER__INTERNAL_PORT=8080
NETWORK__AIRFLOW_WEBSERVER__HOST_PORT=8080
NETWORK__AIRFLOW_SCHEDULER_HEALTH__INTERNAL_HOST=airflow-scheduler
NETWORK__AIRFLOW_SCHEDULER_HEALTH__SCHEME=http
NETWORK__AIRFLOW_SCHEDULER_HEALTH__INTERNAL_PORT=8974
NETWORK__JUPYTER__INTERNAL_HOST=jupyter
NETWORK__JUPYTER__SCHEME=http
NETWORK__JUPYTER__INTERNAL_PORT=8888
NETWORK__JUPYTER__HOST_PORT=8888
```

Public routing:

```env
PUBLIC__BASE_URL=https://agent.antonlab.ru:8443
PUBLIC__AUTH_CALLBACK_PATH=/auth/callback
PUBLIC__AIRFLOW_PATH=/airflow
PUBLIC__GRAFANA_PATH=/grafana
PUBLIC__FLOWER_PATH=/flower
PUBLIC__REDISINSIGHT_PATH=/redis-insight
PUBLIC__RABBITMQ_MGMT_PATH=/rabbitmq
PUBLIC__JUPYTER_PATH=/jupyter
```

Service-owned infrastructure state and credentials:

```env
POSTGRES_DB=mlflow
POSTGRES_AIRFLOW_DB=airflow
POSTGRES_APP_DB=agent042
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=your-postgres-password-here

MLFLOW_SERVER_ALLOWED_HOSTS=agent.antonlab.ru,agent.antonlab.ru:8443,antonlab.ru,localhost:5050,127.0.0.1:5050,127.0.0.1,mlflow,mlflow:5000
MLFLOW_ARTIFACT_ROOT=s3://agent-042-mlflow-artifacts/mlflow
MLFLOW_TRACKING_USERNAME=your-username-here
MLFLOW_TRACKING_PASSWORD=your-password-here

RABBITMQ_DEFAULT_USER=agent
RABBITMQ_DEFAULT_PASS=your-rabbitmq-password-here

CLICKHOUSE_DB=agent042_analytics
CLICKHOUSE_USER=agent042
CLICKHOUSE_PASSWORD=your-clickhouse-password-here

AIRFLOW__CORE__FERNET_KEY=your-fernet-key-here
AIRFLOW__API_AUTH__JWT_SECRET=your-jwt-secret-here
AIRFLOW__API_AUTH__JWT_ISSUER=airflow
AIRFLOW_ADMIN_USER=admin
AIRFLOW_ADMIN_PASSWORD=your-airflow-password-here

JUPYTER_TOKEN=your-jupyter-token-here
GF_SECURITY_ADMIN_PASSWORD=your-grafana-password-here
```

Native external integration envs:

```env
MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
AWS_DEFAULT_REGION=ru-central1
AWS_ACCESS_KEY_ID=your-access-key-here
AWS_SECRET_ACCESS_KEY=your-secret-key-here
GITHUB_REPOSITORY=your-org-or-user/agent-042
GITHUB_DATA_SYNC_TOKEN=your-github-token-here
OTEL_TRACES_SAMPLER_ARG=1.0
```

Runtime secrets:

```env
GATEWAY__API_KEY=

AUTH__GOOGLE_CLIENT_ID=your-google-client-id-here
AUTH__GOOGLE_CLIENT_SECRET=your-google-client-secret-here
AUTH__SESSION_SECRET_KEY=your-session-secret-key-here
AUTH__INTERNAL_API_KEY=your-internal-api-key-here

EVAL__JUDGE__API_KEY=
```

### Runtime TOML Inventory

The project keeps one canonical runtime policy file:

```text
config/runtime.toml
```

It contains non-secret application behavior. Missing fields are validation
errors; Python must not fill them from defaults.

```toml
schema_version = 1

[vllm]
model = "/models/Qwen/Qwen3-0.6B"
dtype = "float16"
quantization = "bitsandbytes"
gpu_utilization = 0.7
gpu_count = 1
max_num_seqs = 1
max_num_batched_tokens = 1024
kv_cache_dtype = "fp8"
max_loras = 4
max_lora_rank = 16
allow_runtime_lora_updating = true

[gateway]
vllm_timeout = 60.0
repetition_penalty = 1.1
streaming_timeout = 300.0
embeddings_timeout = 120.0
async_enabled = true
cors_allow_origins = ["*"]
service_name = "agent-042-gateway"

[gateway.budget]
model_max_tokens = 32768
chars_per_token = 4.0
budget_guard = 512
budget_system = 768
budget_turn = 10240
min_budget_history = 4096
budget_rag = 6144
min_response_budget = 256

[rag]
enabled = true
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embedding_device = "cpu"
kb_selection_threshold = 0.3
task_classification_threshold = 0.0
strict_startup = false
sparse_encoder_model = "Qdrant/bm25"
reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

[rag.build]
embedding_batch_size = 32
qdrant_upsert_batch_size = 128

[auth]
google_discovery_url = "https://accounts.google.com/.well-known/openid-configuration"
session_ttl_seconds = 86400

[adapter_registry]
adapters_dir = "./assets/adapters"
production_alias = ""
sync_aliases = ["champion", "challenger"]
auto_sync = false

[events]
inference_topic = "inference.events.v1"

[eval.judge]
backend = "local_vllm"
model = "/models/Qwen/Qwen3-0.6B"
base_url = ""
timeout = 60.0
request_delay_seconds = 0.0

[eval.metrics]
bert_score_model = "microsoft/deberta-v3-base"
temperature = 0.0
max_completion_tokens = 2048

[eval.sandbox]
code_exec_timeout = 30
code_exec_mem_limit = "512m"
code_exec_cpus = 1.0

[ui]
health_timeout = 10.0
models_timeout = 30.0
chat_timeout = 300.0

[worker]
default_timeout = 300
max_retries = 3
retry_delay = 5
pool = "prefork"
concurrency = 2
send_task_events = true
cancel_long_running_tasks_on_connection_loss = true
```

### Native Env Whitelist

Only these native or native-shaped env names remain canonical:

```text
POSTGRES_DB
POSTGRES_AIRFLOW_DB
POSTGRES_APP_DB
POSTGRES_USER
POSTGRES_PASSWORD
MLFLOW_SERVER_ALLOWED_HOSTS
MLFLOW_ARTIFACT_ROOT
MLFLOW_S3_ENDPOINT_URL
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
AWS_DEFAULT_REGION
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
RABBITMQ_DEFAULT_USER
RABBITMQ_DEFAULT_PASS
CLICKHOUSE_DB
CLICKHOUSE_USER
CLICKHOUSE_PASSWORD
AIRFLOW__CORE__FERNET_KEY
AIRFLOW__API_AUTH__JWT_SECRET
AIRFLOW__API_AUTH__JWT_ISSUER
AIRFLOW_ADMIN_USER
AIRFLOW_ADMIN_PASSWORD
JUPYTER_TOKEN
GF_SECURITY_ADMIN_PASSWORD
GITHUB_REPOSITORY
GITHUB_DATA_SYNC_TOKEN
OTEL_TRACES_SAMPLER_ARG
COMPOSE_PROJECT_NAME
```

Compose may still export additional third-party adapter names such as
`AIRFLOW__DATABASE__SQL_ALCHEMY_CONN`, `AIRFLOW__CELERY__BROKER_URL`,
`CELERY_BROKER_URL`, or `MLFLOW_TRACKING_URI` when a specific image, SDK, or
CLI expects that spelling. Those extra names are adapter outputs, not canonical
`.env` inputs.

### Compatibility Policy

No legacy compatibility envs are kept. The refactor is allowed to be a breaking
configuration change.

Remove these old canonical inputs entirely:

```text
PROJECT_ROOT
ASSETS_ROOT
ARTIFACTS_ROOT
DVC_CONFIG_LOCAL_PATH
IMAGE_TAG
POSTGRES_PORT
MLFLOW_PORT
MLFLOW_BACKEND_URI
VLLM_MODEL
VLLM_PORT
QDRANT_PORT
QDRANT_GRPC_PORT
GATEWAY_PORT
GATEWAY__DEFAULT_MODEL
GATEWAY__URL
AUTH__GOOGLE_REDIRECT_URI
AUTH__AGENT042_DB_URL
RABBITMQ_PORT
RABBITMQ_MGMT_PORT
RABBITMQ_USER
RABBITMQ_PASS
REDIS_PORT
UI_PORT
FLOWER_PORT
REDISINSIGHT_PORT
REDPANDA_*_PORT
CLICKHOUSE_HTTP_PORT
CLICKHOUSE_NATIVE_PORT
AIRFLOW_PORT
AIRFLOW_DB
AIRFLOW_FERNET_KEY
AIRFLOW_JWT_SECRET
AIRFLOW_JWT_ISSUER
JUPYTER_PORT
GRAFANA_PORT
GRAFANA_ADMIN_PASSWORD
PROMETHEUS_PORT
LOKI_PORT
TEMPO_PORT
ALLOY_PORT
OTEL_COLLECTOR_*_PORT
PLATFORM__*
VLLM__*
GATEWAY__VLLM_TIMEOUT
GATEWAY__REPETITION_PENALTY
GATEWAY__STREAMING_TIMEOUT
GATEWAY__EMBEDDINGS_TIMEOUT
GATEWAY__ASYNC_ENABLED
GATEWAY__CORS_ALLOW_ORIGINS
GATEWAY__SERVICE_NAME
GATEWAY__BUDGET__*
RAG__*
AUTH__GOOGLE_DISCOVERY_URL
AUTH__SESSION_TTL_SECONDS
CATALOG__PATH
ADAPTER_REGISTRY__*
EVENTS__INFERENCE_TOPIC
EVAL__JUDGE__BACKEND
EVAL__JUDGE__MODEL
EVAL__JUDGE__BASE_URL
EVAL__JUDGE__TIMEOUT
EVAL__JUDGE__REQUEST_DELAY_SECONDS
EVAL__METRICS__*
EVAL__SANDBOX__*
UI__*
WORKER__*
```

### Model Ownership

- `[vllm].model` in `config/runtime.toml` is the model served by the local vLLM
  container.
- `GATEWAY__DEFAULT_MODEL` is removed. When a request omits a model, gateway
  uses `[vllm].model`; there is no second setting for the same local model.
- `[eval.judge].model` remains independent. It may equal `[vllm].model` for a
  local judge, but no fallback is allowed.
- `[eval.judge].base_url` is required only when
  `[eval.judge].backend="openai_compatible"`. For `local_vllm`, judge transport
  uses `NETWORK__VLLM__...`.

Other project-owned collisions to remove:

- `GATEWAY__URL`: derive host/internal gateway URLs from `NETWORK__GATEWAY`.
- `PLATFORM__VLLM_BASE_URL`, `PLATFORM__MLFLOW_TRACKING_URI`,
  `PLATFORM__EMBEDDINGS_URL`, `RAG__RERANKER_URL`, `PLATFORM__REDIS_URL`,
  `PLATFORM__CELERY_BROKER_URL`, and
  `PLATFORM__KAFKA_BOOTSTRAP_SERVERS`: derive from `NETWORK__...`,
  native service credentials, and runtime TOML.
- `EVAL__DB_URL`, `AUTH__AGENT042_DB_URL`, `MLFLOW_BACKEND_URI`, and
  `DATABASE_URL`: derive from `POSTGRES_*` and
  `NETWORK__POSTGRES__...`.
- `GF_SERVER_ROOT_URL`, `AIRFLOW__API__BASE_URL`, OAuth redirect URI, Flower
  prefix, RedisInsight prefix, and Jupyter base URL: derive from `PUBLIC__...`.

### No Defaults

No operator-setting defaults remain in `docker-compose.yaml` or
`src/shared/config.py`.

In Compose, this means no fallback expressions such as `${NAME:-value}` for
canonical values. Fixed container contracts may still be literal values when
they are not operator configuration.

All current `Field(default=...)` and `Field(default_factory=...)` values in
`src/shared/config.py` must either:

- become required env-backed or TOML-backed settings listed above; or
- move out of `Settings` entirely if they are not operator configuration.

Non-operator constants may still exist in normal code, templates, or domain
artifacts. They must not masquerade as configurable settings with hidden
defaults in `Settings`.

Today, `config.py` contains many defaults that do not rely on `.env`, TOML, or
any external config source: endpoint defaults, prompt budgets, timeouts,
feature flags, model ids, RAG thresholds, auth placeholders, adapter registry
paths/aliases, eval controls, UI timeouts, and worker controls. The refactor
must eliminate that hidden configuration layer.

Current hidden-default groups to eliminate:

- `PlatformSettings`: all endpoint, Qdrant, Redis, Celery, Kafka, MLflow, and
  inference topic defaults.
- `GatewayConfig` / `BudgetSettings`: model, API key, timeouts, CORS, service
  name, gateway URL, and budget values.
- `RagSettings` / `RagBuildSettings`: enable flag, embedding/reranker models,
  device, batch sizes, thresholds, strict startup flag, and reranker URL.
- `AuthSettings`: OAuth placeholders, redirect URI, auth DB URL, session
  secret, session TTL, and internal API key.
- `CatalogConfig` / `AdapterRegistryConfig`: catalog path, adapters directory,
  production alias, sync aliases, and auto-sync flag.
- `EvalConfig`: judge backend, judge model, external judge URL/API key,
  judge timeouts, metric controls, sandbox limits, and eval DB URL.
- `UIConfig` and `WorkerConfig`: all request timeout, retry, pool,
  concurrency, task-event, and connection-loss controls.

### Render Step Before Compose

The render step is a required pre-Compose command that materializes config files
and Compose interpolation env files for tools that cannot read runtime TOML
directly.

Locked choice:

- Add `scripts/render_configs.py`.
- The script reads the same `.env` file passed to Compose.
- The script reads `config/runtime.toml` through
  `DEPLOY__RUNTIME_CONFIG_PATH`.
- It writes generated files under `artifacts/generated/`.
- Compose mounts generated files, not source templates.
- Source templates live in `infra/templates/`.
- Generated adapter env files are inputs to Compose interpolation only. They
  are not canonical operator inputs and must not be attached to services with
  service-level `env_file:`.

Initial generated files:

```text
artifacts/generated/compose/runtime.env
artifacts/generated/prometheus/prometheus.yml
artifacts/generated/clickhouse/001_inference_events.sql
```

Generated Compose interpolation variables from runtime TOML must use the
`COMPOSE__...` prefix. They are build/deploy adapter values, not settings read
by Python application code and not operator-maintained inputs. Examples:

```env
COMPOSE__VLLM__MODEL=/models/Qwen/Qwen3-0.6B
COMPOSE__VLLM__DTYPE=float16
COMPOSE__EVENTS__INFERENCE_TOPIC=inference.events.v1
```

Deploy and local startup commands must run:

```bash
python scripts/render_configs.py --env-file .env
docker compose \
  --env-file .env \
  --env-file artifacts/generated/compose/runtime.env \
  -f infra/compose/docker-compose.yaml \
  up -d
```

No generated file should be edited manually.

Compose `--env-file` is used for interpolation. It does not mean containers
receive the whole `.env` file; container env remains explicit under each
service's `environment:` block or top-level `x-*` fragment.

### Compose `x-*` Fragment List

Use these top-level fragments:

```text
x-network-env
x-config-path-env
x-config-volumes
x-observability-env
x-s3-client-env
x-rabbitmq-client-env
x-postgres-client-env
x-redpanda-client-env
x-runtime-env
x-vllm-adapter-env
x-airflow-common-env
x-airflow-common-build
x-airflow-common-image
x-airflow-worker-build
x-airflow-worker-image
x-airflow-worker-gpu-build
x-airflow-worker-gpu-image
x-rag-ops-build
x-rag-ops-image
x-airflow-common-volumes
x-default-logging
```

Repeated mappings go into these fragments. One-off service-specific values stay
inside the service block.

Fragment contents are fixed:

- `x-network-env`: one-to-one pass-through of every `NETWORK__...` key in the
  canonical inventory.
- `x-config-path-env`: container-local config paths, for example
  `CONFIG__RUNTIME_PATH=/opt/agent/config/runtime.toml` and
  `CONFIG__CATALOG_PATH=/opt/agent/config/catalog.toml`.
- `x-config-volumes`: read-only mounts from `DEPLOY__RUNTIME_CONFIG_PATH` to
  `/opt/agent/config/runtime.toml` and from `DEPLOY__CATALOG_CONFIG_PATH` to
  `/opt/agent/config/catalog.toml`. Every Python service, Airflow service or
  worker, RAG ops runner, and notebook container that reads project settings
  must use this fragment.
- `x-observability-env`: `OTEL_EXPORTER_OTLP_ENDPOINT` derived from
  `NETWORK__OTEL_COLLECTOR_GRPC__...`, `OTEL_EXPORTER_OTLP_PROTOCOL=grpc`,
  `OTEL_TRACES_SAMPLER=parentbased_traceidratio`, and
  `OTEL_TRACES_SAMPLER_ARG`.
- `x-s3-client-env`: `MLFLOW_S3_ENDPOINT_URL`, `AWS_DEFAULT_REGION`,
  `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY`.
- `x-rabbitmq-client-env`: `RABBITMQ_DEFAULT_USER`,
  `RABBITMQ_DEFAULT_PASS`, and `NETWORK__RABBITMQ_AMQP__...`.
- `x-postgres-client-env`: `POSTGRES_DB`, `POSTGRES_AIRFLOW_DB`,
  `POSTGRES_APP_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, and
  `NETWORK__POSTGRES__...`.
- `x-redpanda-client-env`: `NETWORK__REDPANDA_KAFKA__...`. Event topic names
  stay in `config/runtime.toml`; `COMPOSE__EVENTS__INFERENCE_TOPIC` is
  generated only for Compose interpolation or generated non-Python config
  files that need the topic string.
- `x-runtime-env`: `GATEWAY__API_KEY`, `AUTH__GOOGLE_CLIENT_ID`,
  `AUTH__GOOGLE_CLIENT_SECRET`, `AUTH__SESSION_SECRET_KEY`,
  `AUTH__INTERNAL_API_KEY`, and `EVAL__JUDGE__API_KEY`.
- `x-vllm-adapter-env`: vLLM image env and command values generated from
  `[vllm]` in `config/runtime.toml`. Generated interpolation variable names
  must use the `COMPOSE__VLLM__...` prefix, for example
  `COMPOSE__VLLM__MODEL`, `COMPOSE__VLLM__DTYPE`,
  `COMPOSE__VLLM__GPU_COUNT`, and `COMPOSE__VLLM__MAX_LORAS`. The vLLM
  service passes model, dtype, GPU count, scheduler caps, KV-cache dtype,
  quantization, and LoRA caps as command args from `COMPOSE__VLLM__...`
  variables. The only runtime-TOML-derived vLLM env is
  `VLLM_ALLOW_RUNTIME_LORA_UPDATING`, assigned from
  `COMPOSE__VLLM__ALLOW_RUNTIME_LORA_UPDATING` in the vLLM service block.
- `x-airflow-common-env`: Airflow-native adapter exports derived from
  native service envs, `NETWORK__...`, `PUBLIC__...`, runtime TOML,
  generated adapter env, and runtime secrets, plus
  `CONTAINER__PROJECT_ROOT=/opt/airflow/project`. It owns Airflow's native
  names such as `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN`,
  `AIRFLOW__CELERY__BROKER_URL`, `AIRFLOW__CORE__FERNET_KEY`,
  `AIRFLOW__API_AUTH__JWT_SECRET`, and
  `AIRFLOW__CORE__EXECUTION_API_SERVER_URL`.
- `x-airflow-common-build`: build context and Dockerfile for the common
  Airflow image, using `DEPLOY__PROJECT_ROOT`.
- `x-airflow-common-image`: the common Airflow image tag, using
  `GITHUB_REPOSITORY` and `DEPLOY__IMAGE_TAG`.
- `x-airflow-worker-build`: build context and Dockerfile for the CPU worker.
- `x-airflow-worker-image`: the CPU worker image tag.
- `x-airflow-worker-gpu-build`: build context and Dockerfile for the GPU
  worker.
- `x-airflow-worker-gpu-image`: the GPU worker image tag.
- `x-rag-ops-build`: build context and Dockerfile for the RAG ops runner.
- `x-rag-ops-image`: the RAG ops image tag.
- `x-airflow-common-volumes`: Airflow and RAG ops mounts derived from
  `DEPLOY__PROJECT_ROOT`, `DEPLOY__ASSETS_ROOT`,
  `DEPLOY__ARTIFACTS_ROOT`, and `DEPLOY__DVC_CONFIG_LOCAL_PATH`.
- `x-default-logging`: shared Docker logging driver/options.

Do not introduce additional top-level `x-*` fragments during implementation
without updating this document first.

### No Deferred Design Choices

The implementation must follow these rules without introducing new choices:

- The canonical `.env.example` inventory above is complete. Any new operator
  setting requires a document update before implementation.
- The canonical `config/runtime.toml` inventory above is complete. Any new
  runtime setting requires a document update before implementation.
- Native env names are allowed only if they are listed in the whitelist above.
- If a stable native env name exists and correctly describes the value, use it
  as canonical instead of inventing a nested project alias.
- No old compatibility envs are accepted.
- No Compose fallback expressions such as `${NAME:-value}` are allowed for
  canonical values.
- No `Settings` field may invent an operator value with a Python default.
- No generic `settings.network.url(...)` helper is allowed. Call sites must use
  explicit `host_url(...)` or `internal_url(...)` helpers.
- Project-owned DB URLs, broker URLs, and service URLs are derived values, not
  canonical env inputs.
- Generic Python application code must not load `.env`.
- Container entrypoints, DAGs, services, and shared library code must consume
  process env and mounted config files only.
- Host-side scripts that read `.env` must live under a host/deploy script
  boundary and must be explicitly named as host-side entrypoints.
- Do not use service-level Compose `env_file:` to pass the whole `.env` into
  containers. Every container env must be mapped explicitly in Compose or an
  `x-*` fragment.

## Problem 1: Endpoint Coordinates Are Encoded As Duplicated URL Strings

Current symptoms:

- `PLATFORM__VLLM_BASE_URL`, `PLATFORM__EMBEDDINGS_URL`,
  `RAG__RERANKER_URL`, `GATEWAY__URL`, and
  `PLATFORM__MLFLOW_TRACKING_URI` store full URLs.
- Those URLs duplicate primitives that already exist elsewhere:
  service name, scheme, internal port, host port, and sometimes bind policy.
- `VLLM_PORT` and similar names usually mean "host-published port", while the
  actual container listen port is hardcoded separately.
- Internal ports and service names are repeated in Compose command args,
  healthchecks, port mappings, Prometheus config, SQL init files, and Python
  settings.
- Host-side clients need `localhost:<host-port>`, while containers need
  `<compose-service-name>:<internal-port>`.

The root problem is not `localhost` vs `vllm`. Those are real network locations.
The problem is that manually maintained full URLs are treated as primary
configuration.

### Solution: Introduce A Canonical Network Namespace And Derive Endpoints

Keep the project's nested-env convention, but move project-owned service
coordinates into a dedicated `NETWORK__...` namespace instead of mixing them
into `PLATFORM__*`, `GATEWAY__*`, and `RAG__*` runtime settings.

Proposed naming:

```env
NETWORK__VLLM__INTERNAL_HOST=vllm
NETWORK__VLLM__SCHEME=http
NETWORK__VLLM__INTERNAL_PORT=8000
NETWORK__VLLM__HOST_PORT=8000

NETWORK__GATEWAY__INTERNAL_HOST=gateway
NETWORK__GATEWAY__SCHEME=http
NETWORK__GATEWAY__INTERNAL_PORT=9000
NETWORK__GATEWAY__HOST_PORT=9001

NETWORK__MLFLOW__INTERNAL_HOST=mlflow
NETWORK__MLFLOW__SCHEME=http
NETWORK__MLFLOW__INTERNAL_PORT=5000
NETWORK__MLFLOW__HOST_PORT=5050

NETWORK__QDRANT_HTTP__INTERNAL_HOST=qdrant
NETWORK__QDRANT_HTTP__INTERNAL_PORT=6333
NETWORK__QDRANT_HTTP__HOST_PORT=6333

NETWORK__QDRANT_GRPC__INTERNAL_HOST=qdrant
NETWORK__QDRANT_GRPC__INTERNAL_PORT=6334
NETWORK__QDRANT_GRPC__HOST_PORT=6334
```

Rules:

- Use `INTERNAL_PORT` for the port the service listens on inside Docker.
- Use `HOST_PORT` for the port published on the server loopback interface.
- Use `INTERNAL_HOST` for the internal Docker DNS hostname.
- Use `SCHEME` only for URL-based services.
- Avoid `INNER` / `OUTER`: "outer" could mean host loopback, LAN, public
  internet, reverse proxy, or a cloud load balancer.
- Avoid `DOCKER_PORT`: Docker has both container ports and host-published
  ports.
- Keep `127.0.0.1` hardcoded in Compose for host-published ports. Do not add a
  host-bind env variable in this refactor.

Endpoint resolver behavior should be explicit at the call site:

```text
internal_url(service) = scheme://internal_host:internal_port
host_url(service)     = scheme://localhost:host_port
```

Example derived values:

```text
host vLLM URL:   http://localhost:8000
internal vLLM URL: http://vllm:8000
```

Implementation shape:

- Add a `NetworkSettings` tree to `src/shared/config.py`.
- Add explicit computed endpoint helpers such as:
  - `settings.network.host_url("vllm")`
  - `settings.network.internal_url("vllm")`
- Replace manually configured endpoint env vars with computed properties.
- Docker services call internal endpoint helpers.
- Host-side scripts call host endpoint helpers.
- Do not add a generic `settings.network.url("vllm")` helper.
- `.env.example` may keep network primitives because they are not derived
  endpoints; it should not keep full internal URLs.

Full URL settings should remain only for endpoints that are genuinely external
or not derivable from local network primitives:

- public application base URL
- S3 / object storage endpoint
- external OpenAI-compatible judge URL

## Problem 2: Config Sources Have No Single Taxonomy

Current symptoms:

- `PLATFORM__*`, `GATEWAY__*`, `RAG__*`, and `EVAL__*` currently contain a mix
  of application behavior and infrastructure wiring.
- The nested settings convention is useful, but it became blurry because
  network and infrastructure concerns were stored under runtime-looking names.
- Some canonical project values are nested while others are flat
  (`POSTGRES_PASSWORD`, `RABBITMQ_PASS`, `MLFLOW_S3_ENDPOINT_URL`,
  `AWS_ACCESS_KEY_ID`, and so on).
- Third-party container env names are mixed directly into the project-level
  operator contract.

### Solution: Use Native Env Names First

Canonical env vars should keep the original upstream/native name when the
value has a clear owner in an external tool, SDK, provider, container image, or
orchestrator. Project-owned nested names are for values the project invented
itself, such as deployment paths, network topology, public routes, and runtime
secrets. Non-secret runtime behavior belongs in `config/runtime.toml`.

Canonical env roots:

- `DEPLOY__...`: deployment mechanics and host filesystem roots.
- `NETWORK__...`: internal host, internal port, host port, and scheme for
  project-owned services.
- `PUBLIC__...`: project-owned public routing primitives.
- Runtime secret envs: `GATEWAY__API_KEY`, `AUTH__...` secrets, and
  `EVAL__JUDGE__API_KEY`.
- Native service/integration envs: well-known variables required by external
  SDKs, providers, images, or orchestrators, for example
  `POSTGRES_PASSWORD`, `RABBITMQ_DEFAULT_PASS`,
  `GF_SECURITY_ADMIN_PASSWORD`, `AWS_ACCESS_KEY_ID`, or
  `MLFLOW_S3_ENDPOINT_URL`.

Canonical runtime TOML tables:

- `[vllm]`: served model and vLLM engine/runtime controls.
- `[gateway]` and `[gateway.budget]`: request handling and prompt budget
  policy.
- `[rag]` and `[rag.build]`: RAG behavior, embedding/reranker models, and
  build batch controls.
- `[auth]`: non-secret auth policy such as discovery URL and session TTL.
- `[adapter_registry]`: adapter sync behavior.
- `[events]`: project event topic names.
- `[eval.*]`: judge, metric, and sandbox behavior.
- `[ui]`: UI client timeouts.
- `[worker]`: worker behavior.

Deployment mechanics:

```env
DEPLOY__PROJECT_ROOT=/home/anton-m/agent-042/current
DEPLOY__ASSETS_ROOT=/home/anton-m/agent-042/assets
DEPLOY__ARTIFACTS_ROOT=/home/anton-m/agent-042/artifacts
DEPLOY__DVC_CONFIG_LOCAL_PATH=/home/anton-m/agent-042/.dvc/config.local
DEPLOY__RUNTIME_CONFIG_PATH=/home/anton-m/agent-042/config/runtime.toml
DEPLOY__CATALOG_CONFIG_PATH=/home/anton-m/agent-042/src/shared/catalog.toml
DEPLOY__IMAGE_TAG=local
COMPOSE_PROJECT_NAME=agent-042
```

Network coordinates:

```env
NETWORK__VLLM__INTERNAL_HOST=vllm
NETWORK__VLLM__INTERNAL_PORT=8000
NETWORK__VLLM__HOST_PORT=8000
```

Application runtime behavior:

```toml
[vllm]
model = "/models/Qwen/Qwen3-0.6B"

[gateway]
async_enabled = true

[rag]
enabled = true
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

[eval.judge]
backend = "local_vllm"
```

Service-owned infrastructure state and credentials:

```env
POSTGRES_DB=mlflow
POSTGRES_AIRFLOW_DB=airflow
POSTGRES_APP_DB=agent042
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=...

RABBITMQ_DEFAULT_USER=agent
RABBITMQ_DEFAULT_PASS=...
```

Runtime-owned secrets:

```env
AUTH__SESSION_SECRET_KEY=...
AUTH__INTERNAL_API_KEY=...
```

External integrations:

```env
PUBLIC__BASE_URL=https://agent.antonlab.ru:8443
PUBLIC__AUTH_CALLBACK_PATH=/auth/callback

MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
AWS_DEFAULT_REGION=ru-central1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

GITHUB_REPOSITORY=your-org-or-user/agent-042
GITHUB_DATA_SYNC_TOKEN=...
```

Rule of thumb:

- If a value describes where a project-owned service lives, put its primitives
  under `NETWORK__...`.
- If a value has an upstream/native env name, keep that name as canonical.
- If several native names need the same value, pick the native name closest to
  the owning service as canonical and derive the other native names in Compose.
- If a native name is misleading for this project, use the service owner prefix
  instead of pretending the upstream variable has the right meaning. Example:
  keep `POSTGRES_DB` for the Postgres image's default/bootstrap DB, but use
  `POSTGRES_AIRFLOW_DB` and `POSTGRES_APP_DB` for additional project-created
  databases.
- If a value changes non-secret application behavior, put it in
  `config/runtime.toml` under the owning table.
- If a value is a runtime secret, keep it in `.env` under the owning nested env
  namespace.
- If a value is a stable, well-known env contract for an external SDK,
  provider, or tool, use that native name directly.
- If a value is project-specific but public-facing, put it under a clear
  project namespace such as `PUBLIC__...`.

Adapter boundary examples:

```yaml
postgres:
  environment:
    POSTGRES_DB: ${POSTGRES_DB}
    POSTGRES_USER: ${POSTGRES_USER}
    POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}

rabbitmq:
  environment:
    RABBITMQ_DEFAULT_USER: ${RABBITMQ_DEFAULT_USER}
    RABBITMQ_DEFAULT_PASS: ${RABBITMQ_DEFAULT_PASS}

mlflow:
  environment:
    MLFLOW_S3_ENDPOINT_URL: ${MLFLOW_S3_ENDPOINT_URL}
    AWS_DEFAULT_REGION: ${AWS_DEFAULT_REGION}
    AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
    AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}
```

Native names may appear:

- as canonical envs when they are the native contract of an external SDK,
  provider, image, tool, or orchestrator;
- inside Compose `environment:` blocks as adapter outputs for third-party
  images, SDKs, or CLIs;
- for native orchestrator control variables such as `COMPOSE_PROJECT_NAME`.

Do not define both a native external env and a project alias for the same
source value. For example, use `MLFLOW_S3_ENDPOINT_URL` directly instead of
also introducing `EXTERNAL__OBJECT_STORAGE__ENDPOINT_URL`. Derived values are
different: `MLFLOW_TRACKING_URI` points to project-owned MLflow and should be
derived from `NETWORK__MLFLOW__...` when a specific MLflow client invocation
requires that native env name. It is not a canonical input.

This keeps nested naming for project-created values while avoiding a pointless
wrapper around native env names that operators already know.

## Problem 3: Defaults Create Hidden Sources Of Truth

Current symptoms:

- `config.py` contains network defaults such as `http://localhost:8000`,
  `localhost`, and `redis://localhost:6379/0`.
- Compose injects Docker-network values on top.
- Some defaults are host-oriented, while `RAG__RERANKER_URL` is Docker-oriented.
- The active config can come from Python defaults, `.env`, runtime TOML,
  Compose defaults, service-level Compose env, CLI flags, or notebooks mutating
  `os.environ`.

### Solution: No Defaults For Settings Values

Required settings must be explicit.

Rules:

- No Python defaults for project-owned service locations.
- No Python defaults for credentials.
- No Python defaults for ports that Compose also controls.
- No Python defaults for runtime behavior settings such as model ids, timeouts,
  budgets, feature flags, retries, aliases, or thresholds.
- Values that are not operator configuration should be removed from `Settings`
  instead of being kept there with defaults.

Implementation:

- Remove all operator-setting defaults from `config.py`.
- Replace them with required fields plus validation.
- Add a config validation command or test that loads the intended `.env` and
  runtime TOML profile and fails before services start if required values are
  missing.
- Make `.env.example` explicit enough that a copied file is complete for local
  checkout use.
- Make `config/runtime.toml` explicit enough that it is a complete runtime
  policy file.
- Remove Compose fallback expressions for canonical values.

This means canonical files decide configuration: `.env` for deployment/secrets
and runtime TOML for behavior. Python settings validate them; they do not
invent missing values.

## Problem 4: MLflow Uses Multiple Client Contracts

Current symptoms:

- Some training and notebook code reads `MLFLOW_TRACKING_URI`.
- Shared settings use `PLATFORM__MLFLOW_TRACKING_URI`.
- Compose injects `http://mlflow:5000`.
- `.env.example` includes MLflow username/password but not a tracking URI.

### Solution: Route MLflow Through The Same Network Resolver

MLflow is project-owned infrastructure in this deployment, so its local
tracking URI should be derived from network primitives:

```env
NETWORK__MLFLOW__INTERNAL_HOST=mlflow
NETWORK__MLFLOW__SCHEME=http
NETWORK__MLFLOW__INTERNAL_PORT=5000
NETWORK__MLFLOW__HOST_PORT=5050
```

Derived values:

```text
host tracking URI:   http://localhost:5050
internal tracking URI: http://mlflow:5000
```

Implementation:

- Make shared settings expose a canonical MLflow tracking URI through network
  primitives.
- Migrate project code that reads `MLFLOW_TRACKING_URI` directly to shared
  settings or a tiny MLflow resolver.
- Export `MLFLOW_TRACKING_URI` only inside the boundary that invokes an MLflow
  client, CLI, SDK, or third-party container requiring that native name.
- Keep `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` as MLflow
  client auth env vars.
- Do not ask operators to maintain both `MLFLOW_TRACKING_URI` and
  `PLATFORM__MLFLOW_TRACKING_URI`.

## Problem 5: Hardcoded Duplicates Exist Outside Python Settings

Current symptoms:

- vLLM `8000` appears in command, port mapping, healthcheck, shared endpoint,
  and Prometheus.
- Gateway `9000`, MLflow `5000`, Redpanda `9092`, and the inference events
  topic have similar duplication.
- ClickHouse init SQL hardcodes `redpanda:9092` and
  `inference.events.v1`.

### Solution: Every Configurable Primitive Must Feed Every Consumer

If a value is configurable, all consumers must read it from the same primitive.
If a value is not configurable, do not expose an env knob for it.

Compose examples:

```yaml
command:
  - "--port"
  - "${NETWORK__VLLM__INTERNAL_PORT}"

ports:
  - "127.0.0.1:${NETWORK__VLLM__HOST_PORT}:${NETWORK__VLLM__INTERNAL_PORT}"

healthcheck:
  test:
    [
      "CMD",
      "curl",
      "-f",
      "http://localhost:${NETWORK__VLLM__INTERNAL_PORT}/v1/models"
    ]
```

Monitoring / SQL options:

- Render `prometheus.yml` from network primitives before Compose starts.
- Render ClickHouse init SQL from Kafka broker/topic primitives before Compose
  starts.

Preferred rule:

- Configurable means single-source.
- Hardcoded values may remain only when they are not operator configuration.
- No half-configurable values.

## Problem 6: Compose Adapter Mappings Are Not Classified

Current symptoms:

- Some Compose mappings are legitimate adapter boundaries. For example,
  `RABBITMQ_DEFAULT_USER` / `RABBITMQ_DEFAULT_PASS` feed Airflow's
  `AIRFLOW__CELERY__BROKER_URL`, Flower's `CELERY_BROKER_URL`, and any Python
  service broker URL.
- Some values are project-owned but fan out into several project-owned names.
  For example, `VLLM_MODEL` becomes vLLM's model arg,
  `GATEWAY__DEFAULT_MODEL`, and `EVAL__JUDGE__MODEL`.
- Some env names are reused with different meanings. For example, host
  `PROJECT_ROOT` is a host path, while container `PROJECT_ROOT` can mean
  `/opt/airflow/project` or `/home/jovyan`.

### Solution: Separate Adapter Exports From Canonical Inputs

Compose should remain the adapter layer, but every mapping should be classified:

- Native third-party adapter:
  canonical value exported under the name required by an image or SDK.
- Derived project endpoint:
  generated from canonical `NETWORK__...`, native service env, or runtime TOML
  primitives.
- Container-local convenience:
  a value that is meaningful only inside one container, such as
  `CONTAINER__PROJECT_ROOT=/home/jovyan`.
- Suspicious duplicate:
  a project-owned value copied into multiple project-owned names instead of
  resolved through shared settings.

Preferred rule:

- Native third-party names may appear in service `environment:` blocks.
- Derived project values should be built once from canonical primitives.
- Container-local convenience names should not reuse host-level names unless
  the meaning is genuinely identical.
- Suspicious duplicates should be removed or replaced with computed settings.
- Repeated env mappings should live in named `x-*` fragments.
- Service-specific env should remain in the service block so each service still
  documents what it receives.

Examples to classify during implementation:

- `AIRFLOW__API_AUTH__JWT_SECRET` exported also as `AIRFLOW__API__SECRET_KEY`:
  legitimate Airflow adapter mapping.
- `GF_SECURITY_ADMIN_PASSWORD`: canonical Grafana-native admin password, not a
  project alias.
- `VLLM_MODEL` exported as `GATEWAY__DEFAULT_MODEL` and
  defaulted into `EVAL__JUDGE__MODEL`: project-owned duplication to remove.
- Host `PROJECT_ROOT` remapped to container `PROJECT_ROOT`: rename host input
  to `DEPLOY__PROJECT_ROOT` and container-local values to
  `CONTAINER__PROJECT_ROOT`.

Compose organization rule:

- Put repeated cross-service env groups in top-level fragments, for example
  `x-network-env`, `x-observability-env`, `x-s3-client-env`,
  `x-rabbitmq-client-env`, `x-postgres-client-env`, and
  `x-airflow-common-env`.
- Keep one-off service env values under the service that consumes them.
- Do not move every service's full `environment:` block to the top-level
  `x-*` section; that hides the service dependency surface.

## Problem 7: Host And Container Entrypoints Are Blurred

Current symptoms:

- `model_registry.py` can call `get_settings()` before loading the requested
  env file.
- Since settings are cached, `--env-file` may not affect the settings used by
  the command.
- Some code is designed to work both from the host and from containers, which
  forces shared code to care about `.env` discovery, host-local URLs, and
  Docker-internal URLs.
- `shared.local_env` exists for host convenience, but imports of it can leak
  into generic runtime code.

### Solution: Split Host-Side And Container-Side Entrypoints

Rules:

- Generic code under `src/...` must not load `.env`.
- DAGs, services, and container entrypoints must not load `.env`.
- Containers receive all required env through Compose `environment:` mappings
  and `x-*` fragments.
- Containers receive runtime and catalog TOML as mounted files.
- Host-side scripts may read `.env`, but they must be explicitly host-side
  entrypoints.
- Shared business logic should be importable from both host wrappers and
  container entrypoints without performing env loading itself.

Directory/entrypoint pattern:

```text
src/...                 shared library and service code; no `.env` loading
dags/...                container-only Airflow code; no `.env` loading
scripts/host/...        host wrappers; may read `.env`
scripts/deploy/...      deploy automation; may read and update `.env`
scripts/render_configs.py host/deploy-time renderer; reads `.env` and TOML
container entrypoints   consume process env and mounted config files only
```

For `model_registry.py`, split the concerns:

- keep registry/sync logic as shared code;
- move host `.env` loading to a host wrapper;
- let the container adapter-sync entrypoint receive env from Compose and
  runtime TOML from a mounted file.

## Out Of Scope For This Refactor

- Host nginx upstream hardcoding. This is known technical debt, but this
  proposal does not change nginx.
- The decision to keep internal services bound to server-local loopback ports.
  Those ports can remain useful for SSH port forwarding and direct server
  operations.

## Proposed Refactor Order

1. Lock the source split: `.env` for deployment/secrets/native env contracts,
   `config/runtime.toml` for runtime behavior, and `catalog.toml` for
   KB/domain config.
2. Add runtime TOML loading and validation.
3. Add `NetworkSettings` and the nested `NETWORK__...` contract.
4. Add explicit endpoint resolver helpers for host and internal endpoints.
5. Rename existing ambiguous port variables into network primitives.
6. Move project-owned service URLs out of runtime namespaces and into computed
   network-derived properties.
7. Remove all operator-setting defaults from `config.py` and validate required
   fields.
8. Unify MLflow tracking around the network resolver.
9. Split host-side wrappers from container entrypoints and remove `.env`
   loading from shared/runtime code.
10. Classify Compose env mappings as adapter exports, derived values,
   container-local conveniences, or suspicious duplicates.
11. Reorganize Compose env mappings into repeated top-level `x-*` fragments
   while keeping service-specific env in service blocks.
12. Add the required render step for Compose adapter env, Prometheus, and
    ClickHouse generated configs.
13. Update `.env.example`, `config/runtime.toml`, and docs to describe
    primitives, runtime policy, and derived URLs.

## Acceptance Criteria

- No operator-maintained env var stores `http://localhost:...` or
  `http://service:...` for project-owned services.
- Native env names are canonical when a stable upstream owner exists.
- Nested naming remains the project convention only for project-created values;
  project-owned service coordinates have their own `NETWORK__...` root.
- Host ports and internal ports are named differently.
- Host-side tools derive localhost endpoints from host ports.
- Containers derive Docker endpoints from service names and internal ports.
- Code chooses host or internal endpoint helpers explicitly.
- Generic endpoint helpers such as `settings.network.url(...)` do not exist.
- Containers cannot accidentally receive host-local endpoint strings from the
  repo-root `.env`.
- Python settings fail fast when required network coordinates are missing.
- Python settings do not provide hidden operator defaults.
- Compose does not use fallback defaults for canonical values.
- Generated runtime adapter variables use the `COMPOSE__...` prefix and are
  not read by Python application code.
- Python services, DAGs, and shared library code do not read `.env`.
- Containers receive env only through explicit Compose `environment:` mappings
  or `x-*` fragments.
- Containers that read project settings receive read-only runtime/catalog TOML
  mounts plus explicit `CONFIG__...` paths.
- Compose does not use service-level `env_file:` to inject the whole `.env`
  into containers.
- Host-side scripts that read `.env` are explicitly separated from container
  entrypoints.
- Runtime behavior is loaded from `config/runtime.toml`, not from runtime env
  sprawl.
- Python settings read runtime and catalog config paths from explicit
  `CONFIG__...` process env values injected by Compose or host wrappers.
- MLflow has one documented client contract.
- Configurable ports, service names, scrape targets, and Kafka endpoints do not
  have duplicate hardcoded copies.
- Compose env mappings are classified; legitimate adapter mappings remain,
  project-owned duplicate aliases are removed, and container-local names are
  clear.
- Repeated Compose env groups are extracted into named `x-*` fragments;
  service-specific env remains local to each service block.
- The model registry host wrapper can load `.env`; the adapter-sync container
  entrypoint cannot.
- One `.env.example` contains the complete canonical env contract.
- One `config/runtime.toml` contains the complete canonical runtime policy.
