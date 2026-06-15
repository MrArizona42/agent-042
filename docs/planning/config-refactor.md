# Configuration Refactor Plan

This document is the locked design for the config refactor. It replaces the
earlier generated-config approach.

The target model is:

- `.env` is the Compose, deployment, and container-startup contract.
- `runtime.toml` is the application runtime policy contract.
- `catalog.toml` is the knowledge-base/domain catalog contract.
- All three operator-facing config files live at the project root.
- Docker Compose reads `.env` directly.
- Containers receive env only through explicit Compose `environment:` mappings
  and `x-*` fragments.
- Python reads process env plus mounted TOML files. Python does not read
  `.env`.
- Host-side scripts may read `.env`, but only from explicit host/deploy
  entrypoints.
- No repo-level render step creates generated boot config under `artifacts/`.

## Source Files

The canonical operator files are:

```text
.env
runtime.toml
catalog.toml
```

`.env.example` is the complete template for `.env`.

`artifacts/` is for produced runtime/build artifacts. It must not contain
generated Compose env files, generated Prometheus config, generated ClickHouse
init SQL, or other startup configuration.

## Root Paths

Use one host project root:

```env
PROJECT_ROOT=/home/anton-m/agent-042/current
```

Do not keep separate host config path envs such as
`DEPLOY__RUNTIME_CONFIG_PATH` or `DEPLOY__CATALOG_CONFIG_PATH`. Compose derives
root config mounts from `PROJECT_ROOT`:

```yaml
- ${PROJECT_ROOT}/runtime.toml:/opt/agent/runtime.toml:ro
- ${PROJECT_ROOT}/catalog.toml:/opt/agent/catalog.toml:ro
```

Containers receive explicit container-local config paths:

```yaml
CONFIG__RUNTIME_PATH: /opt/agent/runtime.toml
CONFIG__CATALOG_PATH: /opt/agent/catalog.toml
```

`PROJECT_ROOT` means a host path in `.env`. If a container needs a project-root
path, use `CONTAINER__PROJECT_ROOT`. Temporary legacy container exports named
`PROJECT_ROOT` are allowed only where existing DAG code still requires them and
must be removed when that code is migrated.

## Env Naming

Use native names first.

If a value has a stable upstream/native env name from a container image, SDK,
provider, tool, or orchestrator, keep that name canonical:

```env
RABBITMQ_DEFAULT_PASS=...
GF_SECURITY_ADMIN_PASSWORD=...
MLFLOW_S3_ENDPOINT_URL=...
AWS_ACCESS_KEY_ID=...
AIRFLOW__CORE__FERNET_KEY=...
```

Use project-owned nested names only for values invented by this project:

- `NETWORK__...`: internal host, internal port, host port, and scheme for
  project-owned services.
- `PUBLIC__...`: public route primitives.
- runtime secrets such as `GATEWAY__API_KEY`, `AUTH__...`, and
  `EVAL__JUDGE__API_KEY`.
- vLLM container launch values, because those are project-owned startup
  controls for the local vLLM service.

Do not define a native env and a project alias for the same source value. For
example, use `MLFLOW_S3_ENDPOINT_URL` directly; do not also introduce
`EXTERNAL__OBJECT_STORAGE__ENDPOINT_URL`.

## Final `.env` Inventory

The final `.env.example` must contain this complete inventory.

```env
# Deployment mechanics
PROJECT_ROOT=/home/anton-m/agent-042/current
IMAGE_TAG=local
COMPOSE_PROJECT_NAME=agent-042

# vLLM container launch
VLLM__MODEL=/models/Qwen/Qwen3-0.6B
VLLM__DTYPE=float16
VLLM__QUANTIZATION=bitsandbytes
VLLM__GPU_UTILIZATION=0.7
VLLM__GPU_COUNT=1
VLLM__MAX_NUM_SEQS=1
VLLM__MAX_NUM_BATCHED_TOKENS=1024
VLLM__KV_CACHE_DTYPE=fp8
VLLM__MAX_LORAS=4
VLLM__MAX_LORA_RANK=16
VLLM__ALLOW_RUNTIME_LORA_UPDATING=true

# Network coordinates
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

# Public routing
PUBLIC__BASE_URL=https://agent.antonlab.ru:8443
PUBLIC__AUTH_CALLBACK_PATH=/auth/callback
PUBLIC__AIRFLOW_PATH=/airflow
PUBLIC__GRAFANA_PATH=/grafana
PUBLIC__FLOWER_PATH=/flower
PUBLIC__REDISINSIGHT_PATH=/redis-insight
PUBLIC__RABBITMQ_MGMT_PATH=/rabbitmq
PUBLIC__JUPYTER_PATH=/jupyter

# Service-owned infrastructure
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

# Native external integrations
MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
AWS_DEFAULT_REGION=ru-central1
AWS_ACCESS_KEY_ID=your-access-key-here
AWS_SECRET_ACCESS_KEY=your-secret-key-here
GITHUB_REPOSITORY=your-org-or-user/agent-042
GITHUB_DATA_SYNC_TOKEN=your-github-token-here
OTEL_TRACES_SAMPLER_ARG=1.0

# Runtime secrets
GATEWAY__API_KEY=
AUTH__GOOGLE_CLIENT_ID=your-google-client-id-here
AUTH__GOOGLE_CLIENT_SECRET=your-google-client-secret-here
AUTH__SESSION_SECRET_KEY=your-session-secret-key-here
AUTH__INTERNAL_API_KEY=your-internal-api-key-here
EVAL__JUDGE__API_KEY=
```

## Runtime TOML Inventory

The final `runtime.toml` contains non-secret application behavior only.
It must not contain container launch settings such as vLLM dtype, GPU count,
ports, service names, or host paths.

```toml
schema_version = 1

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

`VLLM__MODEL` is env because it is required by the vLLM container launch and by
Python services that need to know the local served model identity. It replaces
`GATEWAY__DEFAULT_MODEL`; there is no second default-model setting.

`EVAL__JUDGE__MODEL` is not an env in the final contract. Judge model policy
stays in `runtime.toml` as `[eval.judge].model`.

## Derived Endpoints

Do not maintain full project-owned URLs as operator inputs.

The following are derived from `NETWORK__...`, native credentials, and
`PUBLIC__...` values:

- `PLATFORM__VLLM_BASE_URL`
- `PLATFORM__MLFLOW_TRACKING_URI`
- `PLATFORM__EMBEDDINGS_URL`
- `RAG__RERANKER_URL`
- `PLATFORM__REDIS_URL`
- `PLATFORM__CELERY_BROKER_URL`
- `PLATFORM__KAFKA_BOOTSTRAP_SERVERS`
- `GATEWAY__URL`
- `MLFLOW_BACKEND_URI`
- `AUTH__GOOGLE_REDIRECT_URI`
- `AUTH__AGENT042_DB_URL`
- `EVAL__DB_URL`
- Airflow, Flower, RedisInsight, Jupyter, and Grafana public adapter URLs.

During intermediate phases, Compose may still export some of these legacy-shaped
names to containers because current Python code consumes them. They are adapter
outputs, not `.env` inputs. The final Python settings refactor must replace
them with explicit network resolver fields.

Endpoint helpers must be explicit:

```text
internal_url(service) = scheme://internal_host:internal_port
host_url(service)     = scheme://localhost:host_port
```

Do not add a generic `url(service)` helper.

## Compose Env Organization

Compose is the adapter layer. `.env` supplies interpolation values, but
containers receive only explicitly mapped env.

Final reusable fragments:

```text
x-network-env
x-derived-endpoints
x-config-path-env
x-config-volumes
x-observability-env
x-s3-client-env
x-rabbitmq-client-env
x-postgres-client-env
x-redpanda-client-env
x-runtime-secret-env
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

Rules:

- Repeated env mappings go into a named `x-*` fragment.
- One-off service-specific values stay in the service block.
- Do not use service-level `env_file:` to inject the whole `.env`.
- `x-network-env` passes canonical `NETWORK__...` primitives to Python
  containers that build derived endpoints through `Settings`.
- `x-config-path-env` contains only container-local config paths and any
  env values Python still needs directly, such as `VLLM__MODEL`.
- `x-config-volumes` mounts root `runtime.toml` and `catalog.toml` read-only.
- `x-derived-endpoints` may exist during migration for legacy Python settings,
  but it must disappear when Python reads network primitives directly.
- Native adapter envs such as `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN`,
  `AIRFLOW__CELERY__BROKER_URL`, `CELERY_BROKER_URL`, and
  `MLFLOW_TRACKING_URI` are allowed only inside the service/image boundary that
  requires them.

## No Render Step

There is no `scripts/render_configs.py`.

Startup is:

```bash
docker compose --env-file .env -f infra/compose/docker-compose.yaml up -d
```

Deploy automation may validate `.env`, update image tags, and run Compose. It
must not generate Compose env files or startup configs under `artifacts/`.

Static infra config files that do not support Compose interpolation, such as
Prometheus config or ClickHouse init SQL, remain normal source-controlled infra
files for now. If they must become dynamic later, solve that with a specific
service-native startup design and update this document first. Do not reintroduce
a project-wide pre-Compose renderer.

## No Defaults

No operator-setting defaults remain in Compose or `src/shared/config.py`.

Rules:

- No Compose fallback expressions such as `${NAME:-value}` for canonical
  values.
- No Python defaults for ports, credentials, URLs, model ids, runtime behavior,
  retries, timeouts, budgets, aliases, feature flags, or thresholds.
- Fixed container contracts may still be literal values when they are not
  operator configuration.
- Non-operator constants may exist in normal code, but must not masquerade as
  configurable settings in `Settings`.

## Host And Container Entrypoints

Do not write scripts intended to work equally as host and container entrypoints.

Rules:

- Generic code under `src/...` must not load `.env`.
- DAGs, services, and container entrypoints must not load `.env`.
- Host-side scripts that read `.env` must be explicit host/deploy wrappers.
- Shared business logic should receive already-loaded config.
- `model_registry.py` should keep registry/sync logic only; host `.env`
  loading belongs in a host wrapper, while adapter-sync receives env from
  Compose and TOML from mounts.

## Removed Canonical Inputs

These names are not canonical inputs in the final design:

```text
DEPLOY__PROJECT_ROOT
DEPLOY__ASSETS_ROOT
DEPLOY__ARTIFACTS_ROOT
DEPLOY__DVC_CONFIG_LOCAL_PATH
DEPLOY__RUNTIME_CONFIG_PATH
DEPLOY__CATALOG_CONFIG_PATH
DEPLOY__IMAGE_TAG
ASSETS_ROOT
ARTIFACTS_ROOT
DVC_CONFIG_LOCAL_PATH
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
COMPOSE__*
```

`PROJECT_ROOT`, `IMAGE_TAG`, and `VLLM__*` are intentionally not in this
removed list: they are canonical `.env` values in the revised design.

## Implementation Order

1. Move operator TOML files to the project root: `runtime.toml` and
   `catalog.toml`.
2. Remove `scripts/render_configs.py`, generated startup config under
   `artifacts/generated`, and generated template plumbing.
3. Make Compose read `.env` directly and derive config mounts from
   `PROJECT_ROOT`.
4. Move vLLM container launch settings to `.env.example` and remove `[vllm]`
   from `runtime.toml`.
5. Pass `CONFIG__RUNTIME_PATH`, `CONFIG__CATALOG_PATH`, and required direct env
   such as `VLLM__MODEL` explicitly through Compose.
6. Update Python settings/tests to load runtime policy from root
   `runtime.toml`, catalog config from root `catalog.toml`, and vLLM model from
   process env.
7. Reorganize repeated Compose env into `x-*` fragments while keeping one-off
   env local to services.
8. Add network settings and explicit `internal_url(...)` / `host_url(...)`
   helpers.
9. Replace legacy project-owned URL env consumption in Python with derived
   network settings.
10. Remove hidden defaults from `src/shared/config.py`.
11. Split host-side wrappers from container entrypoints and remove `.env`
   loading from shared/runtime code.

## Acceptance Criteria

- One root `.env.example` contains the complete canonical env contract.
- One root `runtime.toml` contains the complete runtime policy.
- One root `catalog.toml` contains the KB/domain catalog.
- Docker Compose starts with only `--env-file .env`.
- No generated boot configs are written under `artifacts/`.
- Containers receive root TOML files through read-only mounts.
- Containers receive `CONFIG__RUNTIME_PATH` and `CONFIG__CATALOG_PATH`
  explicitly from Compose.
- Python services, DAGs, and shared library code do not read `.env`.
- Host-side `.env` loading exists only in explicit host/deploy wrappers.
- No operator-maintained env var stores full URLs for project-owned services.
- Native env names remain canonical when a stable upstream owner exists.
- Project-owned service endpoints are derived from `NETWORK__...`.
- Host ports and internal ports are named differently.
- Code chooses host or internal endpoint helpers explicitly.
- Compose does not use fallback defaults for canonical values.
- Python settings do not provide hidden operator defaults.
- Compose env mappings are classified as native adapter exports, derived
  endpoints, runtime secrets, container-local conveniences, or one-off service
  env.
- Repeated Compose env groups are extracted into named `x-*` fragments.
- `GATEWAY__DEFAULT_MODEL` is gone; the local served model is `VLLM__MODEL`.
- Runtime judge policy stays in `[eval.judge]`.
