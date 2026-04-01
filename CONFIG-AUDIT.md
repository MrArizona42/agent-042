# Configuration Audit

Status: **historical audit snapshot** — captured before the config cleanup pass.

Current runtime contract:

- `shared/config.py` owns shared cross-service settings.
- Canonical internal endpoint vars in compose/docs are `GATEWAY_VLLM_BASE_URL`, `GATEWAY_EMBEDDINGS_URL`, and `EVAL_GATEWAY_URL`.
- `REGISTRY_VLLM_BASE_URL` and `EMBEDDINGS_URL` are still accepted by Python settings as backward-compatible aliases, but they are no longer the documented compose contract.
- `src/embeddings/config.py` was removed; embeddings runtime config now lives in `shared.config` plus the compose and Docker service definition.

The sections below describe the pre-refactor state that was audited.

## Architecture overview

Configuration lives in four layers:

| Layer | File(s) | Role |
|-------|---------|------|
| **Python defaults** | `src/shared/config.py` | Pydantic Settings classes with `default=` values |
| **`.env` file** | `infra/compose/.env` (from `.env.example`) | Per-deployment overrides, passed to Docker Compose |
| **Compose env blocks** | `infra/compose/docker-compose.yaml` | Maps `.env` vars → container env; sometimes adds `:-fallback` defaults |
| **Per-service configs** | `src/worker/config.py`, `src/embeddings/config.py`, `src/eval_worker/config.py` | Separate Settings classes that re-declare the same fields |

The **intended** single source of truth is `shared/config.py`.
In practice, defaults are scattered across all four layers.

---

## Issue 1 — Duplicate Settings classes

### `src/worker/config.py` → `WorkerSettings`

Re-declares fields already in `shared/config.Settings`:

| WorkerSettings field | Env var (alias) | Default | shared/config equivalent | Shared default |
|---|---|---|---|---|
| `redis_url` | `REDIS_URL` | `redis://localhost:6379/0` | `Settings.redis_url` | `redis://localhost:6379/0` |
| `vllm_base_url` | `VLLM_BASE_URL` | `http://localhost:8000` | `Settings.vllm_base_url` | `http://localhost:8000` |
| `vllm_model` | `VLLM_MODEL` | `/models/Qwen/Qwen3-0.6B` | `Settings.default_model` | `/models/Qwen/Qwen3-0.6B` |

If someone changes the default model in `shared/config.py`, the worker keeps the old one.

**Worker-only fields** (`task_default_timeout`, `task_max_retries`, `task_retry_delay`) are legitimate and don't belong in the shared config.

### `src/embeddings/config.py` → `EmbeddingsSettings`

| EmbeddingsSettings field | Env var | Default | shared/config equivalent | Shared default |
|---|---|---|---|---|
| `model` | `EMBEDDINGS_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | `Settings.embedding_model` | same |
| `device` | `EMBEDDINGS_DEVICE` | `cpu` | `Settings.embedding_device` | same |
| `batch_size` | `EMBEDDINGS_BATCH_SIZE` | `32` | `Settings.embedding_batch_size` | same |

The env var names differ (`EMBEDDINGS_MODEL` vs `GATEWAY_EMBEDDING_MODEL`), so Docker Compose has to map one to the other:
```yaml
# docker-compose.yaml, embeddings service
EMBEDDINGS_MODEL: ${GATEWAY_EMBEDDING_MODEL}   # translation layer
```

**Service-only fields** (`host`, `port`) are legitimate.

### `src/eval_worker/config.py` → `EvalWorkerSettings`

Only has `celery_broker_url` — no overlap with shared config. Clean.

### Plan — Issue 1

**Option A — Delete duplicate classes, import from shared (recommended)**

Both Dockerfiles already `COPY src /app/src` with `PYTHONPATH=/app/src`, and
both lock files include `pydantic-settings`. So `shared.config` is already
importable from inside every container.

1. **`src/worker/config.py`** — delete `redis_url`, `vllm_base_url`, `vllm_model`
   fields. Keep `WorkerSettings` for the three worker-only fields
   (`task_default_timeout`, `task_max_retries`, `task_retry_delay`) plus
   `celery_broker_url`. For the deleted fields, import `get_settings` from
   `shared.config` in `tasks.py` where they're used.

   ```python
   # src/worker/tasks.py  (after)
   from shared.config import get_settings
   from worker.config import get_worker_settings

   shared = get_settings()       # redis_url, vllm_base_url, default_model
   worker = get_worker_settings() # celery_broker_url, task_* fields
   ```

   Docker Compose `celery-worker` env block: drop `VLLM_BASE_URL` and
   `VLLM_MODEL` (shared.config reads `GATEWAY_VLLM_BASE_URL` and
   `GATEWAY_DEFAULT_MODEL`). Keep `REDIS_URL` since Settings already has a
   `validation_alias` for it.

2. **`src/embeddings/config.py`** — keep `EmbeddingsSettings` for `host` and
   `port` only (service-specific). For `model`, `device`, `batch_size`, add
   `validation_alias=AliasChoices("EMBEDDINGS_MODEL", "GATEWAY_EMBEDDING_MODEL")`
   fields to the shared `Settings` class, or add them to `EmbeddingsSettings`
   with the alias pointing to both env var names — either way, one class owns
   the default.

   The cleaner version: `embeddings/main.py` imports from `shared.config` for
   model/device/batch_size, and from `embeddings/config.py` only for host/port.
   Docker Compose drops the `EMBEDDINGS_MODEL: ${GATEWAY_EMBEDDING_MODEL}`
   translation — shared.config reads `GATEWAY_EMBEDDING_MODEL` directly.

**Verdict:** Option A. Affected files:
- `src/worker/config.py` — remove 3 fields
- `src/worker/tasks.py` — add `from shared.config import get_settings`
- `src/worker/celery_app.py` — `celery_broker_url` stays in WorkerSettings (no change)
- `src/embeddings/config.py` — remove 3 fields, keep host/port
- `src/embeddings/main.py` — add `from shared.config import get_settings`
- `infra/compose/docker-compose.yaml` — update celery-worker and embeddings env blocks

---

## Issue 2 — Docker Compose `:-` defaults that shadow Python defaults

These compose lines bake in a default value that also exists in the pydantic class.
If someone updates the Python default but forgets compose, they diverge silently.

| Compose line | Variable | `:-` default | Python class | Python default |
|---|---|---|---|---|
| `REGISTRY_SYNC_ALIASES: ${REGISTRY_SYNC_ALIASES:-champion,challenger}` | sync_aliases | `champion,challenger` | `ModelRegistrySettings` | `["champion", "challenger"]` |
| `EMBEDDINGS_DEVICE: ${EMBEDDINGS_DEVICE:-cpu}` | device | `cpu` | `EmbeddingsSettings` / `Settings` | `cpu` |
| `GATEWAY_SESSION_TTL_SECONDS: ${GATEWAY_SESSION_TTL_SECONDS:-86400}` | session_ttl_seconds | `86400` | `Settings` | `86400` |
| `GATEWAY_GOOGLE_CLIENT_ID: ${GATEWAY_GOOGLE_CLIENT_ID:-}` | google_client_id | `""` | `Settings` | `""` |
| `GATEWAY_GOOGLE_CLIENT_SECRET: ${GATEWAY_GOOGLE_CLIENT_SECRET:-}` | google_client_secret | `""` | `Settings` | `""` |
| `GATEWAY_GOOGLE_REDIRECT_URI: ${GATEWAY_GOOGLE_REDIRECT_URI:-}` | google_redirect_uri | `""` | `Settings` | `""` |
| `GATEWAY_AGENT042_DB_URL: ${GATEWAY_AGENT042_DB_URL:-}` | agent042_db_url | `""` | `Settings` | `None` |
| `GATEWAY_SESSION_SECRET_KEY: ${GATEWAY_SESSION_SECRET_KEY:-}` | session_secret_key | `""` | `Settings` | `""` |
| `GATEWAY_INTERNAL_API_KEY: ${GATEWAY_INTERNAL_API_KEY:-}` | internal_api_key | `""` | `Settings` | `""` |
| Airflow `AIRFLOW_JWT_SECRET:-airflow_jwt_secret` (×3) | — | `airflow_jwt_secret` | N/A | N/A |
| Airflow `AIRFLOW_JWT_ISSUER:-airflow` (×2) | — | `airflow` | N/A | N/A |
| Airflow `AIRFLOW_ADMIN_USER:-airflow` (×2) | — | `airflow` | N/A | N/A |

The `:-` fallbacks for empty-string vars (`GATEWAY_GOOGLE_CLIENT_ID:-`) are harmless since they match the pydantic default. The `:-champion,challenger` and `:-86400` ones are the risky ones.

The Airflow `:-` defaults are not managed by pydantic so they're a separate concern — but the JWT secret fallback `:-airflow_jwt_secret` means Airflow runs with a known secret if the `.env` is missing that var.

### Plan — Issue 2

**Option A — Strip `:-` defaults from compose, let pydantic own defaults (recommended)**

For every variable managed by a pydantic Settings class, change
`${VAR:-default}` → `${VAR}` in compose. If `.env` doesn't define the var,
compose passes nothing, and pydantic uses its own default. One source of truth.

Concrete changes in `docker-compose.yaml`:
```yaml
# Before                                         # After
REGISTRY_SYNC_ALIASES: ${REGISTRY_SYNC_ALIASES:-champion,challenger}  →  REGISTRY_SYNC_ALIASES: ${REGISTRY_SYNC_ALIASES}
EMBEDDINGS_DEVICE: ${EMBEDDINGS_DEVICE:-cpu}      →  EMBEDDINGS_DEVICE: ${EMBEDDINGS_DEVICE}
GATEWAY_SESSION_TTL_SECONDS: ${GATEWAY_SESSION_TTL_SECONDS:-86400}  →  GATEWAY_SESSION_TTL_SECONDS: ${GATEWAY_SESSION_TTL_SECONDS}
```

For the empty-string fallbacks (`GATEWAY_GOOGLE_CLIENT_ID: ${GATEWAY_GOOGLE_CLIENT_ID:-}`),
these are harmless but inconsistent — strip them too for uniformity, or remove
the lines entirely from compose and let the container inherit from `.env`
automatically (compose passes host env vars to containers when the var has no
value mapping in the `environment:` block).

For Airflow vars (`AIRFLOW_JWT_SECRET:-airflow_jwt_secret`): these are NOT
managed by pydantic, but the fallback is a security risk. Strip the `:-`
default and make `.env` the sole source — if the var is missing, compose
should fail to start rather than silently use a known secret.

**Option B — Move defaults from pydantic to `.env.example` only**

The opposite approach: remove `default=` from pydantic fields and make every
var required. `.env.example` becomes the canonical list of defaults.
Simpler mental model but loses local-dev convenience (can't run Python scripts
without an `.env` file).

Not recommended — too disruptive, and pydantic defaults for local dev are
valuable.

**Verdict:** Option A. Edit `docker-compose.yaml` only (~10 lines).

---

## Issue 3 — Same concept behind different env var names

### vLLM URL is configured three times

| Who uses it | Env var | Where set |
|---|---|---|
| Gateway | `GATEWAY_VLLM_BASE_URL` | `.env` → compose → `Settings.vllm_base_url` |
| Adapter sync | `REGISTRY_VLLM_BASE_URL` | Hardcoded `http://vllm:8000` in compose |
| Celery worker | `VLLM_BASE_URL` | Hardcoded `http://vllm:8000` in compose |

All three point to the same vLLM instance. If someone changes the vLLM hostname, they have to update three places in compose.

### Embedding model is configured twice

| Consumer | Env var | Where set |
|---|---|---|
| Gateway | `GATEWAY_EMBEDDING_MODEL` | `.env` |
| Embeddings service | `EMBEDDINGS_MODEL` | Set to `${GATEWAY_EMBEDDING_MODEL}` in compose |

Compose translates between them. This is fragile — if the embeddings service reads `EMBEDDINGS_MODEL` but `.env` only defines `GATEWAY_EMBEDDING_MODEL`, compose must glue them.

### Redis URL

| Consumer | Env var | Source |
|---|---|---|
| Gateway (Settings) | `REDIS_URL` | `redis://redis:6379/0` hardcoded in compose |
| Worker (WorkerSettings) | `REDIS_URL` | `redis://redis:6379/0` hardcoded in compose |

Same value set twice in compose for two services. They could share an anchor.

### Plan — Issue 3

**Option A — YAML anchors for internal service URLs (recommended)**

Add a reusable env fragment at the top of `docker-compose.yaml` for URLs that
multiple services need. This eliminates duplication within compose without
touching Python code.

```yaml
x-internal-urls: &internal-urls
   GATEWAY_VLLM_BASE_URL: http://vllm:8000
  REDIS_URL: redis://redis:6379/0
   GATEWAY_EMBEDDINGS_URL: http://embeddings:8100
   EVAL_GATEWAY_URL: http://gateway:9000
  MLFLOW_TRACKING_URI: http://mlflow:5000
```

Then each service merges the anchor:
```yaml
celery-worker:
  environment:
    <<: *internal-urls
    CELERY_BROKER_URL: amqp://${RABBITMQ_USER}:${RABBITMQ_PASS}@rabbitmq:5672//
    VLLM_MODEL: ${VLLM_MODEL}
```

This requires matching the env var names across services. After Issue 1 is
fixed (worker reads from `shared.config` via `GATEWAY_*` vars), the celery
worker no longer needs `VLLM_BASE_URL` at all — it reads
`GATEWAY_VLLM_BASE_URL`. So the anchor shrinks to URLs that aren't already
covered by shared.config's `GATEWAY_*` prefix.

After Issue 1:
- `GATEWAY_VLLM_BASE_URL` → used by gateway, worker, adapter-sync (via
  `validation_alias` in `ModelRegistrySettings`)
- `REDIS_URL` → used by gateway, worker (already has `validation_alias`)
- `GATEWAY_EMBEDDINGS_URL` → used by gateway, airflow (already one env var)

**Option C — Add `validation_alias` to `ModelRegistrySettings.vllm_base_url`**

So it can read both `REGISTRY_VLLM_BASE_URL` and `GATEWAY_VLLM_BASE_URL`:
```python
vllm_base_url: str = Field(
    default="http://localhost:8000",
    validation_alias=AliasChoices("REGISTRY_VLLM_BASE_URL", "GATEWAY_VLLM_BASE_URL"),
    ...
)
```
Then compose only sets `GATEWAY_VLLM_BASE_URL` once, and both Settings and
ModelRegistrySettings read it. Same pattern already used for `qdrant_host`,
`redis_url`, `celery_broker_url`.

**Verdict:** Option C for Python-level dedup + Option A for compose-level
dedup. Apply Issue 1 first — most compose duplication disappears naturally.

---

## Issue 4 — Port mapping confusion

skip

---

## Issue 5 — Hardcoded internal URLs in docker-compose

These URLs are hardcoded directly in compose instead of being `.env` vars:

| Service | Variable | Hardcoded value |
|---|---|---|
| airflow-common-env | `GATEWAY_EMBEDDINGS_URL` | `http://embeddings:8100` |
| airflow-common-env | `EVAL_GATEWAY_URL` | `http://gateway:9000` |
| airflow-common-env | `MLFLOW_TRACKING_URI` | `http://mlflow:5000` |
| vllm-adapter-sync | `REGISTRY_VLLM_BASE_URL` | `http://vllm:8000` |
| gateway | `GATEWAY_EMBEDDINGS_URL` | `http://embeddings:8100` |
| celery-worker | `REDIS_URL` | `redis://redis:6379/0` |
| celery-worker | `VLLM_BASE_URL` | `http://vllm:8000` |

These are fine for a single-compose stack (container names are stable), but they can't be overridden without editing compose directly.

### Plan — Issue 5

**Option A — YAML anchor (recommended, pairs with Issue 3)**

The `x-internal-urls` anchor from Issue 3 already centralises these. Services
that need them merge the anchor. If a URL needs to change (e.g. running
embeddings on a separate host), override it in the service's `environment:`
block — YAML merge lets later keys win.

No `.env` vars needed for internal container-to-container URLs.

---

## Issue 6 — Hardcoded timeouts in application code

| File | Location | Value | Could use |
|---|---|---|---|
| `src/gateway/services/redis_stream.py` L55 | `subscribe(timeout=300.0)` | `300.0` | `UISettings.chat_timeout` |
| `src/gateway/services/redis_stream.py` L122 | `subscribe_sse(timeout=300.0)` | `300.0` | `UISettings.chat_timeout` |
| `src/RAG/embeddings.py` L46 | `httpx.Client(timeout=120.0)` | `120.0` | No matching config field |

The 300.0 values happen to match `UISettings.chat_timeout = 300.0` but will diverge if that setting changes.

### Plan — Issue 6

**`redis_stream.py` timeouts:**

The `subscribe()` and `subscribe_sse()` timeouts are function parameter
defaults — the caller (gateway) should pass the value from config.

`gateway/main.py` already has access to settings. Where it calls
`redis_stream.subscribe()`, pass `timeout=settings.vllm_timeout` or a
dedicated streaming timeout field. The method signature stays the same
(timeout is already a parameter), only the call site changes.

Alternatively, add a `streaming_timeout` field to `Settings` (default 300.0)
so it's independently configurable.

**`RAG/embeddings.py` timeout:**

Add an `embeddings_timeout` field to `Settings` (default 120.0), then use it
in the `EmbeddingService` constructor:
```python
settings = get_settings()
self._client = httpx.Client(base_url=base_url, timeout=settings.embeddings_timeout)
```

This is a minor change — one new field in `Settings`, one line in
`embeddings.py`.

---

## Issue 7 — DAG alias dropdowns are hardcoded

`dags/eval_dags.py` lines 299–311:
```python
"rag_aliases": Param(
    enum=["none", "champion", "challenger", "champion,challenger"],
    ...
),
"lora_aliases": Param(
    enum=["none", "champion", "challenger", "champion,challenger"],
    ...
),
```

These match `ModelRegistrySettings.sync_aliases` default `["champion", "challenger"]`, but if `REGISTRY_SYNC_ALIASES` is changed (e.g. add a `"canary"` alias), the Airflow dropdown won't offer it.

Mitigated by the `custom_params` JSON override field, but easy to forget.

### Plan — Issue 7

**Option A — Build enum from env var at DAG parse time (recommended)**

The DAG file already sets up `sys.path` to import from `src/`. Read
`REGISTRY_SYNC_ALIASES` at parse time and build the dropdown dynamically:

```python
import os
_sync_raw = os.environ.get("REGISTRY_SYNC_ALIASES", "champion,challenger")
_sync_aliases = [a.strip() for a in _sync_raw.split(",") if a.strip()]
_alias_options = ["none"] + _sync_aliases
# Add combo option for convenience
if len(_sync_aliases) > 1:
    _alias_options.append(",".join(_sync_aliases))
```

Use `_alias_options` in both `rag_aliases` and `lora_aliases` Param enums.
The `REGISTRY_SYNC_ALIASES` env var is already in the airflow-common-env
block (from `.env` or compose default), so it's available at DAG parse time.

No Python config import needed — just `os.environ.get()`.

---

## Issue 8 — Nginx ports are fully hardcoded

`infra/nginx/agent.antonlab.ru.conf` lines 18–25:
```
upstream streamlit_ui     { server 127.0.0.1:8501; }
upstream gateway_api      { server 127.0.0.1:9001; }
upstream mlflow_web       { server 127.0.0.1:5050; }
upstream flower_web       { server 127.0.0.1:5555; }
upstream redisinsight_web { server 127.0.0.1:5540; }
upstream rabbitmq_mgmt    { server 127.0.0.1:15672; }
```

These must match the `*_PORT` values in `.env.example`. If ports change in `.env`, nginx must be edited manually. This is normal for static nginx config but worth noting.

### Plan — Issue 8

**Option A — Nginx template with envsubst (if desired)**

Rename to `agent.antonlab.ru.conf.template`, use `${GATEWAY_PORT}` etc.,
and run `envsubst` before reloading nginx. Common pattern but adds a build
step.

---

## Summary and priorities

| # | Issue | Severity | Fix complexity |
|---|---|---|---|
| 1 | `worker/config.py` and `embeddings/config.py` duplicate shared settings | **High** — silent divergence risk | Medium — refactor to import from shared or delete overlapping fields |
| 2 | Compose `:-` defaults shadow pydantic defaults | **Medium** — two places to update | Low — strip `:-` fallbacks, let pydantic handle defaults |
| 3 | Same vLLM URL behind 3 env var names | **Medium** — maintenance burden | Low — use YAML anchors or a single env var with aliases |
| 4 | 9000/9001 and 5000/5050 port confusion | **Low** — works correctly, just confusing | Low — document or unify |
| 5 | Hardcoded internal URLs in compose | **Low** — stable in single-compose | Optional — extract to `.env` vars |
| 6 | Hardcoded timeouts in code | **Low** — unlikely to diverge | Low — wire to config |
| 7 | DAG alias dropdowns hardcoded | **Low** — mitigated by custom_params | Medium — would need runtime config read |
| 8 | Nginx ports hardcoded | **Low** — standard for static configs | N/A — acceptable |

---

## Execution order

Issues have dependencies. Recommended sequence:

```
1. Issue 1 (duplicate classes)     ← foundation; unlocks Issue 3 simplification
2. Issue 3 (env var unification)   ← add validation_alias + YAML anchors
3. Issue 2 (strip :- defaults)     ← trivial after 1 and 3 settle the env vars
4. Issue 5 (hardcoded URLs)        ← solved by YAML anchors from Issue 3
5. Issue 6 (hardcoded timeouts)    ← independent, small
6. Issue 7 (DAG dropdowns)         ← independent, small
7. Issue 8 (nginx)                 ← optional, add comment only
```

---

## Alternative — Total config workflow rebuild

If the incremental fixes above feel like patching, here's what a clean-slate
approach would look like.

### Principle

**One class per concern, zero duplication, env vars are the only interface
between Python and infrastructure.**

### Design

1. **`shared/config.py` is the sole owner of every default value.** No other
   Python file declares defaults for settings that shared.config already owns.
   Per-service config files exist only for fields unique to that service
   (e.g. `host`/`port` for embeddings, `task_*` for worker). They import
   shared settings for everything else.

2. **Every Settings class uses `validation_alias=AliasChoices(...)` for
   cross-service vars.** Example: `vllm_base_url` accepts
   `GATEWAY_VLLM_BASE_URL`, `REGISTRY_VLLM_BASE_URL`, and `VLLM_BASE_URL`.
   Compose sets one canonical var; all classes read it.

3. **`.env.example` is the deployment reference.** It lists every env var,
   grouped by service, with comments. It's the "what to configure" doc.

4. **`docker-compose.yaml` never uses `:-` defaults.** It only maps
   `${VAR}` → container env. If a var is missing from `.env`, compose shows
   a warning and pydantic provides the default (or fails if required).

5. **Internal container URLs live in a YAML anchor.** Not in `.env` (they're
   infra, not app config). Services merge the anchor.

6. **Timeouts and tuning knobs are Settings fields.** No magic numbers in
   application code. If a value could ever need changing, it's a field.

### What changes

| Layer | Before | After |
|-------|--------|-------|
| `shared/config.py` | 4 Settings classes, some fields overlap with per-service configs | Same 4 classes, but they're the **only** place defaults live. Add `validation_alias` to ~3 fields. Add `embeddings_timeout` field. |
| `worker/config.py` | 7 fields (3 duplicate) | 4 fields (worker-only). Reads shared config for vllm/redis/model. |
| `embeddings/config.py` | 5 fields (3 duplicate) | 2 fields (host, port). Reads shared config for model/device/batch_size. |
| `docker-compose.yaml` | ~10 `:-` defaults, ~7 hardcoded URLs repeated across services | 0 `:-` defaults. One `x-internal-urls` anchor. Each service's env block is minimal. |
| `.env.example` | Current state | Add `REGISTRY_SYNC_ALIASES` (currently missing — only has `:-` default in compose). |
| `dags/eval_dags.py` | Hardcoded alias enums | Dynamic enum from `os.environ.get("REGISTRY_SYNC_ALIASES")`. |
| `redis_stream.py` | `timeout=300.0` hardcoded | Caller passes `settings.vllm_timeout` or new `streaming_timeout` field. |
| `RAG/embeddings.py` | `timeout=120.0` hardcoded | Reads `settings.embeddings_timeout`. |
| `nginx` | Hardcoded ports | Same, with a comment mapping ports to `.env` vars. |

### Estimated scope

- **Python files touched:** 6 (`shared/config.py`, `worker/config.py`,
  `worker/tasks.py`, `embeddings/config.py`, `embeddings/main.py`,
  `RAG/embeddings.py`)
- **Infra files touched:** 2 (`docker-compose.yaml`, `.env.example`)
- **DAG files touched:** 1 (`eval_dags.py`)
- **No API changes, no new dependencies, no migration.**
