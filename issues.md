# Agent-042 — Project Review: Issues & Implementation Plan

> **Review date:** 2026-02-28
> **Scope:** Infrastructure, pipelines, workflows, security, dependencies, Docker, Airflow DAGs, source code, configuration

---

## Table of Contents

1. [🔴 Critical Issues](#1--critical-issues)
2. [🟠 High-Priority Issues](#2--high-priority-issues)
3. [🟡 Medium-Priority Issues](#3--medium-priority-issues)
4. [🔵 Low-Priority / Improvements](#4--low-priority--improvements)
5. [📋 Implementation Plan](#5--implementation-plan)

---

## 1. 🔴 Critical Issues

### 1.1 Hardcoded Credentials in Source Code

**Files affected:**
- `src/gateway/services/celery_client.py:117-118` — default `amqp://agent:agent@localhost:5672//`
- `src/worker/config.py:15-16` — default `amqp://agent:agent@localhost:5672//`

**Problem:** Default broker URLs contain plaintext credentials (`agent:agent`). Even though they are "fallback defaults," they are committed to Git and reveal the credential pattern used in production. Anyone reading the repo knows the RabbitMQ username and password.

**Fix:** Remove credentials from defaults entirely. Use a placeholder like `amqp://localhost:5672//` (no auth) or require the env var to be set (raise an error if missing). Credentials should only live in `.env` files (which are git-ignored).

---

### 1.2 Secrets in `.env.example`

**Files affected:**
- `infra/compose/.env.example:29-30` — `AWS_ACCESS_KEY_ID=YCAJ...`, `AWS_SECRET_ACCESS_KEY=YCN7o...`
- `infra/compose/.env.example:123` — `AIRFLOW_FERNET_KEY=ZmDfcTF7_60GrrY167zsiPd67pEvs0aGOv2oasOM1Pg=`
- `infra/compose/.env.example:127-128` — `AIRFLOW_ADMIN_USER=admin`, `AIRFLOW_ADMIN_PASSWORD=admin`
- `infra/compose/.env.example:38` — `POSTGRES_PASSWORD=mlflow`
- `infra/compose/.env.example:92` — `RABBITMQ_PASS=agent_secret`
- `infra/compose/.env.example:134` — `JUPYTER_TOKEN=agent042`
- `experiments/.env.example:3-4` — `MLFLOW_TRACKING_USERNAME=user`, `MLFLOW_TRACKING_PASSWORD=password`

**Problem:** While `.env.example` is intended as a template, it contains what appear to be truncated real AWS credentials (`YCAJ...`, `YCN7o...`) and a real Fernet key. Even "example" passwords (`admin/admin`, `mlflow`) are dangerous if someone copies the file to `.env` without changing them — which is the common workflow.

**Fix:**
- Replace all credentials with clearly fake placeholders: `AWS_ACCESS_KEY_ID=your-access-key-here`
- Add a `# CHANGE ME!` comment next to every credential
- Replace the Fernet key with a generation command comment only
- Consider adding a startup check that refuses to start with known-insecure defaults

---

### 1.3 No CI/CD Pipeline

**Problem:** There is no `.github/workflows/` directory at all. No automated testing, linting, Docker image building, or deployment happens on push/PR. This means:
- Regressions can be merged uncaught
- Pre-commit hooks are the only quality gate, and they're local-only (can be skipped)
- Docker images are never tested automatically
- No dependency vulnerability scanning

**Fix:** Create GitHub Actions workflows for:
1. **Lint & type-check** — `ruff check`, `ruff format --check` on every PR
2. **Unit tests** — run `pytest` (once tests exist) on every PR
3. **Docker build validation** — `docker compose config --quiet` + build key images
4. **DAG validation** — `python -c "from airflow.models import DagBag; ..."` in an Airflow container
5. **Dependency audit** — `pip audit` or GitHub Dependabot for known CVEs
6. **Security scanning** — CodeQL or similar for SAST

---

### 1.4 CORS Wildcard Default Allows All Origins

**File affected:** `src/shared/config.py:115-117`

```python
cors_allow_origins: list[str] = Field(
    default_factory=lambda: ["*"],
)
```

**Problem:** The default CORS policy allows every origin. If the gateway is exposed (it is, through nginx), any website can make cross-origin requests to the API. Combined with no authentication on the API endpoints (see §1.5), this means any website a user visits could silently call the LLM API.

**Fix:** Change the default to an empty list `[]` or a restrictive list like `["https://agent.antonlab.ru"]`. The `*` value should only be used in development with an explicit override.

---

### 1.5 No Authentication on Gateway API Endpoints

**Files affected:**
- `src/gateway/api/v1/openai_compat.py` — `/v1/chat/completions`, `/v1/models`
- `src/gateway/api/v1/discovery.py` — `/health`, `/config`

**Problem:** The FastAPI gateway has zero authentication. While nginx adds `auth_basic` for `/api/`, direct access on port 9001 (even from `127.0.0.1`) bypasses this. Any container on the Docker network, or any process on the host, can call the API freely. The `/config` endpoint also exposes internal configuration details.

**Fix:**
- Add optional Bearer token auth via `GATEWAY_API_KEY` (the setting exists in config but is never checked)
- Restrict `/config` endpoint to authenticated requests or remove it in production
- Add middleware that validates `Authorization: Bearer <token>` when `GATEWAY_API_KEY` is set

---

## 2. 🟠 High-Priority Issues

### 2.1 No Test Suite

**Problem:** There are zero test files (`*_test.py`, `test_*.py`, `conftest.py`) in the entire repository. No `pytest` configuration in `pyproject.toml`. No test infrastructure at all.

**Impact:** Every change is a gamble — there's no way to verify that existing functionality still works after modifications.

**Fix:**
- Add `pytest` and `pytest-asyncio` to dev dependencies
- Create test directories: `tests/gateway/`, `tests/rag/`, `tests/worker/`, `tests/shared/`
- Start with unit tests for critical paths: settings parsing, prompt building, task routing, chunking, schema validation
- Add integration tests for the RAG pipeline (using a local Qdrant instance)

---

### 2.2 Dockerfiles Missing `.dockerignore` — Large Build Contexts

**Files affected:** All Dockerfiles use `context: ${PROJECT_ROOT}` (the entire repo as build context).

**Problem:** Every `docker build` sends the entire repository (including `assets/models/`, `.git/`, `node_modules/`, `experiments/outputs/`, etc.) to the Docker daemon. For a project with large ML models, this could mean sending gigabytes of data on each build, even though only `src/` or `infra/docker/` files are needed.

**Fix:** Create a `.dockerignore` at the project root:
```
.git
assets/models
assets/datasets
assets/newly_trained
experiments/outputs
experiments/logs
*.pyc
__pycache__
.venv
.env
```

---

### 2.3 Dockerfile Dependency Management — No Pinning, No Lock Files

**Files affected:**
- `infra/docker/gateway/requirements-gateway.txt` — all `>=` with no upper bounds
- `infra/docker/ui/requirements-ui.txt` — same
- `infra/docker/celery/requirements-celery.txt` — same
- `infra/docker/jupyter/requirements-jupyter.txt` — same
- `infra/docker/mlflow/Dockerfile` — inline pip install with ranges

**Problem:** Using `>=` without upper bounds means builds are non-reproducible. A new major version of `fastapi`, `httpx`, `celery`, etc. could ship at any time and break the build. The `dags/requirements.txt` correctly pins upper bounds, but the service Dockerfiles don't.

**Fix:** Either:
1. Pin exact versions (`==`) in all requirements files and update them explicitly, OR
2. Generate lock files with `pip-compile` or `uv pip compile` and commit them, OR
3. At minimum, add upper-bound constraints (e.g., `fastapi>=0.128.0,<1.0`)

---

### 2.4 `vllm/vllm-openai:latest` Tag — Non-Reproducible Builds

**File affected:** `infra/compose/docker-compose.yaml:129`

```yaml
vllm:
  image: vllm/vllm-openai:latest
```

**Problem:** The `latest` tag is mutable — it changes with every vLLM release. Today's `latest` might work; tomorrow's might not. This leads to "works on my machine" issues and makes rollbacks impossible.

**Fix:** Pin to a specific version tag (e.g., `vllm/vllm-openai:v0.6.0`). Document the version in a comment and update it intentionally.

Same issue applies to:
- `qdrant/qdrant:latest` (line 184)
- `redis/redisinsight:latest` (line 384)

---

### 2.5 Health Check Too Simplistic — No Dependency Validation

**File affected:** `src/gateway/api/v1/discovery.py`

```python
@router.get("/health")
async def health():
    return {"status": "ok"}
```

**Problem:** The health endpoint always returns `ok`, even if vLLM is down, Qdrant is unreachable, Redis is disconnected, or RabbitMQ is offline. Docker Compose and nginx rely on this health check to determine if the gateway is ready, so a false-positive "healthy" status means traffic gets routed to a broken service.

**Fix:** Implement a composite health check:
```python
async def health():
    checks = {}
    checks["vllm"] = await ping_vllm()
    checks["qdrant"] = await ping_qdrant()
    if settings.async_enabled:
        checks["redis"] = await ping_redis()
        checks["celery"] = await ping_celery_broker()
    all_ok = all(checks.values())
    status_code = 200 if all_ok else 503
    return JSONResponse({"status": "ok" if all_ok else "degraded", "checks": checks}, status_code=status_code)
```

---

### 2.6 Shared PostgreSQL Without Logical Separation

**File affected:** `infra/compose/docker-compose.yaml:37-72`

**Problem:** A single PostgreSQL 15 instance serves both MLflow and Airflow databases. While the `airflow-init` service creates a separate `airflow` database, they share:
- The same credentials (`mlflow:mlflow`)
- The same server (resource contention)
- The same failure domain (if PG crashes, both MLflow and Airflow go down)

For production, this is a single point of failure. Airflow heavy DAG parsing and scheduler writes can impact MLflow query performance.

**Fix (short-term):** Acceptable for development. Document this as a known limitation.
**Fix (long-term):** Either use separate PG instances, or at minimum create dedicated users with restricted permissions per database.

---

### 2.7 Airflow Workers Run Index Builds With `--force-recreate`

**Files affected:**
- `dags/arxiv_rag_update.py:137` — `--force-recreate`
- `dags/pytorch_docs_rag_update.py:169` — `--force-recreate`

**Problem:** Every scheduled DAG run deletes and recreates the entire Qdrant collection from scratch. During the rebuild window:
- The collection doesn't exist → live queries to the gateway return no RAG results
- If the build fails halfway, the collection is permanently empty until the next successful run
- All embeddings are recomputed from scratch every time, wasting CPU/GPU resources

**Fix:**
1. Build into a temporary collection (`chat_documents_staging`)
2. Once complete, atomically swap the alias (Qdrant supports collection aliases)
3. Delete the old collection only after the swap succeeds
4. This ensures zero downtime and safe rollback

---

### 2.8 No Rate Limiting on API Endpoints

**Problem:** The gateway has no request rate limiting. A single client can send unlimited `/v1/chat/completions` requests, saturating vLLM, exhausting GPU resources, and causing denial of service for all other users.

**Fix:** Add rate limiting middleware (e.g., `slowapi` for FastAPI, or implement at the nginx level with `limit_req_zone`).

---

## 3. 🟡 Medium-Priority Issues

### 3.1 Unsafe HTML Rendering in UI (Potential XSS)

**File affected:** `src/ui/app.py` — `unsafe_allow_html=True`

**Problem:** The Streamlit UI uses `st.markdown(..., unsafe_allow_html=True)` to render thinking blocks with custom HTML/CSS. If the LLM output (or RAG-injected context) contains malicious HTML, it will be rendered in the user's browser.

**Fix:** Sanitize the HTML output before rendering, or use Streamlit's native expander component instead of raw HTML for thinking blocks.

---

### 3.2 Redis and Celery Connection Lifecycle Issues

**Files affected:**
- `src/gateway/services/redis_stream.py` — global `_redis_stream_service` singleton never closed
- `src/gateway/services/celery_client.py` — global `_celery_client` singleton never closed

**Problem:** Global singleton connections are created lazily but never cleaned up on shutdown. This can cause:
- Connection leaks if the gateway process is recycled
- Stale connections after Redis/RabbitMQ restarts
- No reconnection logic if the broker goes down temporarily

**Fix:**
- Use FastAPI's lifespan context manager to create/close connections
- Implement connection health checks and reconnection logic
- Use connection pools for Redis

---

### 3.3 Streaming Timeout Can Hang Indefinitely

**File affected:** `src/gateway/services/vllm_client.py:59`

```python
timeout=None  # streaming can take long
```

**Problem:** When streaming from vLLM, the timeout is set to `None`, meaning the connection will wait forever. If vLLM hangs (e.g., GPU OOM, process freeze), the gateway request thread also hangs forever.

**Fix:** Use a generous but finite streaming timeout (e.g., 300s) with per-chunk idle timeout.

---

### 3.4 `pyproject.toml` Is a Monolithic Kitchen Sink

**File affected:** `pyproject.toml`

**Problem:** The root `pyproject.toml` lists dependencies for ALL services combined: ML training (`peft`, `datasets`, `hydra-core`), gateway (`fastapi`, `uvicorn`), UI (`streamlit`), RAG (`qdrant-client`, `sentence-transformers`), plus dev tools (`ruff`, `pre-commit`). Several heavy deps are commented out (`torch`, `transformers`, `vllm`, `pytorch-lightning`).

Issues:
- Installing all dependencies is slow and has conflicting constraints
- `uv sync` pulls in everything, even if you only need the gateway
- The commented-out deps suggest the lock file may not be accurate

**Fix:**
- Use UV workspaces or optional dependency groups (`[project.optional-dependencies]`) to separate concerns:
  ```toml
  [project.optional-dependencies]
  gateway = ["fastapi", "uvicorn", ...]
  training = ["torch", "peft", "transformers", ...]
  rag = ["qdrant-client", "sentence-transformers", ...]
  dev = ["ruff", "pre-commit", "pytest"]
  ```
- This allows `uv pip install -e ".[gateway]"` for targeted installs

---

### 3.5 Docker Network Isolation Is Incomplete

**File affected:** `infra/compose/docker-compose.yaml`

**Problem:**
- The `gateway` service is on both `backend_net` and `frontend_net`, which is correct.
- However, `airflow-webserver` and `airflow-scheduler` are on both `mlflow_db_net` and `backend_net`, giving them access to vLLM, Qdrant, Redis, RabbitMQ, etc. — services they don't need direct access to except Qdrant (for index builds).
- The `jupyter` service is on `backend_net`, giving it access to all backend services with no restriction.

**Fix:** Create a dedicated `airflow_net` or tighten access so Airflow only reaches PG and Qdrant. Jupyter should be on a restricted network unless it explicitly needs access to all services.

---

### 3.6 `pre-commit-hooks` Pinned to Old Version

**File affected:** `.pre-commit-config.yaml:3`

```yaml
rev: v4.1.0  # pre-commit-hooks
```

**Problem:** `pre-commit-hooks` v4.1.0 is from early 2022. The current version is v5.x. Old hooks may miss newer file patterns and lack bug fixes.

**Fix:** Run `pre-commit autoupdate` to update all hooks to their latest versions.

---

### 3.7 DVC Configuration Hardcodes Developer Machine Path

**File affected:** `experiments/conf/paths/paths_config.yaml:1`

```yaml
project_root: "/home/anton-m/Git/agent-042"
```

**Problem:** This hardcodes a specific developer's home directory. Any other developer or CI environment will fail.

**Fix:** Use a relative path or environment variable resolution:
```yaml
project_root: ${oc.env:PROJECT_ROOT,/opt/agent-042}
```

---

### 3.8 Qdrant Vector ID Generation — Collision Risk

**File affected:** `src/rag/vector_store.py` — sequential ID generation using `collection.count() + 1`

**Problem:** If documents are ever deleted from a collection, the count decreases but existing IDs still occupy the old numbers. New inserts will generate IDs that collide with existing ones, causing silent data overwrites.

**Fix:** Use UUIDs for point IDs, or use Qdrant's built-in UUID generation.

---

### 3.9 RAG Failures Are Silent — User Gets No Indication

**Files affected:**
- `src/gateway/services/rag_service.py` — catches exceptions and logs them, returns `None`
- `src/gateway/services/processing.py` — continues without RAG if it fails

**Problem:** If Qdrant is down or the embedding model fails, the gateway silently falls back to plain LLM chat with no RAG context. The user doesn't know their "knowledge base" selection had no effect. Answers will be lower quality with no explanation.

**Fix:** Include RAG status in the response metadata:
```json
{
  "choices": [...],
  "rag_status": "unavailable",
  "rag_message": "Knowledge base 'arxiv' is currently unavailable"
}
```

---

### 3.10 Airflow `AIRFLOW__WEBSERVER__EXPOSE_CONFIG: "true"`

**File affected:** `infra/compose/docker-compose.yaml:491`

**Problem:** This setting exposes the full Airflow configuration (including DB connection strings and Fernet key) through the Airflow web UI. Even though Airflow has its own auth, if credentials are weak (see §1.2 — `admin/admin`), this is a significant information disclosure risk.

**Fix:** Set to `"false"` in production. Only enable it for debugging in development.

---

### 3.11 Windows Path in `.env.example`

**File affected:** `infra/compose/.env.example:1`

```
PROJECT_ROOT=C:/Users/user/MyGitRepos/agent-042
```

**Problem:** The `PROJECT_ROOT` example uses a Windows path, but the Docker Compose stack runs on Linux (it uses Linux containers). This path won't work on Linux/Mac hosts and will confuse users trying to set up the project.

**Fix:** Use a Linux-style path like `/home/user/agent-042` or document both:
```
# Linux/Mac:
PROJECT_ROOT=/home/user/agent-042
# Windows (WSL):
PROJECT_ROOT=/mnt/c/Users/user/MyGitRepos/agent-042
```

---

## 4. 🔵 Low-Priority / Improvements

### 4.1 No Structured Logging

**Problem:** All services use Python's basic `logging` module with default formatting. In a multi-container environment, correlating logs across services (gateway → worker → Redis) is very difficult without structured JSON logging and request/correlation IDs.

**Fix:** Add structured JSON logging (e.g., `python-json-logger`) with fields like `service`, `request_id`, `conversation_id`, `timestamp`. This is essential for debugging production issues.

---

### 4.2 No Monitoring or Metrics

**Problem:** No Prometheus metrics, no Grafana dashboards, no alerting. There's no way to know:
- How many requests per second the gateway handles
- What the average response latency is
- If GPU memory is exhausted
- If Qdrant is running out of disk space

**Fix:** Add `/metrics` endpoint using `prometheus-fastapi-instrumentator` and optionally add a Prometheus + Grafana stack to docker-compose.

---

### 4.3 Celery Worker Pool Configuration

**File affected:** `infra/docker/celery/Dockerfile:23`

```
CMD [..., "-c", "1", "-P", "solo"]
```

**Problem:** Single worker with solo pool is correct for GPU-bound tasks, but there's no dead letter queue or task failure persistence. Failed tasks disappear without trace.

**Fix:** Configure a `task_reject_on_worker_lost=True` and a dead letter exchange in RabbitMQ for post-mortem analysis.

---

### 4.4 No `.dockerignore`

**Problem:** Without a `.dockerignore`, every `docker build` command sends the full project directory (including `.git`, large model files, datasets, etc.) as build context to the Docker daemon.

**Fix:** Create `.dockerignore` at project root with sensible excludes.

---

### 4.5 Duplicate Dependency Specifications

**Problem:** Dependencies are specified in multiple places with potentially divergent versions:
- `pyproject.toml` — project-level
- `dags/requirements.txt` — Airflow DAGs
- `infra/docker/gateway/requirements-gateway.txt` — gateway image
- `infra/docker/ui/requirements-ui.txt` — UI image
- `infra/docker/celery/requirements-celery.txt` — worker image
- `infra/docker/jupyter/requirements-jupyter.txt` — Jupyter image
- `infra/docker/mlflow/Dockerfile` — inline pip install

Example: `pydantic-settings` is specified as `>=2.10.1` in gateway requirements but `>=2.0.0` in UI requirements. They may resolve to different versions.

**Fix:** Consider a single source of truth for shared dependencies (e.g., have Docker requirements reference `pyproject.toml` extras), or at minimum document the relationship and keep versions consistent.

---

### 4.6 `sentence-transformers` Pulled into Gateway Container — Heavy Dependency

**File affected:** `infra/docker/gateway/requirements-gateway.txt:16`

**Problem:** `sentence-transformers` pulls in `torch`, `transformers`, `huggingface-hub`, and their transitive dependencies. This makes the gateway Docker image very large (potentially 2-4 GB+), slow to build, and has a large attack surface. All of this just for embedding generation that could run as a separate microservice.

**Fix (short-term):** Accept this as a known trade-off. Document the expected image size.
**Fix (long-term):** Extract the embedding service into a separate container, or use a lighter embedding library (e.g., `onnxruntime` with exported ONNX model).

---

### 4.7 No Backup Strategy for Qdrant Data

**Problem:** Qdrant data lives in a Docker named volume (`qdrant_data`). There's no backup, snapshot, or disaster recovery process. If the volume is lost (e.g., `docker compose down -v` by accident), all vector data is lost and must be rebuilt from scratch.

**Fix:** Implement periodic Qdrant snapshots (Qdrant has a built-in snapshot API) and store them in S3.

---

### 4.8 DAGs Have No Automated Tests

**Problem:** The Airflow DAGs (`arxiv_rag_update.py`, `pytorch_docs_rag_update.py`) have no unit tests. The Python callables (`_download_arxiv_papers`, `_collect_pytorch_docs`) are not tested independently. There's no DAG validation test to ensure they parse correctly.

**Fix:** Add DAG validation tests:
```python
def test_dag_loads():
    from airflow.models import DagBag
    bag = DagBag(dag_folder="dags", include_examples=False)
    assert len(bag.import_errors) == 0
```

---

### 4.9 `pip install` Without `--no-cache-dir` in Some Dockerfiles

**Files affected:**
- `infra/docker/gateway/Dockerfile:12` — `RUN pip install -r requirements-gateway.txt` (no `--no-cache-dir`)
- `infra/docker/ui/Dockerfile:9` — same

**Problem:** Without `--no-cache-dir`, pip caches downloaded packages inside the image layer, increasing image size unnecessarily. The MLflow, Airflow, and Celery Dockerfiles correctly use `--no-cache-dir`, but gateway and UI don't.

**Fix:** Add `--no-cache-dir` to all `pip install` commands.

---

### 4.10 MLflow Healthcheck Uses `curl` But Image Is `python:3.12-slim`

**File affected:** `infra/docker/mlflow/Dockerfile` + `docker-compose.yaml:114`

**Problem:** The MLflow healthcheck uses `curl -f http://localhost:5000/health`, and `curl` is installed in the Dockerfile. This is fine, but if the Dockerfile is simplified to remove `curl`, the healthcheck breaks silently.

**Fix:** Consider using `python -c "import urllib.request; ..."` as healthcheck to avoid the `curl` dependency. This is a minor point — current setup works correctly.

---

### 4.11 Worker Config Uses Deprecated Pydantic `class Config`

**File affected:** `src/worker/config.py:57-58`

```python
class Config:
    extra = "ignore"
```

**Problem:** Pydantic V2 deprecated the inner `class Config` pattern in favor of `model_config = SettingsConfigDict(...)`. The shared config (`src/shared/config.py`) already uses the new pattern, but the worker config doesn't. This will emit deprecation warnings and may break in Pydantic V3.

**Fix:** Replace with:
```python
model_config = SettingsConfigDict(extra="ignore")
```

---

### 4.12 Nginx Configuration Not Integrated with Docker Compose

**File affected:** `infra/nginx/agent.antonlab.ru.conf`

**Problem:** The nginx configuration exists in the repo but there's no nginx service in `docker-compose.yaml`. This suggests nginx runs on the host (outside Docker), creating an operational gap — the stack isn't fully self-contained.

**Fix:** Either:
1. Add an nginx service to docker-compose.yaml with the config mounted, OR
2. Document clearly that nginx must be installed and configured separately on the host

---

### 4.13 No Resource Limits on Docker Services

**Problem:** Most services in `docker-compose.yaml` have no memory or CPU limits (except vLLM which has GPU reservations). A single misbehaving service (e.g., Airflow scheduler consuming too much memory) could starve other services.

**Fix:** Add `deploy.resources.limits` for critical services:
```yaml
deploy:
  resources:
    limits:
      memory: 512M
      cpus: "1.0"
```

---

## 5. 📋 Implementation Plan

### Phase 1: Security Fixes (Immediate)

| # | Task | Files | Effort |
|---|------|-------|--------|
| 1 | Remove hardcoded credentials from source code defaults | `celery_client.py`, `worker/config.py` | 30 min |
| 2 | Sanitize `.env.example` — replace real/near-real secrets with placeholders | `infra/compose/.env.example`, `experiments/.env.example` | 30 min |
| 3 | Fix CORS default to restrictive list | `src/shared/config.py` | 15 min |
| 4 | Implement API key auth middleware for gateway | `src/gateway/main.py`, new `auth.py` | 2 hrs |
| 5 | Disable `AIRFLOW__WEBSERVER__EXPOSE_CONFIG` | `docker-compose.yaml` | 5 min |
| 6 | Sanitize UI HTML rendering | `src/ui/app.py` | 30 min |

### Phase 2: Reliability & Build Quality (1-2 weeks)

| # | Task | Files | Effort |
|---|------|-------|--------|
| 7 | Create `.dockerignore` | New file at project root | 15 min |
| 8 | Pin Docker image versions (`vllm`, `qdrant`, `redisinsight`) | `docker-compose.yaml` | 15 min |
| 9 | Add upper bounds to all requirements files | All `requirements-*.txt` | 1 hr |
| 10 | Fix gateway/UI Dockerfiles to use `--no-cache-dir` | `gateway/Dockerfile`, `ui/Dockerfile` | 5 min |
| 11 | Implement composite health check | `discovery.py`, new health module | 2 hrs |
| 12 | Fix Qdrant ID generation to use UUIDs | `src/rag/vector_store.py` | 1 hr |
| 13 | Fix atomic index rebuild (staging collection + alias swap) | DAGs + `build_vector_index.py` | 4 hrs |
| 14 | Fix hardcoded path in Hydra config | `experiments/conf/paths/paths_config.yaml` | 15 min |
| 15 | Fix Windows path in `.env.example` | `infra/compose/.env.example` | 5 min |
| 16 | Fix deprecated Pydantic Config in worker | `src/worker/config.py` | 15 min |

### Phase 3: CI/CD & Testing (2-3 weeks)

| # | Task | Files | Effort |
|---|------|-------|--------|
| 17 | Set up pytest infrastructure | `pyproject.toml`, `conftest.py`, test dirs | 2 hrs |
| 18 | Write unit tests for critical paths | `tests/` directory | 1 week |
| 19 | Create GitHub Actions lint/test workflow | `.github/workflows/ci.yaml` | 3 hrs |
| 20 | Create GitHub Actions Docker build workflow | `.github/workflows/docker.yaml` | 2 hrs |
| 21 | Add DAG validation tests | `tests/dags/` | 2 hrs |
| 22 | Add Dependabot configuration | `.github/dependabot.yml` | 30 min |
| 23 | Update `pre-commit-hooks` to latest version | `.pre-commit-config.yaml` | 15 min |

### Phase 4: Architecture Improvements (Ongoing)

| # | Task | Files | Effort |
|---|------|-------|--------|
| 24 | Split `pyproject.toml` into dependency groups | `pyproject.toml` | 2 hrs |
| 25 | Add structured JSON logging | All services | 3 hrs |
| 26 | Add RAG status to API responses | `processing.py`, schemas | 2 hrs |
| 27 | Implement connection lifecycle management | `redis_stream.py`, `celery_client.py` | 3 hrs |
| 28 | Add rate limiting | `gateway/main.py` | 2 hrs |
| 29 | Add resource limits to Docker services | `docker-compose.yaml` | 1 hr |
| 30 | Add nginx to Docker Compose (or document external setup) | `docker-compose.yaml`, docs | 2 hrs |
| 31 | Add Prometheus metrics endpoint | `gateway/main.py` | 2 hrs |
| 32 | Implement Qdrant backup strategy | New DAG or script | 3 hrs |
| 33 | Add streaming timeout to vLLM client | `vllm_client.py` | 30 min |
| 34 | Tighten Docker network isolation for Airflow/Jupyter | `docker-compose.yaml` | 1 hr |

---

> **Total estimated effort:** ~4-6 weeks for full implementation
> **Recommended priority:** Phase 1 (security) → Phase 2 (reliability) → Phase 3 (CI/CD) → Phase 4 (architecture)
