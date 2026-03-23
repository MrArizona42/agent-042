# Airflow: Structural Review & Roadmap

Current baseline: `apache/airflow:2.10.4`, `LocalExecutor`, docker-compose deployment.
Target: `apache/airflow:3.1.x` (latest: **3.1.8**, released 2026-03-10).

---

## 1. Version Update

Upgrade `apache/airflow:2.10.4` → latest **2.10.x** patch as a low-risk immediate bump.

**Airflow 3.0** is now stable and is the target for the K8s phase. Key breaking changes
relevant to this project:

| Area | Airflow 2.x (current) | Airflow 3.0 |
|---|---|---|
| SQLAlchemy | `>=1.4,<2.0` | `>=2.0` (fully supported since 3.1.0) |
| Flask / Werkzeug | `Flask<2.3`, `Werkzeug<3.0` | `Flask>=3.0`, `Werkzeug>=3.0` |
| Python | 3.8+ | 3.10+ since 3.1.0 (3.9 dropped; project is on 3.12 ✓) |

All DAGs in `dags/` already use `schedule=` (not the deprecated `schedule_interval=`), so
the DAG code itself requires no changes for 3.0.

**Plan:** migrate to 3.1.x now alongside the dependency cleanup — it's a 30-minute change once
the dep pins are dropped (see §3). There is no K8s prerequisite. The K8s-specific features
(GitSync, KubernetesExecutor, remote logging) are additive on top of 3.1.x and can follow later.

#### Concrete 3.1.x migration delta (docker-compose)

1. Bump image tag: `apache/airflow:2.10.4` → `apache/airflow:3.1.8` in the Dockerfile.
2. Drop the three conflicting dep pins from `pyproject.toml` (already required — see §3).
3. Fix the scheduler healthcheck command — `airflow jobs check` was removed:
   ```yaml
   # Before (2.x)
   test: ["CMD-SHELL", "airflow jobs check --job-type SchedulerJob --hostname \"$${HOSTNAME}\""]
   # After (3.x) — scheduler exposes an internal health API on port 8974
   test: ["CMD", "curl", "-f", "http://localhost:8974/health"]
   ```
4. Remove `AIRFLOW__WEBSERVER__EXPOSE_CONFIG: "true"` — setting was dropped in 3.0.
5. `airflow db migrate`, `airflow users create`, `PythonOperator`, `BashOperator`,
   `Param(enum=)`, and `schedule=` syntax are all unchanged in 3.x ✓.
6. **DAG import namespace (3.1+):** `from airflow.sdk import DAG, task, asset` is now the
   stable, forward-compat import path. The old `from airflow import DAG` / `from airflow.decorators import task`
   still works but is deprecated. Update DAG files at your convenience — no runtime breakage either way.

---

## 2. Critical: Image Bloat from `bert-score`

The current `requirements.lock` for the Airflow image includes the full PyTorch + CUDA
stack (~2–3 GB) pulled transitively by a single dependency:

```
bert-score → torch → nvidia-cublas-cu12, nvidia-cudnn-cu12, triton, ...
```

`bert-score` is only used in `eval_dags.py` inside `_calculate_metrics_task`.
The Airflow scheduler and webserver have no business running transformer inference.

### Is BERTScore GPU-bound?

No, not for this use case. BERTScore uses a transformer encoder (typically `roberta-large`)
and falls back gracefully to CPU. For periodic batch evaluation over hundreds of samples
(HotpotQA, ArXiv, etc.) — nothing user-facing — CPU is completely acceptable (5–20 min
vs 1–2 min on GPU). The CUDA wheel bloat comes from `uv pip compile` resolving `torch`
to the default CUDA wheels on a Linux target; CPU-only torch is 10× smaller.

### Solution: dedicated `eval-worker` service

A dedicated eval worker shares the **existing RabbitMQ broker** (no new infra) but has
its own image, its own queue, and its own concurrency:

```
Airflow DAG  →  "eval" queue (RabbitMQ)  →  eval-worker (CPU torch, transformer models)
                "celery" queue            →  celery-worker (LLM inference, GPU)
```

The existing `celery-worker` is deliberately minimal: single concurrency, solo pool,
purpose-built for vLLM inference. Adding BERTScore there would bloat its image,
compete for its single concurrency slot, and contend with live user requests.

#### Practical steps

1. Add an `eval-worker` service in `docker-compose.yaml` pointing to a new
   `infra/docker/eval-worker/Dockerfile`.
2. Strip `bert-score` (and therefore `torch`/CUDA) from the `airflow` extra in
   `pyproject.toml`; add it to a new `eval-worker` extra, installing `torch` from the
   CPU-only index (`https://download.pytorch.org/whl/cpu`).
3. In `eval_dags.py`, change `_calculate_metrics_task` to dispatch a Celery task to the
   `"eval"` queue and block on `AsyncResult` until done — Airflow waits, the
   eval-worker executes.
4. The eval-worker image installs `experiments/scripts/eval/` code — this is also a
   forcing function to make that code properly installable (no more `sys.path.insert`
   workarounds in `eval_dags.py`).

**K8s payoff:** the eval-worker image built now becomes a `pod_override` image for
`KubernetesExecutor` later. Instead of a persistent eval-worker pod, each metric
calculation task spins up an ephemeral pod from this image — no throwaway work.

---

## 3. Dependency Cleanup (`pyproject.toml` `airflow` extra)

### Remove: Flask/Werkzeug/SQLAlchemy compat pins

```toml
# REMOVE — these are Airflow 2.x internal constraints, not yours.
# The base image already enforces them; putting them here pollutes the lock file.
"flask>=2.2,<2.3",
"werkzeug>=2.2,<3.0",
"sqlalchemy>=1.4,<2.0",
"flask-cors>=3.0,<5.0",
```

### Replace: `mlflow` → `mlflow-skinny`

Airflow DAGs only log metrics and read run data — they don't serve models. `mlflow-skinny`
provides the tracking client without pulling in `gunicorn`, `graphene`, `scikit-learn`,
`scipy`, `skops`, etc.

```toml
# Before
"mlflow>=3.8.1,<4",

# After
"mlflow-skinny>=3.8.1,<4",
```

### Relax: boto/S3 upper bounds

```toml
# Current — overly tight, effectively allows only a single minor version
"boto3>=1.35.36,<1.36.0",
"botocore>=1.35.36,<1.36.0",
"s3fs>=2024.9.0,<2025.0.0",   # already stale

# Better — only pin aiobotocore; let it constrain boto3/botocore as it already does
"aiobotocore>=2.15.0,<3.0.0",
"s3fs>=2024.9.0",
# boto3 and botocore: remove explicit pins, resolved transitively
```

### Target `airflow` extra after cleanup

```toml
airflow = [
    # S3 / DVC
    "aiobotocore>=2.15.0,<3.0.0",
    "s3fs>=2024.9.0",
    "aiohttp>=3.9.2,<4",
    "cffi>=1.9,<2.0.0",
    "dvc[s3]>=3.66.1",
    # Data ingestion
    "arxiv>=2.1.3",
    "beautifulsoup4>=4.12.3",
    "lxml>=5.3.0",
    "requests>=2.32.4",
    "datasets>=4.4.2",
    # Vector store & text processing
    "qdrant-client>=1.13.2",
    "langchain-text-splitters>=0.4.2",
    "numpy>=1.26",
    # DB / infra
    "psycopg2-binary>=2.9,<3",
    "httpx>=0.28.1",
    "protobuf>=5.26.0",
    # Experiment tracking (client only)
    "mlflow-skinny>=3.8.1,<4",
]
# Removed: bert-score, torch/CUDA, flask, werkzeug, sqlalchemy, flask-cors, boto3, botocore
```

Regenerate the lock file after this change — it will shrink dramatically.

---

## 4. Kubernetes Readiness

### 4a. Executor: `LocalExecutor` → `KubernetesExecutor`

`LocalExecutor` runs tasks as subprocesses on the scheduler pod. In K8s, the right
executor is `KubernetesExecutor`: each task runs in its own ephemeral pod with full
resource isolation. Define a `pod_override` per task to use a GPU-capable image for
metric tasks and a lightweight image for data ingestion tasks.

### 4b. DAG distribution: bind mount → GitSync

```yaml
# Current (docker-compose, single host only)
- ${PROJECT_ROOT}/dags:/opt/airflow/dags

# K8s: GitSync sidecar in Helm values
dags:
  gitSync:
    enabled: true
    repo: https://github.com/your-org/agent-042.git
    branch: main
    subPath: dags/
```

### 4c. Log storage: local volume → S3

With multiple pods, logs on a local volume are unreachable from the webserver pod.
S3 credentials are already present in every service.

```ini
AIRFLOW__LOGGING__REMOTE_LOGGING: "true"
AIRFLOW__LOGGING__REMOTE_BASE_LOG_FOLDER: "s3://your-bucket/airflow/logs"
AIRFLOW__LOGGING__REMOTE_LOG_CONN_ID: "aws_default"
```

### 4d. Init containers

| docker-compose service | K8s equivalent |
|---|---|
| `airflow-init` | Helm hook (`helm.sh/hook: pre-install,pre-upgrade`) |
| `airflow-prepare-dirs` | Pod `initContainer` or dropped entirely (PVC `fsGroup`) |

### 4e. `sys.path.insert` in `eval_dags.py`

```python
# Current — couples the entire project tree to the DAG runtime
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
```

With GitSync, only `dags/` is present in the pod — `PROJECT_ROOT/src` won't exist.
Fix: package `src/` and `experiments/scripts/eval/` as proper installable packages
baked into the image. This is already required by the eval-worker work above.

### 4f. Hardcoded base URL

```yaml
# Move this to a Helm values override per environment
AIRFLOW__WEBSERVER__BASE_URL: https://agent.antonlab.ru:8443/airflow
```

---

## Summary: Prioritized Action Plan

### Now (pre-K8s)

| # | Action | Effort |
|---|---|---|
| 1 | Remove `bert-score` from `airflow` extra | Low |
| 2 | Replace `mlflow` → `mlflow-skinny` | Low |
| 3 | Remove Flask/Werkzeug/SQLAlchemy compat pins | Low |
| 4 | Relax boto/S3 upper bounds | Low |
| 5 | Regenerate `requirements.lock` (will shrink ~90%) | Low |
| 6 | Upgrade to `apache/airflow:3.1.8` (fix healthcheck cmd + remove `EXPOSE_CONFIG`) | Low |
| 7 | Add `eval-worker` service + Dockerfile + `eval` extra | Medium |
| 8 | Refactor `_calculate_metrics_task` to dispatch to `eval` queue | Medium |
| 9 | Make `experiments/scripts/eval/` properly installable | Medium |

### K8s phase (Helm rollout)

| # | Action |
|---|---|
| 10 | Deploy via official Airflow Helm chart (`apache-airflow/airflow`) |
| 11 | Switch to `KubernetesExecutor` |
| 12 | Enable GitSync for DAG distribution |
| 13 | Configure remote S3 logging |
| 14 | Define `pod_override` with `eval-worker` image for metric tasks (replaces persistent eval-worker pod) |
| 15 | Drop `airflow-prepare-dirs` — handle via PVC `fsGroup` |
| 16 | Move `AIRFLOW__WEBSERVER__BASE_URL` to Helm values |
