# Experimentation Platform — Implementation Plan

## Goal

Replace the broken CLI-driven workflow with two clear entry points:
**Airflow DAGs** for heavy compute (training, evaluation) and
**Jupyter notebooks** for manual decisions (register, promote, sync, inspect).

The CLI scripts (`manage_registry.py`, `manage_rag.py`) never worked outside the
Docker network because they resolve internal hostnames (`mlflow`, `qdrant`, `vllm`).
Instead of fixing them, we remove them entirely — Jupyter runs inside the Docker
network and can call the same Python APIs directly. Airflow DAGs already work.

---

## Architecture

Two entry points for experimenters. No CLI scripts.

```
┌────────────────────────────────────────────────────────────────┐
│                AIRFLOW  (fire & forget)                        │
│                                                                │
│  Training DAGs           RAG Data DAGs        Eval DAGs        │
│  ┌───────────────┐     ┌───────────────┐    ┌──────────────┐  │
│  │ prepare_data  │     │ download_data │    │ fetch_preds  │  │
│  │ train_adapter │     │ dvc_version   │    │ calc_metrics │  │
│  │ log_to_mlflow │     │ build_index   │    │ log_to_db    │  │
│  └───────┬───────┘     └───────────────┘    └──────▲───────┘  │
│          │  "done, run_id=abc123"                   │          │
└──────────┼──────────────────────────────────────────┼──────────┘
           ▼                                          │
┌────────────────────────────────────────────────────────────────┐
│            JUPYTER  (interactive decisions)                     │
│                                                                │
│  operations.ipynb                                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  § LoRA Ops          § RAG Ops         § Eval Results    │  │
│  │  - inspect run       - build challenger - query eval_runs│  │
│  │  - register adapter  - assign alias     - compare table  │  │
│  │  - promote alias     - spot-check       - trend plots    │  │
│  │  - sync to vLLM      - promote alias    - trigger eval ──┼──┘
│  │  - trigger eval ─────────────────────────────────────────┼──┘
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘

Adapter-sync container: standalone sync logic via
    python -m shared.model_registry sync  (unchanged)
```

**Deleted:** `scripts/manage_registry.py`, `scripts/manage_rag.py`.
`scripts/` keeps only infra shell scripts (`dump_docker_logs.sh`,
`fetch_logs_ssh.sh`, `update_locks.sh`).

---

## Move `experiments/scripts/` into `src/`

These modules are shared libraries imported by DAGs and notebooks, not
standalone scripts. Moving them into `src/` eliminates the `sys.path`
hacks that every consumer currently needs.

| Current path | New path | Called by |
|---|---|---|
| `experiments/scripts/train_hydra.py` | `src/training/train_hydra.py` | Training DAG (new) |
| `experiments/scripts/train_adapter/` | `src/training/` | `train_hydra.py` |
| `experiments/scripts/eval/runner.py` | `src/eval/runner.py` | Eval DAGs (existing) |
| `experiments/scripts/eval/datasets.py` | `src/eval/datasets.py` | `runner.py` |
| `experiments/scripts/eval/metrics/` | `src/eval/metrics/` | `runner.py` |
| `experiments/scripts/eval/retrieval_bench.py` | `src/eval/retrieval_bench.py` | Eval DAGs (existing) |
| `experiments/scripts/rag_data/build_arxiv_index.py` | `src/rag_data/build_arxiv_index.py` | RAG DAGs (existing) + operations notebook |
| `experiments/scripts/rag_data/build_pytorch_docs_index.py` | `src/rag_data/build_pytorch_docs_index.py` | RAG DAGs (existing) + operations notebook |

After the move, `experiments/scripts/` is deleted. `experiments/` keeps
only `conf/` (Hydra configs), `notebooks/`, and `logs/`.

All `sys.path.insert(0, .../src)` lines in DAGs and build scripts are
removed — `src/` is already on `PYTHONPATH` in every container. Imports
become natural:

```python
# before (in a DAG or build script)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rag.chunking import get_chunker

# after
from rag.chunking import get_chunker
```

---

## Experiment Flows

### LoRA Experiment Flow

LoRA experiments change the adapter, not the code. The codebase stays the same.

```
Airflow: train_lora DAG (fire & forget)
  → trains adapter, logs to MLflow, returns run_id

Jupyter: operations notebook
  → inspect run metrics
  → register adapter: registry.register_adapter(run_id=...)
  → promote to challenger: registry.promote(..., alias="challenger")
  → sync to vLLM: syncer.sync()
  → trigger eval DAGs
  → compare champion vs challenger
  → promote to champion (or discard)
```

### RAG Experiment Flow A: New Collection (same code)

Change chunking **parameters** — strategy, chunk_size, overlap — without
writing any new code. The existing `get_chunker()` factory in
`src/RAG/chunking.py` already supports `fixed_token`, `code`, and
`section_aware` strategies.

```
Jupyter: operations notebook
  → call build_arxiv(chunking_strategy="section_aware",
                     chunk_size=256, chunk_overlap=25)
     → creates new timestamped collection (e.g. arxiv_20260326_143012)
     → writes _meta sentinel with build config
  → assign challenger alias to the new collection
  → spot-check retrieval quality (embed query, inspect results)
  → trigger retrieval eval DAGs (champion vs challenger)
  → compare recall@10, nDCG@10
  → promote challenger → champion (or drop it)
```

The daily Airflow RAG DAG keeps updating whichever collection `champion`
points to. The `_meta` sentinel stores the build config so incremental
updates use the same chunking params.

### RAG Experiment Flow B: New Strategy (new code)

Add a new chunking strategy, reranker, or retrieval approach. This
requires code changes in `src/RAG/` followed by a collection rebuild.

```
1. Write code:
   - New chunker → add class to src/RAG/chunking.py
                  → add elif to get_chunker()
   - New reranker → add class to src/RAG/rerankers.py (new file)
                   → add elif to get_reranker()
   - New retriever variant → modify src/RAG/retriever.py

2. Build challenger collection (Jupyter: operations notebook):
   → call build_arxiv(..., chunking_strategy="my_new_strategy")
   → assign challenger alias

3. Evaluate (same as Flow A):
   → trigger eval DAGs, compare, promote or discard

4. Activate in production:
   - Chunker: automatic — collection's _meta records the strategy,
     build scripts and daily DAGs use it for incremental updates.
   - Reranker: one-line change in gateway's rag_service.py:
       Retriever(..., reranker=get_reranker("cross_encoder"))
     To roll back: change back to reranker=None.
```

The factory pattern (`get_chunker`, `get_reranker`) keeps production
switching to a single line. Old strategies stay in the codebase —
rolling back means pointing to the old factory key, not reverting code.

---

## Cleanup: Dead Config Files

`src/RAG/config.py` is dead code. It re-exports `Settings` and
`get_settings` from `shared.config` under backward-compat aliases
(`RAGSettings`, `get_rag_settings`). **Nothing imports from it** — every
consumer already uses `from shared.config import get_settings` directly.
Delete it.

This is consistent with the centralized config approach: `shared/config.py`
is the single source of truth. No per-module config wrappers needed.

---

## Implementation Tasks

### 1. Training DAG

**File:** `dags/train_lora.py`

Parameterized DAG that runs LoRA training on the Airflow worker (GPU access).

**Params (Airflow UI dropdowns/inputs):**
- `experiment_config`: Hydra experiment config name (default: `train_adapter`)
- `hydra_overrides`: free-form Hydra overrides as JSON list
  (e.g., `["training.lr=2e-5", "lora.r=16", "trainer.max_epochs=3"]`)

**Tasks:**
1. `train_adapter` — `PythonOperator` that calls the existing
   `train_adapter.pipeline.run_training()` function with Hydra config resolved
   from params. Logs to MLflow. Returns `run_id` via XCom.

**What it does NOT do:** register, promote, or sync. Those are manual decisions
made in the operations notebook after inspecting the training run.

**Infrastructure change:** the `airflow-worker` service needs GPU passthrough
in `docker-compose.yaml` (`deploy.resources.reservations.devices`). Training
deps (torch, peft, bitsandbytes, transformers, lightning) must be added to the
airflow-worker lock file. The Jupyter image can drop training extras.

**Alternatives considered:**
- `DockerOperator` that launches a dedicated training container — cleaner
  isolation but adds complexity (image build, GPU scheduling via Docker).
  Can migrate later if the worker image becomes unmanageable.

---

### 2. Operations Notebook

**File:** `experiments/notebooks/operations.ipynb`

Single notebook with clearly separated sections for all manual operations.
Replaces `register_pretrained_loras.ipynb` and all CLI usage.

**Sections:**

**§0 Setup** — imports, MLflow/Qdrant connections (all Docker-internal URLs).

**§1 LoRA Operations**
- Inspect recent MLflow runs (list, filter, view metrics/params)
- Register adapter from run: `registry.register_adapter(run_id=..., model_name=...)`
- List registered adapters + versions
- Promote to challenger/champion: `registry.promote(model, version, alias=...)`
- Demote (remove alias): `registry.demote(model, alias=...)`
- Sync to vLLM: `syncer.sync()` — hot-load adapters without restart
- Show current production adapters

**§2 RAG Operations** (supports both Flow A and Flow B above)
- List Qdrant aliases and their target collections
- Inspect collection metadata (`_meta` sentinel: chunking strategy, params)
- Build a challenger collection with different chunking params
  (calls `build_arxiv` / `build_pytorch_docs` functions directly)
- Assign challenger alias to the new collection
- Spot-check retrieval quality (embed a query, search, inspect results)
- Promote challenger → champion
- Drop stale collections (or let cleanup DAG handle it)

**§3 Trigger Evaluation**
- Helper function: `trigger_eval_dag(dag_id, metric, rag_aliases, lora_aliases)`
  via Airflow REST API (`POST /api/v1/dags/{dag_id}/dagRuns`)
- Trigger individual eval DAGs
- Trigger full suite (loop over all DAGs for a task)

**§4 Eval Results**
- Query `eval_runs` table (via SQLAlchemy / raw SQL)
- Comparison tables: champion vs challenger per metric
- Trend plots over time (matplotlib/plotly)
- Export results for thesis

---

### 3. Airflow Worker GPU Access

**File:** `infra/compose/docker-compose.yaml`

Add GPU reservation to `airflow-worker` service:

```yaml
airflow-worker:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

Add training dependencies to the airflow-worker lock file
(`infra/docker/airflow-worker/requirements-airflow-worker.lock`).

Remove `training` extra from Jupyter lock file
(`infra/docker/jupyter/requirements-jupyter.lock`) — Jupyter no longer needs
torch/peft/bitsandbytes since training runs in Airflow.

**GPU contention note:** training DAG and vLLM both need the GPU. Options:
- Manual: only trigger training DAG when vLLM load is low.
- Automated: training DAG's first task calls `docker compose stop vllm`,
  last task calls `docker compose start vllm`. Simple but causes serving
  downtime. Acceptable for thesis scope.

---

### 4. Update Existing Notebooks

**Delete:** `experiments/notebooks/register_pretrained_loras.ipynb` (replaced
by §1 of operations notebook).

**Update:** `experiments/notebooks/mlflow_quickref.ipynb` — remove CLI
references, replace with Python API examples.

**Keep as-is:** `debug_eval.ipynb`, `postgres_diagnostics.ipynb`,
`prefetch_assets.ipynb`.

---

### 5. Update Documentation

Update these files to remove CLI references and describe the new workflow:

- `experiments/README.md` — training is now via Airflow, manual ops via notebook
- `README-SYSTEM-DESIGN.md` — update architecture section
- `PLAN-LORA-HOTSWAP.md` — CLI references → notebook/Airflow references
- `PLAN-CLI-AND-EVAL-REFACTOR.md` — mark CLI sections as superseded

---

## Execution Order

| # | Task | Depends on | Effort |
|---|---|---|---|
| 1 | Move `experiments/scripts/` → `src/` + fix imports | — | Medium |
| 2 | Training DAG (`dags/train_lora.py`) | #1 | Medium |
| 3 | Airflow worker GPU + training deps | — | Low |
| 4 | Operations notebook | #1 | Medium |
| 5 | Delete `src/RAG/config.py` | — | Trivial |
| 6 | Drop training extras from Jupyter lock | #3 | Low |
| 7 | Delete old notebooks, update docs | #4 | Low |

Task #1 goes first — everything else imports from the new paths.

---

## Not Covered by This Plan

- **Multi-task training** — only ArxivDataModule exists; adding CodeInstruct or other DataModules is a separate effort.
- **Hyperparameter sweeps** — Hydra `--multirun` with Optuna/Ax sweeper configs are not included.
- **CI/CD pipelines** — no GitHub Actions / GitLab CI.
- **Observability** — no Prometheus, Grafana, Langfuse, or token tracking.
- **Reranking / hybrid search** — RAG quality improvements are not in scope.
- **Code execution sandbox hardening** — current rlimit-only approach is unchanged.
- **Eval result visualization beyond notebook** — no Grafana dashboards.
- **GPU scheduling automation** — no preemption logic; training vs serving contention is managed manually.
- **eval-worker cleanup** — `src/eval_worker/` Celery app is unused but not removed.
- **Config drift fixes** — duplicate Settings classes and Docker Compose `:-` defaults (see CONFIG-AUDIT.md).
