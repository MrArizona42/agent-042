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
│  training/lora_ops.ipynb   rag/rag_ops.ipynb   eval/eval_results.ipynb
│  ┌──────────────┐  ┌──────────────────┐  ┌─────────────────────┐  │
│  │ § LoRA Ops   │  │ § RAG Ops        │  │ § Eval Results      │  │
│  │ inspect run  │  │ build challenger │  │ query eval_runs     │  │
│  │ register     │  │ assign alias     │  │ compare table       │  │
│  │ promote      │  │ spot-check       │  │ trend plots         │  │
│  │ sync to vLLM │  │ promote alias    │  │ trigger eval ───────┼──┘
│  │ trigger eval─┼──┼──────────────────┼──┼─────────────────────┼──┘
│  └──────────────┘  └──────────────────┘  └─────────────────────┘  │
└────────────────────────────────────────────────────────────────┘

Adapter-sync container: standalone sync logic via
    python -m shared.model_registry sync  (unchanged)
```

**Deleted:** `scripts/manage_registry.py`, `scripts/manage_rag.py`.
`scripts/` keeps only infra shell scripts (`dump_docker_logs.sh`,
`fetch_logs_ssh.sh`, `update_locks.sh`).

---

## Reorganise `experiments/scripts/` within `experiments/`

Experiment orchestration code is not production library code. It stays in
`experiments/`, reorganised into three subdirectories that mirror the three
experiment domains. Notebooks are distributed into those same subdirectories
so that exports (tables, plots) live next to the notebook that produces them.
`experiments/scripts/` is deleted after the move.

```
experiments/
  training/
    conf/                              ← Hydra configs (shared by all training runs)
    train_adapter/
      __init__.py
      start_train.py
      pipeline.py, config.py,
      data_module.py, lit_module.py,
      modeling.py, mlflow_utils.py   ← from experiments/scripts/train_adapter/
    lora_ops.ipynb                 ← LoRA manual operations (NEW)
  rag/
    __init__.py
    notebook_ops.py               ← notebook wrappers around production rag.ops
    sandboxes/                    ← notebook-only experimental forks of production code
    rag_ops.ipynb                  ← RAG manual operations (NEW)
  eval/
    __init__.py
    eval_scripts/
      metrics/
      runner.py, datasets.py,
      retrieval_bench.py
    eval_results.ipynb             ← eval results + comparison tables (NEW)
    debug_eval.ipynb               ← moved
  misc_ops/
    mlflow_quickref.ipynb          ← moved from experiments/notebooks/
    prefetch_assets.ipynb          ← moved
    postgres_diagnostics.ipynb     ← moved
```

`PYTHONPATH` in every container must include **both** the project root and
`src/` so all imports resolve without hacks:

```
ENV PYTHONPATH=/opt/airflow/project:/opt/airflow/project/src
```

All `sys.path.insert` calls are removed. Imports become natural:

```python
# before (in a DAG or build script)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rag.chunking import get_chunker

# after
from rag.chunking import get_chunker                         # via src/ on PYTHONPATH
from rag.ops.update import update_arxiv_collection          # production ops via src/
```

---

## `artifacts/` Directory

All runtime-generated outputs that are not source inputs go here. The
directory is **gitignored** in full.

```
artifacts/
  training/
    checkpoints/   ← Lightning ModelCheckpoint output (top-3 by val_loss)
    hydra/         ← Hydra timestamped output dirs (.hydra/ dumps, overrides.yaml)
  infra/
    logs/          ← Airflow/Docker logs downloaded manually via fetch_logs_ssh.sh
```

Adapter weights downloaded by `AdapterSyncer.download_adapter()` stay in
`assets/adapters/` — they are model inputs to the serving stack, not
generated byproducts.

Notebook exports (comparison tables, plots) live next to their notebook in
`experiments/training/`, `experiments/rag/`, and `experiments/eval/`.
There is no central export directory.

---

## RAG Pipeline Config

`src/shared/knowledge_bases.json` is the single authority for what exists at
runtime: which tasks are supported, which KBs serve each task, and which aliases
each KB exposes. It does **not** store chunking params — those live only in
the `_meta` sentinel inside each Qdrant collection, written at build time and
read by production update ops on incremental or replace refreshes.

### Chunking factory (`get_chunker`)

`get_chunker(strategy, **kwargs)` in `src/rag/chunking.py` maps string keys
to chunker classes. The supported keys after this plan are:

| Key | Class |
|---|---|
| `fixed_token` | `FixedTokenChunker` |
| `code` | `CodeChunker` |
| `section_aware` | `SectionAwareChunker` |

**Two bugs exist in the current factory that are fixed together in task #6:**

1. The parameter is named `task`, not `strategy`. The name comes from Qdrant
   retrieval task types, which is a different concept. Every caller passes a
   chunking strategy string (e.g. `"fixed_token"`) but the parameter is named
   `task`, which is misleading and wrong. Rename the parameter to `strategy`.

2. `section_aware` is not a registered key. The factory only recognises
   `summarize` as the key for `SectionAwareChunker` (an internal task-type
   name that leaked into the config). Add `section_aware` as a direct key and
   keep `summarize` as a deprecated alias.

The old experiment build scripts have been deleted. Notebook operations now call
production workflows from `src/rag/ops`, while notebook-only experimental code
lives under `experiments/rag/sandboxes/` and is never imported by production.

Adding a new strategy:
1. Add class to `src/rag/chunking.py`
2. Add key to `get_chunker()` factory
3. Build a new collection through `experiments.rag.notebook_ops.create_*()` so `_meta` records the new key

No DAG, no notebook, no gateway code needs to change.

### Task + KB + alias config (`knowledge_bases.json`)

The JSON is reorganised around **tasks** — the top-level grouping used by the
discovery API, the daily update DAGs, and the gateway's startup validation.
Under each task sit the KBs that serve it; under each KB sit its aliases.
Chunking params are removed entirely — they live only in `_meta` per Qdrant
collection and are never needed at inference time.

```json
[
  {
    "task": "chat",
    "label": "General knowledge",
    "knowledge_bases": [
      {
        "name": "arxiv",
        "aliases": ["champion", "challenger"],
        "update_strategy": "incremental",
        "label": "ArXiv papers (ML / AI theory)",
        "description": "Deep discussions about ML/AI theory and latest trends"
      }
    ]
  },
  {
    "task": "code",
    "label": "Coding assistance",
    "knowledge_bases": [
      {
        "name": "pytorch_docs",
        "aliases": ["champion", "challenger"],
        "update_strategy": "replace",
        "label": "PyTorch docs",
        "description": "PyTorch documentation for coding assistance"
      }
    ]
  }
]
```

**How this is used in production:**
- **Discovery API** — `GET /v1/knowledge-bases` returns the full task → KB → alias
  tree. The UI populates its KB selector from this with no hardcoding.
- **Daily update DAGs** — read this config to build the set of
  `(kb, alias="champion")` pairs to update, including the `update_strategy` per
  KB (`incremental` vs `replace`). No KB names are hardcoded in the DAG.
- **Gateway startup validation** — iterates all `(kb, alias)` pairs derived from
  the task config and checks Qdrant existence at deploy time (task #8).
- **Request validation** — gateway confirms that every `(kb, alias)` in an
  incoming request exists in this config before querying Qdrant.

**Chunker versioning across champion/challenger:**

A common experiment is replacing one chunker implementation with a better one —
e.g. `CodeChunker` → `ASTCodeChunker`. The strategy key stored in `_meta` is
the version handle; the KB config does not need to know about it.

1. Add `ASTCodeChunker` class and register key `"code_ast"` in the factory.
   This is a one-time code change before the experiment.
2. Build the challenger via the notebook façade:
  `from experiments.rag.notebook_ops import create_pytorch_docs` and then
  `create_pytorch_docs(chunking_strategy="code_ast", ...)`.
  `_meta` records `"chunking_strategy": "code_ast"`.
3. The challenger collection is a **snapshot** — daily update DAGs only update
   `alias="champion"`. The challenger is evaluated as-is and then either
   promoted or dropped.
4. Evaluate. Promote or discard. If promoting: point `champion` alias at the
   challenger collection. From this point on, daily updates run against that
   collection reading its `_meta` (`"chunking_strategy": "code_ast"`).
   If discarding: point alias back to the old champion collection.
   Champion's `_meta` still has `"code"` → `CodeChunker`. Nothing changes in
   production. `ASTCodeChunker` stays in the codebase inertly.

`KnowledgeBaseConfig` in `src/shared/config.py` is updated to match the new
schema:
- Top-level entry becomes `TaskConfig`: `task`, `label`, `knowledge_bases: list[KBConfig]`
- `KBConfig` has: `name`, `aliases: list[str]`, `update_strategy`, `label`, `description`
- `chunking_strategy`, `chunk_size`, `chunk_overlap` fields are deleted
- Build script fallback lines (`chunking_strategy = chunking_strategy or kb_cfg.chunking_strategy`
  etc.) are removed — callers must provide explicit params
- `get_knowledge_bases()` returns `dict[str, TaskConfig]` keyed by task name

**All call sites that depend on the current flat `kb_name → KnowledgeBaseConfig`
map must be updated as part of task #7:**

- `src/gateway/api/v1/openai_compat.py` — request validation calls
  `kb_registry = get_knowledge_bases()` then `kb_registry[src.knowledge_base]`
  and `kb_cfg.aliases`. After the schema change the registry is keyed by task,
  not KB name. Introduce a helper `get_kb_config(kb_name) → KBConfig` that
  searches across all tasks — this is the flat kb-name index used at request
  time.
- `src/gateway/services/rag_service.py` — `_get_retriever()` and
  `available_knowledge_bases()` both call `get_knowledge_bases()` and iterate
  by KB name. Update to use the same `get_kb_config()` helper.
- `src/ui/app.py` — imports `KNOWLEDGE_BASES` (a lazy proxy over
  `get_knowledge_bases()`) and iterates `KNOWLEDGE_BASES.items()` to build
  the KB selector. The proxy in `shared/config.py` must be updated to
  flatten the task-first structure back into `{kb_name: {label, description,
  aliases, update_strategy}}` so the UI and any other legacy caller keeps
  working without changes to those files.
- `src/shared/config.py` `_KBProxy` — update `_ensure()` to iterate the
  new task-first structure when populating the flat dict.
- `src/gateway/api/v1/knowledge_bases.py` — currently returns a flat
  per-KB list. Update to return the task-grouped structure so the UI can
  show task categories if desired (or keep flat — decision at implementation
  time, but the endpoint must be updated either way).

### Gateway startup validation

Currently the gateway creates retrievers **lazily on first request**. A KB
alias listed in `knowledge_bases.json` that does not exist in Qdrant is only
discovered when a user makes a request — `_get_retriever()` silently adds it
to `self._unavailable` and returns `None`. The misconfiguration is invisible
until traffic hits it.

**Task #8 adds explicit startup validation:**

In `src/gateway/main.py`, the `lifespan()` function calls
`rag_service.validate_knowledge_bases()` immediately after the RAG service is
initialised. This method iterates every `(kb, alias)` pair in
`knowledge_bases.json`, resolves the alias to a Qdrant collection name, and
checks whether the collection exists. Missing collections are **logged as
warnings, not errors** — the service starts regardless, and missing aliases
end up in `_unavailable` as before. The benefit is that the operator sees the
problem at deploy time in the startup log, not at user-request time.

```python
# src/gateway/services/rag_service.py
def validate_knowledge_bases(self) -> None:
    """Check every alias in config resolves to an existing Qdrant collection.
    Logs warnings for missing collections; does not raise."""
    for task_cfg in self._task_configs.values():
        for kb_cfg in task_cfg.knowledge_bases:
            for alias in kb_cfg.aliases:
                collection = self._resolve_collection(kb_cfg.name, alias)
                if not self._qdrant.collection_exists(collection):
                    logger.warning(
                        "KB alias not found in Qdrant at startup: "
                        "task=%s kb=%s alias=%s collection=%s — marking unavailable",
                        task_cfg.task, kb_cfg.name, alias, collection,
                    )
                    self._unavailable.add((kb_cfg.name, alias))
```

This is also the right place to surface Qdrant connectivity issues at startup
rather than on the first user request.

---

### LoRA Experiment Flow

LoRA experiments change the adapter, not the code. The codebase stays the same.

```
Airflow: train_lora DAG (fire & forget)
  → trains adapter, logs to MLflow, returns run_id

Jupyter: experiments/training/lora_ops.ipynb
  → inspect run metrics
  → register adapter: registry.register_adapter(run_id=..., artifact_path="model", model_name=...)
  → promote to challenger: registry.promote(..., alias="challenger")
  → sync to vLLM: syncer.sync()
  → trigger eval DAGs
  → compare champion vs challenger
  → promote to champion (or discard)
```

### RAG Experiment Flow A: New Collection (same code)

Change chunking **parameters** — strategy, chunk_size, overlap — without
writing any new code. All three strategies (`fixed_token`, `code`,
`section_aware`) are registered in the factory after task #6.

```
Jupyter: experiments/rag/rag_ops.ipynb
  → from experiments.rag.notebook_ops import create_arxiv
  → call create_arxiv(chunking_strategy="section_aware",
                      chunk_size=256, chunk_overlap=25,
                      alias="challenger")
     → creates new timestamped collection (e.g. arxiv_20260326_143012)
     → writes _meta sentinel with build config
  → spot-check retrieval quality (embed query, inspect results)
  → trigger retrieval eval DAGs (champion vs challenger)
  → compare recall@10, nDCG@10
  → promote challenger → champion (or drop it)
```

The daily Airflow RAG DAGs pass `alias="champion"` explicitly — only the
collection behind `champion` is updated. Challenger collections are built
once by the notebook and evaluated as a snapshot; they do not receive
continuous updates. The `_meta` sentinel stores the build config so the
champion collection uses consistent chunking params across daily runs.

### RAG Experiment Flow B: New Strategy (new code)

Add a new chunking strategy, reranker, or retrieval approach. This
requires code changes in `src/rag/` followed by a collection rebuild.

```
1. Write code:
  - New chunker → add class to src/rag/chunking.py
                  → add key to get_chunker() factory
                  (no change to knowledge_bases.json — chunking is not stored there)
  - New reranker → add class to src/rag/rerankers.py (new file)
                   → add key to get_reranker() factory
  - New retriever variant → modify src/rag/retriever.py
  - Notebook-only prototype → keep it under experiments/rag/sandboxes/<experiment>/...
                     until it is ready to graduate into src/rag/

2. Build challenger collection (Jupyter: `experiments/rag/rag_ops.ipynb`):
  → call `create_arxiv(..., chunking_strategy="my_new_strategy", alias="challenger")`

3. Evaluate (same as Flow A):
   → trigger eval DAGs, compare, promote or discard

4. Activate in production:
   - Chunker: automatic — collection's _meta records the strategy,
    production update ops and daily DAGs use it for incremental updates.
   - Reranker: one-line change in gateway's rag_service.py:
       Retriever(..., reranker=get_reranker("cross_encoder"))
     To roll back: change back to reranker=None.
```

The factory pattern (`get_chunker`, `get_reranker`) keeps production
switching to a single line. Old strategies stay in the codebase —
rolling back means pointing to the old factory key, not reverting code.

---

## Cleanup: Dead Config Files

Legacy references to `src/RAG/config.py` are obsolete. Production code reads
settings directly from `shared.config`, and RAG-specific operator flows go
through `src/rag/ops` plus `experiments/rag/notebook_ops.py`.

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
1. `train_adapter` — `BashOperator` that invokes `start_train.py` as a
   subprocess:

   ```bash
   cd /opt/airflow/project && \
   python -m experiments.training.train_adapter.start_train \
     experiment={{ params.experiment_config }} \
     {{ params.hydra_overrides | join(' ') }}
   ```

   `run_training()` is modified to return `(run_id, save_dir, logs_dir)` —
   `run_id` is printed on the last line and captured as XCom. Using a
   subprocess instead of a `PythonOperator` avoids Hydra global-state
   collisions when multiple DAG runs are active on the same Celery worker,
   and keeps `config_path="../conf"` valid (`start_train.py` resolves it
   relative to its own location → `experiments/training/conf/`).

   `hydra_overrides` is a JSON array of strings in `key=value` format where
   values are **scalars only** (no spaces, no shell-special characters, no
   nested lists). Example: `["training.lr=2e-5", "lora.r=16"]`. For
   complex non-scalar config changes, add a named Hydra experiment YAML
   under `experiments/training/conf/experiment/` and pass it via
   `experiment_config` instead.

**What it does NOT do:** register, promote, or sync. Those are manual decisions
made in `experiments/training/lora_ops.ipynb` after inspecting the training run.

**Infrastructure change:** training runs on the dedicated `airflow-worker-gpu`
service (task #3), not on the existing `airflow-worker`. The training DAG
routes the `train_adapter` task to a named Celery queue so only the GPU worker
picks it up:

```python
train_task = BashOperator(
    task_id="train_adapter",
    bash_command="...",
    queue="gpu",          # routes to airflow-worker-gpu only
)
```

The `airflow-worker-gpu` service starts its Celery worker listening on the
`gpu` queue:

```yaml
# docker-compose.yaml — airflow-worker-gpu service
command: celery worker -Q gpu --concurrency 1
```

The existing `airflow-worker` service listens on the default queue (implicitly
`celery`) and is unchanged. Tasks without an explicit `queue=` continue to
route there.

**Alternatives considered:**
- `DockerOperator` that launches a dedicated training container — cleaner
  isolation but adds complexity (image build, GPU scheduling via Docker).
  Can migrate later if the worker image becomes unmanageable.

---

### 2. Operations Notebooks

The monolithic `operations.ipynb` is replaced by three focused notebooks, one
per subdomain. Each notebook's exports (CSV tables, plots) are saved in the
same directory as the notebook.

---

**File:** `experiments/training/lora_ops.ipynb`

Replaces `register_pretrained_loras.ipynb` and all LoRA CLI usage.

**§0 Setup** — imports, MLflow/vLLM connections (Docker-internal URLs).

**§1 Inspect Training Runs**
- List recent MLflow runs (filter by experiment, status, date)
- View metrics, params, artifacts for a specific run

**§2 Register & Promote**
- Register adapter from run: `registry.register_adapter(run_id=..., artifact_path="model", model_name=...)`
  (`artifact_path="model"` is the fixed path used by the training pipeline)
- List registered adapters + versions
- Promote to challenger/champion: `registry.promote(model, version, alias=...)`
- Demote (remove alias): `registry.demote(model, alias=...)`

**§3 Sync & Verify**
- Sync to vLLM: `syncer.sync()` — hot-load adapters without restart
- Show current production adapters

**§4 Trigger LoRA Eval**
- Call `trigger_eval_dag(dag_id, metric, rag_aliases, lora_aliases)` via Airflow REST API
  (`POST /api/v1/dags/{dag_id}/dagRuns`). Both alias lists are required — the DAG
  validates that neither is empty. When triggering from here, pass `rag_aliases=["champion"]`
  (or whichever RAG collection to test against) alongside the LoRA aliases under evaluation.

---

**File:** `experiments/rag/rag_ops.ipynb`

**§0 Setup** — imports, Qdrant/embeddings connections (Docker-internal URLs).

**§1 Inspect Collections**
- List Qdrant aliases and their target collections
- Inspect `_meta` sentinel (chunking strategy, params, build timestamp)

**§2 Build Challenger** (supports Flow A and Flow B)
- Import notebook wrappers from `experiments.rag.notebook_ops`
- Call `create_arxiv(...)` / `create_pytorch_docs(...)` for a fresh collection,
  or `refresh_arxiv(...)` / `refresh_pytorch_docs(...)` when intentionally re-running
  an existing alias from `_meta`
- Creates new timestamped collection, writes `_meta`, and can attach the requested alias immediately

**§3 Spot-check**
- Embed a query, search, inspect results from champion vs challenger

**§4 Promote / Discard**
- Promote challenger → champion
- Drop stale collection (or defer to cleanup DAG)

**Boundary rule**
- Notebook-only experiments live under `experiments/rag/sandboxes/` and are never imported by
  Gateway, Airflow, or production eval code. Promote the code into `src/rag/` before promoting a
  sandbox-built retrieval approach to champion.

**§5 Trigger RAG Eval**
- Call `trigger_eval_dag(dag_id, metric, rag_aliases, lora_aliases)` via Airflow REST API.
  Both alias lists are required. When triggering from here, pass `lora_aliases=["champion"]`
  (or whichever LoRA adapter to hold fixed) alongside the RAG aliases under evaluation.

**`trigger_eval_dag` helper contract:**
```python
import os, requests

def trigger_eval_dag(
    dag_id: str,
    metric: str,
    rag_aliases: list[str],
    lora_aliases: list[str],
) -> dict:
    """Trigger an eval DAG run via the Airflow REST API."""
    base_url = os.environ["AIRFLOW_BASE_URL"]  # e.g. http://airflow-webserver:8080
    user = os.environ["AIRFLOW_USER"]
    password = os.environ["AIRFLOW_PASSWORD"]
    resp = requests.post(
        f"{base_url}/api/v1/dags/{dag_id}/dagRuns",
        auth=(user, password),
        json={"conf": {"metric": metric, "rag_aliases": rag_aliases, "lora_aliases": lora_aliases}},
    )
    resp.raise_for_status()
    return resp.json()
```

The payload lands in `dag_run.conf`. The eval DAG's `_resolve_params()` reads
`custom_params` first — `conf` keys flow through exactly as `custom_params`
overrides, which take precedence over the Airflow UI param defaults. The
`metric`, `rag_aliases`, and `lora_aliases` keys match what `_resolve_params()`
extracts directly from `custom`. No DAG code change is needed.

`AIRFLOW_BASE_URL`, `AIRFLOW_USER`, and `AIRFLOW_PASSWORD` must be set in the
Jupyter container's environment (add to `docker-compose.yaml` for the jupyter service).

---

**File:** `experiments/eval/eval_results.ipynb`

**§0 Setup** — SQLAlchemy connection to `eval_runs` table.

**§1 Query Results**
- Filter `eval_runs` by task, dataset, date range
- Champion vs challenger comparison table

**§2 Trend Plots**
- Metric trends over time (matplotlib/plotly)

**§3 Export**
- Save comparison tables and plots in the same directory as this notebook
  (`experiments/eval/`).

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

Add training dependencies to a new **`airflow-worker-gpu` extra** in
`pyproject.toml`. This duplicates the `airflow-worker` dep list and adds
all training deps on top. It is **not** added to the `conflicts` array —
it is a standalone extra resolved independently.

A new lock file is generated without the CPU torch index flag:
```bash
# added to scripts/update_locks.sh:
"airflow-worker-gpu|--extra airflow-worker-gpu|3.12|infra/docker/airflow-worker/requirements-airflow-worker-gpu.lock|--constraint ${AIRFLOW_CONSTRAINTS}"
```
(No `--extra-index-url cpu` — uv picks GPU torch via `torch-backend = "auto"`)

A new Docker Compose service `airflow-worker-gpu` (or a separate Dockerfile
target) uses this lock. The existing `airflow-worker` service, its lock, and
its Dockerfile are **unchanged**.

Keep a comment in `pyproject.toml` next to `airflow-worker-gpu` noting that
its base deps must be kept in sync with `airflow-worker`.

**GPU contention note:** training DAG and vLLM both need the GPU. Options:
- Manual: only trigger training DAG when vLLM load is low. Implement this one.
- Automated: training DAG's first task calls `docker compose stop vllm`,
  last task calls `docker compose start vllm`. Simple but causes serving
  downtime. Not acceptable for a production system, so we defer it.

---

### 4. Redistribute Existing Notebooks

**Delete:** `experiments/notebooks/register_pretrained_loras.ipynb` — replaced
by `experiments/training/lora_ops.ipynb`.

**Move + update:** `experiments/notebooks/mlflow_quickref.ipynb` →
`experiments/misc_ops/` — remove CLI references, replace with Python API examples.

**Move:** `experiments/notebooks/debug_eval.ipynb` →
`experiments/eval/`

**Move:** `experiments/notebooks/prefetch_assets.ipynb` →
`experiments/misc_ops/`

**Move:** `experiments/notebooks/postgres_diagnostics.ipynb` →
`experiments/misc_ops/`

After redistribution, `experiments/notebooks/` is deleted.

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
| 1 | Reorganise `experiments/scripts/` → `training/`, `rag/`, `eval/`; move `conf/` to `experiments/training/conf/`; rename `train_hydra.py` → `start_train.py`; update PYTHONPATH in all Dockerfiles; redirect `pipeline.py` checkpoint path from `experiments/logs/lightning_logs/checkpoints/` → `artifacts/training/checkpoints/` and Hydra output dir from `experiments/logs/hydra-logs/` → `artifacts/training/hydra/` (both are Hydra interpolations in `conf/config.yaml` and `conf/experiment/train_adapter.yaml`) | — | Medium |
| 2 | Training DAG (`dags/train_lora.py`) | #1 | Medium |
| 3 | Add `airflow-worker-gpu` extra + lock + compose service; update `update_locks.sh` | — | Low |
| 4 | Update daily RAG DAGs to pass `alias="champion"` explicitly | — | Trivial |
| 5 | Operations notebooks (`lora_ops`, `rag_ops`, `eval_results`) | #1 | Medium |
| 6 | Fix `get_chunker()`: rename `task` parameter → `strategy`; add `section_aware` key; keep `summarize` as deprecated alias; remove old mixed build-path hacks from production ops | — | Trivial |
| 7 | `knowledge_bases.json` task-first schema: task → KB → aliases, `update_strategy`/`label`/`description` per KB, remove all chunking fields; update `shared/config.py` to `TaskConfig`/`KBConfig`; add `get_kb_config(kb_name)` flat-lookup helper; update `_KBProxy._ensure()` to flatten task-first structure; update `openai_compat.py` request validation, `rag_service.py` retriever lookup and KB listing, `knowledge_bases.py` discovery endpoint; remove old chunking fallback lines from notebook/ops callers; update daily RAG DAGs to iterate task config | — | Small |
| 8 | Gateway startup validation: add `validate_knowledge_bases()` to `RAGService`; call it from `lifespan()` after RAG service init; warn (not fail) for missing collections | #7 | Small |
| 9 | Remove obsolete `src/RAG/config.py` references and keep `shared.config` as the only live settings entrypoint | — | Trivial |
| 10 | Delete old notebooks, update docs | #5 | Low |

Task #1 goes first — everything else imports from the new paths.
Tasks #6–#8 are independent of #1 and can be done in parallel with the infrastructure work.

---
