# Evaluation & Benchmarking — Design Document

This document describes the architecture, data model, UI, and implementation plan for the
evaluation and benchmarking subsystem of agent-042.

---

## Table of Contents

1. [Goals and Non-Goals](#1-goals-and-non-goals)
2. [Evaluation Taxonomy](#2-evaluation-taxonomy)
3. [Pinning Configs — What Exactly Am I Evaluating?](#3-pinning-configs--what-exactly-am-i-evaluating)
4. [Database Schema](#4-database-schema)
5. [RAG Config Combinatorics](#5-rag-config-combinatorics)
6. [LoRA Adapter Loading](#6-lora-adapter-loading)
7. [LLM-as-Judge Strategy](#7-llm-as-judge-strategy)
8. [Regression vs Full Eval](#8-regression-vs-full-eval)
9. [Code Generation Evaluation (Sandboxed Execution)](#9-code-generation-evaluation-sandboxed-execution)
10. [Streamlit Multi-Page Migration](#10-streamlit-multi-page-migration)
11. [Eval Runner Architecture](#11-eval-runner-architecture)
12. [Implementation Plan](#12-implementation-plan)

---

## 1. Goals and Non-Goals

### Goals

- Every eval run is **fully reproducible**: the exact model, adapter, RAG index, dataset split,
  embedding model, chunking strategy, and judge config are recorded.
- Results are stored in **PostgreSQL** (the same `agent042` database used by the app), queryable
  by structured config dimensions.
- A dedicated **Streamlit benchmarks page** shows results, comparisons, and drill-downs.
- Two eval tiers: **regression** (fast, CI-friendly, ~100–500 examples) and **full eval**
  (on-demand, hours-long, full dataset splits).
- Code generation eval runs generated code in a **sandboxed environment**.
- The system can evaluate **any** adapter version from MLflow, not just the currently loaded
  champion.

### Non-Goals

- Real-time online evaluation (user-facing A/B testing) — out of scope for now.
- Evaluating third-party commercial models — we only evaluate our own vLLM-served models.
- Building a standalone evaluation microservice — the eval runner is a CLI/Airflow task that
  writes to the shared Postgres.

---

## 2. Evaluation Taxonomy

The project has three task types, each requiring different metrics and methodologies:

| Task | Primary metrics | Judge method | Dataset (regression / full) |
|---|---|---|---|
| **Chat (QA)** | Relevance (1–5), Correctness (1–5), BERTScore, ROUGE-L | LLM-as-judge + automatic | HotpotQA (500 / 7.4k), Natural Questions (500 / full) |
| **Summarization** | Faithfulness (1–5), Coverage (1–5), BERTScore, ROUGE-L | LLM-as-judge + automatic | ArXiv-summarization val (500 / full) |
| **Code generation** | Executable rate, Test pass rate (pass@1) | Sandboxed execution | HumanEval (all 164 — small enough for regression too) |

Additionally, **RAG-specific metrics** are evaluated independently of generation:

| Metric | What it measures | Dataset |
|---|---|---|
| Recall@k | Fraction of queries with ≥1 relevant doc in top-k | MS MARCO (500 / full), BEIR-SciFact, BEIR-NFCorpus |
| nDCG@k | Ranking quality weighted by position | Same |
| Groundedness | Fraction of answer claims supported by retrieved docs | Evaluated alongside chat/summarization runs |

---

## 3. Pinning Configs — What Exactly Am I Evaluating?

Every eval run records a **frozen snapshot** of all moving parts. Without this, metrics become
uninterpretable the moment anything changes.

### EvalConfig — the immutable identity of a run

```
EvalConfig:
  # Model
  base_model: str              # e.g. "Qwen/Qwen3-0.6B"
  adapter_name: str | None     # e.g. "lora-summarization" (MLflow registered model name)
  adapter_version: int | None  # e.g. 3 (MLflow model version number, NOT alias)
  adapter_mlflow_run_id: str | None  # for traceability back to the training run

  # RAG
  rag_enabled: bool
  knowledge_base: str | None   # "arxiv" | "pytorch_docs" | None
  qdrant_collection: str | None  # resolved collection name at eval time
  qdrant_snapshot_id: str | None # Qdrant snapshot ID or DVC hash of index artifact
  embedding_model: str         # "sentence-transformers/all-MiniLM-L6-v2"
  chunking_strategy: str       # "fixed_token" | "code" | "section_aware"
  chunk_size: int
  chunk_overlap: int
  retrieval_top_k: int
  score_threshold: float
  reranking_strategy: str | None  # "none" | "cross_encoder" | "llm"

  # Eval
  dataset_name: str            # "hotpotqa" | "arxiv-summarization" | "humaneval" | ...
  dataset_split: str           # "validation" | "test" | "regression_500"
  dataset_dvc_hash: str | None # exact content hash from DVC
  task: str                    # "chat" | "summarize" | "code"
  judge_model: str | None      # "google/gemini-2.0-flash" | None (for automatic-only metrics)

  # Generation params
  temperature: float
  top_p: float
  max_tokens: int
```

This config is serialized as a JSONB column on the eval run record. Every field is queryable.

**Where hashes come from:**
- `adapter_version`: read from MLflow Model Registry at eval start.
- `qdrant_snapshot_id`: call `POST /collections/{name}/snapshots` before eval, record the ID.
  Alternatively, record the DVC hash of the index build script's output.
- `dataset_dvc_hash`: `dvc status` or read from `.dvc` file at eval start.

---

## 4. Database Schema

New tables in the existing `agent042` PostgreSQL database, alongside `users`, `chat_sessions`,
and `chat_messages`.

```sql
-- ──────────────────────────────────────────────
-- Eval run: one row per evaluation execution
-- ──────────────────────────────────────────────
CREATE TABLE eval_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    status          TEXT NOT NULL DEFAULT 'running',  -- running | completed | failed
    tier            TEXT NOT NULL,                     -- regression | full
    task            TEXT NOT NULL,                     -- chat | summarize | code
    config          JSONB NOT NULL,                    -- full EvalConfig snapshot
    -- Denormalized for fast filtering:
    base_model      TEXT NOT NULL,
    adapter_name    TEXT,
    adapter_version INTEGER,
    dataset_name    TEXT NOT NULL,
    dataset_split   TEXT NOT NULL,
    knowledge_base  TEXT,
    error_message   TEXT                               -- populated if status = 'failed'
);

CREATE INDEX idx_eval_runs_task ON eval_runs (task);
CREATE INDEX idx_eval_runs_adapter ON eval_runs (adapter_name, adapter_version);
CREATE INDEX idx_eval_runs_created ON eval_runs (created_at DESC);
CREATE INDEX idx_eval_runs_config ON eval_runs USING gin (config);

-- ──────────────────────────────────────────────
-- Aggregate metrics: one row per metric per run
-- ──────────────────────────────────────────────
CREATE TABLE eval_metrics (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id      UUID NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    metric_name TEXT NOT NULL,       -- "relevance_mean", "rouge_l", "pass_at_1", "recall_at_5", ...
    value       DOUBLE PRECISION NOT NULL,
    UNIQUE (run_id, metric_name)
);

CREATE INDEX idx_eval_metrics_run ON eval_metrics (run_id);

-- ──────────────────────────────────────────────
-- Per-example results: for drill-down and debugging
-- ──────────────────────────────────────────────
CREATE TABLE eval_examples (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id          UUID NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    example_index   INTEGER NOT NULL,             -- position in dataset
    input_text      TEXT NOT NULL,                 -- the question / article / prompt
    reference_text  TEXT,                          -- gold answer / reference summary / expected output
    generated_text  TEXT NOT NULL,                 -- model output
    -- Per-example scores (nullable — not all metrics apply to all tasks)
    relevance       SMALLINT,                      -- 1-5, from judge
    correctness     SMALLINT,                      -- 1-5, from judge
    faithfulness    SMALLINT,                      -- 1-5, from judge
    coverage        SMALLINT,                      -- 1-5, from judge
    rouge_l         DOUBLE PRECISION,
    bert_score      DOUBLE PRECISION,
    -- Code-specific
    executable      BOOLEAN,
    tests_passed    BOOLEAN,
    execution_error TEXT,
    -- RAG-specific
    retrieved_docs  JSONB,                         -- [{source, score, snippet}, ...]
    groundedness    DOUBLE PRECISION               -- 0.0–1.0
);

CREATE INDEX idx_eval_examples_run ON eval_examples (run_id);
```

### Why JSONB for config?

The combinatorial explosion of RAG parameters (chunking × retrieval × reranking × top_k × KB)
makes a fixed set of columns brittle. JSONB lets us:
- Add new config dimensions without migrations.
- Filter with `config->>'chunking_strategy' = 'code'` or
  `config @> '{"retrieval_top_k": 10}'::jsonb`.
- Still have denormalized columns (`task`, `adapter_name`, etc.) for the most common filters.

### SQLAlchemy Models

Add to `src/shared/db/models.py`:

```python
class EvalRun(Base):
    __tablename__ = "eval_runs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    finished_at = Column(DateTime(timezone=True))
    status = Column(String, nullable=False, default="running")
    tier = Column(String, nullable=False)
    task = Column(String, nullable=False)
    config = Column(JSONB, nullable=False)
    base_model = Column(String, nullable=False)
    adapter_name = Column(String)
    adapter_version = Column(Integer)
    dataset_name = Column(String, nullable=False)
    dataset_split = Column(String, nullable=False)
    knowledge_base = Column(String)
    error_message = Column(String)

    metrics = relationship("EvalMetric", back_populates="run", cascade="all, delete-orphan")
    examples = relationship("EvalExample", back_populates="run", cascade="all, delete-orphan")


class EvalMetric(Base):
    __tablename__ = "eval_metrics"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False)
    metric_name = Column(String, nullable=False)
    value = Column(Float, nullable=False)

    run = relationship("EvalRun", back_populates="metrics")

    __table_args__ = (UniqueConstraint("run_id", "metric_name"),)


class EvalExample(Base):
    __tablename__ = "eval_examples"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False)
    example_index = Column(Integer, nullable=False)
    input_text = Column(Text, nullable=False)
    reference_text = Column(Text)
    generated_text = Column(Text, nullable=False)
    relevance = Column(SmallInteger)
    correctness = Column(SmallInteger)
    faithfulness = Column(SmallInteger)
    coverage = Column(SmallInteger)
    rouge_l = Column(Float)
    bert_score = Column(Float)
    executable = Column(Boolean)
    tests_passed = Column(Boolean)
    execution_error = Column(Text)
    retrieved_docs = Column(JSONB)
    groundedness = Column(Float)

    run = relationship("EvalRun", back_populates="examples")
```

---

## 5. RAG Config Combinatorics

Even modest options produce dozens of combinations:

```
KBs:        arxiv, pytorch_docs                    = 2
Chunking:   fixed_token, code, section_aware       = 3
Retrieval:  dense, sparse (BM25), hybrid           = 3
Reranking:  none, cross_encoder, llm               = 3
Top-k:      3, 5, 10                               = 3
```

That's 2 × 3 × 3 × 3 × 3 = **162 combinations** per task, per adapter version. Running all of
them on every eval is infeasible.

### Strategy: structured sweep + targeted comparison

1. **Baseline config**: Define a single "default" RAG config per task (the one currently deployed).
   Every eval run uses this unless explicitly overridden.

2. **Sweep mode**: A CLI flag `--sweep rag` generates a grid of configs from a YAML spec and
   runs each combination on the **regression** dataset split. Results are stored as separate
   `eval_runs` rows, each with its own `config` JSONB — queryable and comparable in the UI.

3. **Single-dimension ablation**: For focused experiments, sweep only one dimension at a time
   (e.g., `--sweep-dim retrieval_top_k --sweep-values 3,5,10,20`) while holding everything
   else at the baseline.

4. **UI comparison view**: The benchmarks page lets you select two or more eval runs and see
   metric deltas side by side, grouped by the config dimension that differs.

### Sweep config example (`experiments/conf/eval_sweep.yaml`)

```yaml
defaults:
  knowledge_base: arxiv
  chunking_strategy: fixed_token
  chunk_size: 512
  chunk_overlap: 50
  retrieval_top_k: 5
  score_threshold: 0.35
  reranking_strategy: none

sweeps:
  retrieval_top_k: [3, 5, 10, 20]
  chunking_strategy: [fixed_token, section_aware]
  reranking_strategy: [none, cross_encoder]
```

---

## 6. LoRA Adapter Loading

### Problem

The gateway/vLLM currently loads whichever adapter has the `champion` alias. To evaluate a
specific adapter version (e.g., a newly trained `challenger`), we need a mechanism to either:
- (a) Tell vLLM to load a specific adapter at eval time, or
- (b) Run a **dedicated eval vLLM instance** with the target adapter.

### Solution: dedicated eval vLLM instance + on-demand adapter sync

Since vLLM supports hot-loading LoRA adapters via `--enable-lora`, and our `AdapterSyncer`
already generates `lora-modules.json`, the approach is:

1. **Add an `eval-vllm` service** to docker-compose — a second vLLM instance on the same GPU
   (or a different one if available). This avoids contention with production inference.
   For a single-GPU setup, the eval vLLM starts only during eval and stops after — managed by
   the eval runner or a dedicated Airflow DAG.

2. **Extend `AdapterSyncer`** with a `sync_version(model_name, version)` method that downloads
   a specific adapter version (not just champion) to a staging directory and writes a temporary
   `lora-modules.json`.

3. **The eval runner** receives `--adapter-name lora-summarization --adapter-version 3` as
   arguments. It:
   - Calls `sync_version()` to download the adapter.
   - Starts (or reconfigures) the eval vLLM instance with the correct `lora-modules.json`.
   - Points all inference requests at the eval vLLM endpoint.
   - Records the exact `adapter_version` and `adapter_mlflow_run_id` in the eval config.

4. **If no adapter is specified**, the eval runner uses the production vLLM endpoint with the
   current champion adapter (or base model if no adapter is loaded). The adapter version is
   still resolved and recorded.

### Docker-compose addition

```yaml
eval-vllm:
  image: vllm/vllm-openai:v0.16.0
  profiles: ["eval"]                  # only starts when explicitly requested
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  volumes:
    - ${MODELS_DIR:-./assets/models}:/models:ro
    - ${EVAL_ADAPTERS_DIR:-./assets/eval_adapters}:/adapters:ro
  command: >
    --model /models/Qwen/Qwen3-0.6B
    --enable-lora
    --max-loras 2
    --max-lora-rank 64
    --lora-modules /adapters/lora-modules.json
    --port 8000
  environment:
    VLLM_ALLOW_RUNTIME_LORA_UPDATING: "true"
  ports:
    - "127.0.0.1:8002:8000"
  networks:
    - backend_net
```

The `profiles: ["eval"]` means this service is not started by default (`docker compose up`).
It only starts with `docker compose --profile eval up eval-vllm`.

---

## 7. LLM-as-Judge Strategy

### The circularity problem

Using the same model that generated the answer as the judge inflates scores. We need an
external, stronger judge.

### Free / low-cost external judge options

| Provider | Model | Free tier | Rate limits | Notes |
|---|---|---|---|---|
| **Google AI Studio** | `gemini-2.0-flash` | Free tier: 15 RPM, 1M tokens/day | Sufficient for regression (500 examples) | Best free option. Supports structured output. |
| **Google AI Studio** | `gemini-2.5-flash-preview-05-20` | Free tier: same limits | Same | Stronger, good for full eval judging. |
| **Groq** | `llama-3.3-70b-versatile` | Free: 30 RPM, 14.4k tokens/min | Tight for full eval, fine for regression | Open-weight alternative. |
| **Mistral** | `mistral-small-latest` | Free tier with API key | 1 RPM free tier — too slow | Only viable as fallback. |

### Recommended setup

- **Primary judge: `gemini-2.0-flash`** via Google AI Studio API (free tier).
  - For regression runs (500 examples), 500 judge calls ≈ 200k tokens — well within 1M/day.
  - For full eval, can spread across hours or use the paid tier ($0.10/1M input tokens).
- **Fallback judge: `llama-3.3-70b-versatile`** via Groq (free tier).
  - Used if Gemini API is down or quota exceeded.
- **No local judge model**: avoids GPU contention with the eval vLLM instance.

### Judge prompt design

For each eval task, a structured prompt asks the judge to score on specific rubrics:

```
Chat/QA rubric:
  - Relevance (1-5): Does the answer address the question?
  - Correctness (1-5): Are the factual claims accurate given the reference?

Summarization rubric:
  - Faithfulness (1-5): Does the summary introduce unsupported claims?
  - Coverage (1-5): Does it cover the main points of the source?

Groundedness (RAG-specific):
  - Given these retrieved documents and this answer, what fraction of
    the answer's claims are supported by the documents? Return a float 0.0–1.0.
```

The judge must return **structured JSON** (Gemini supports `response_mime_type: "application/json"`
with a schema). This avoids fragile output parsing.

### Configuration

```python
# src/shared/config.py — add to Settings
class JudgeSettings:
    judge_provider: str = "google"       # "google" | "groq"
    judge_model: str = "gemini-2.0-flash"
    judge_api_key: str                   # env: GATEWAY_JUDGE_API_KEY
    judge_fallback_provider: str = "groq"
    judge_fallback_model: str = "llama-3.3-70b-versatile"
    judge_fallback_api_key: str          # env: GATEWAY_JUDGE_FALLBACK_API_KEY
    judge_max_rpm: int = 14              # stay under 15 RPM free tier
    judge_timeout: float = 30.0
```

### Implementation

A `JudgeClient` class in `src/eval/judge.py`:
- Accepts provider + model + API key.
- Methods: `score_qa(question, answer, reference) -> {relevance, correctness}`,
  `score_summary(source, summary) -> {faithfulness, coverage}`,
  `score_groundedness(answer, retrieved_docs) -> float`.
- Handles rate limiting (sleep between calls to stay under RPM).
- Falls back to the secondary provider on 429/5xx errors.

---

## 8. Regression vs Full Eval

### Two tiers, different purposes

| | Regression | Full eval |
|---|---|---|
| **Purpose** | "Did we break anything?" | "How good is this config overall?" |
| **Dataset size** | 100–500 fixed examples | Full dataset split (thousands) |
| **Runtime** | Minutes | Hours |
| **When to run** | After training, before promotion; in CI | On-demand, before major releases |
| **Trigger** | CLI command, CI pipeline, post-training hook | CLI command, Airflow DAG |
| **Judge calls** | Yes (500 max) | Yes (batched, respecting rate limits) |
| **Stored in** | Same `eval_runs` table, `tier = 'regression'` | Same table, `tier = 'full'` |

### Regression dataset creation

Fixed, reproducible subsets extracted by seed:

```python
# experiments/scripts/eval/prepare_regression_splits.py
import hashlib
from datasets import load_from_disk

def sample_regression_split(dataset_path, n=500, seed=42):
    ds = load_from_disk(dataset_path)
    split = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    # Save as a separate DVC-tracked file for reproducibility
    split.save_to_disk(f"{dataset_path}_regression_{n}")
    return split
```

These regression splits are DVC-tracked so their content hash is pinned.

### Where regression tests live

Regression eval is **not** in `tests/`. The `tests/` directory is for unit/integration tests
(pytest) that verify code correctness — they mock services and don't need GPU/vLLM.

Regression eval requires live infrastructure (vLLM, Qdrant, embeddings service). It belongs in:

```
experiments/
  scripts/
    eval/
      run_eval.py          # main CLI entry point
      judge.py             # LLM-as-judge client (also usable as src/eval/judge.py)
      metrics.py           # ROUGE, BERTScore computation
      sandbox.py           # code execution sandbox
      prepare_regression_splits.py
```

### Post-training hook

After `train_hydra.py` finishes and registers a new adapter version in MLflow, it can
optionally trigger a regression eval:

```bash
# In training pipeline or CI:
python experiments/scripts/eval/run_eval.py \
  --task summarize \
  --tier regression \
  --adapter-name lora-summarization \
  --adapter-version latest \
  --dataset arxiv-summarization \
  --split regression_500
```

The result is stored in Postgres. The training pipeline can gate promotion on regression metrics
(e.g., "don't promote to champion if ROUGE-L dropped >5% vs. current champion").

### MLflow integration for training metrics

The training pipeline already logs metrics to MLflow. Regression eval metrics can **also** be
logged to MLflow as post-training metrics on the same run — this allows comparing adapter
versions in the MLflow UI as well:

```python
import mlflow
mlflow.log_metrics({
    "eval/rouge_l": 0.42,
    "eval/relevance_mean": 3.8,
    "eval/pass_at_1": 0.65,
}, step=0)
```

This is in addition to (not instead of) storing full results in Postgres.

---

## 9. Code Generation Evaluation (Sandboxed Execution)

### Why it's different

For code tasks, LLM-as-judge and ROUGE are unreliable proxies. The only trustworthy signal is:
does the generated code **run** and **pass the test cases**?

HumanEval provides 164 problems, each with:
- A function signature + docstring (the prompt).
- A set of test assertions (the ground truth).

### Sandbox design

Generated code must run in an **isolated environment** to prevent:
- File system damage (malicious or buggy `os.remove`, `shutil.rmtree`).
- Network access (data exfiltration).
- Excessive resource consumption (infinite loops, memory bombs).

### Implementation: subprocess with resource limits

For a self-hosted system without Kubernetes, the simplest safe approach is a subprocess with
`ulimit` restrictions:

```python
# experiments/scripts/eval/sandbox.py
import subprocess
import tempfile

def execute_code(code: str, test_code: str, timeout: int = 10) -> dict:
    """Run generated code + tests in an isolated subprocess."""
    full_code = code + "\n\n" + test_code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        f.flush()
        try:
            result = subprocess.run(
                [
                    "firejail", "--noprofile",
                    "--net=none",           # no network
                    "--rlimit-as=512000000", # 512MB memory limit
                    "--rlimit-cpu=10",       # 10s CPU time
                    "--rlimit-fsize=1000000", # 1MB file write limit
                    "--quiet",
                    "python3", f.name,
                ],
                capture_output=True,
                text=True,
                timeout=timeout + 5,  # subprocess timeout > firejail CPU limit
            )
            return {
                "executable": True,
                "tests_passed": result.returncode == 0,
                "stdout": result.stdout[:2000],
                "stderr": result.stderr[:2000],
            }
        except subprocess.TimeoutExpired:
            return {
                "executable": True,
                "tests_passed": False,
                "stdout": "",
                "stderr": "Execution timed out",
            }
        except Exception as e:
            return {
                "executable": False,
                "tests_passed": False,
                "stdout": "",
                "stderr": str(e),
            }
```

**Why `firejail`?**
- Available in Ubuntu repos (`apt install firejail`), no Docker-in-Docker needed.
- `--net=none` blocks all network access.
- `--rlimit-*` caps memory, CPU, and disk writes.
- Runs as the current user — no privilege escalation.
- Lightweight: ~1ms overhead per invocation.

**Alternative for Docker environments:** If running inside Docker already (e.g., Airflow worker),
use a second container with `--network=none --memory=512m --cpus=0.5 --read-only` and mount
only the temp file. This is heavier but works when firejail isn't available.

### HumanEval integration

```python
# experiments/scripts/eval/humaneval.py
from datasets import load_dataset

def load_humaneval():
    ds = load_dataset("openai/openai_humaneval", split="test")
    return [
        {
            "task_id": row["task_id"],
            "prompt": row["prompt"],       # function signature + docstring
            "test": row["test"],           # assert statements
            "entry_point": row["entry_point"],
        }
        for row in ds
    ]
```

The eval runner sends each `prompt` to the model, receives generated code, concatenates it with
`test`, and feeds it to `execute_code()`. Metrics:
- `executable_rate = sum(executable) / total`
- `pass_at_1 = sum(tests_passed) / total`

---

## 10. Streamlit Multi-Page Migration

### Current state

`src/ui/app.py` is a single-file Streamlit app. Adding a benchmarks page requires migration to
Streamlit's [multi-page app structure](https://docs.streamlit.io/get-started/multipage-apps).

### Target structure

```
src/ui/
  app.py              # entrypoint: shared init (auth, client, settings, page config)
  client.py           # GatewayClient (unchanged)
  config.py           # settings (unchanged)
  pages/
    1_Chat.py         # current chat functionality (extracted from app.py)
    2_Benchmarks.py   # new eval dashboard
```

### Shared state across pages

Streamlit multi-page apps share `st.session_state`. The entrypoint `app.py` handles:
- Page config (`st.set_page_config`).
- Auth check (redirect to login if not authenticated).
- `GatewayClient` initialization → stored in `st.session_state.client`.
- Sidebar: user profile, logout, navigation.

Each page file in `pages/` accesses the shared client and auth state.

### Benchmarks page features

The `2_Benchmarks.py` page provides:

#### 1. Run list & filtering
- Table of recent eval runs with columns: date, task, tier, adapter, dataset, status.
- Filters: by task, by tier, by adapter, by date range.
- Click a run to see details.

#### 2. Run detail view
- Full config display (collapsible JSON).
- Aggregate metrics table.
- Per-example drill-down: input, reference, generated output, scores.
- For code tasks: execution status, error messages.
- For RAG runs: retrieved documents with scores.

#### 3. Comparison view
- Select 2+ runs to compare.
- Side-by-side metric deltas (absolute and %).
- Highlight which config dimensions differ between runs.
- Chart: metric values across runs (bar chart or line over time).

#### 4. Trends view
- Select a task + metric.
- Time-series chart of that metric across all runs.
- Annotate with adapter version changes.

### API endpoints for the benchmarks page

Add to `src/gateway/api/v1/eval.py`:

```
GET  /v1/eval/runs                    # list runs with filters (task, tier, adapter, date range)
GET  /v1/eval/runs/{run_id}           # single run detail + metrics
GET  /v1/eval/runs/{run_id}/examples  # paginated per-example results
GET  /v1/eval/compare                 # ?run_ids=uuid1,uuid2,uuid3 → comparison data
```

These are read-only endpoints. The eval runner writes directly to Postgres (not through the API).

---

## 11. Eval Runner Architecture

### Entry point: CLI

```bash
python -m experiments.scripts.eval.run_eval \
  --task chat \
  --tier regression \
  --adapter-name lora-chat \
  --adapter-version 5 \
  --dataset hotpotqa \
  --split regression_500 \
  --rag-config experiments/conf/eval_rag_default.yaml \
  --judge google/gemini-2.0-flash
```

### Execution flow

```
1. Parse args → build EvalConfig
2. Resolve versions:
   - Query MLflow for adapter_version → get mlflow_run_id
   - Snapshot Qdrant collection (if RAG enabled)
   - Read dataset DVC hash
3. Create eval_runs row in Postgres (status='running')
4. Prepare inference:
   - If specific adapter requested → sync_version() + start eval-vllm
   - Else → use production vLLM endpoint
5. Load dataset split
6. For each example (with progress bar):
   a. If RAG enabled: retrieve context via RAG service
   b. Build prompt (system + optional context + user input)
   c. Call vLLM for generation
   d. Compute automatic metrics (ROUGE-L, BERTScore)
   e. If task == "code": run sandbox execution
   f. If judge configured: call LLM-as-judge (with rate limiting)
   g. Write eval_examples row
7. Compute aggregate metrics → write eval_metrics rows
8. Update eval_runs: status='completed', finished_at=now()
9. Print summary to stdout
```

### Parallel execution considerations

- **Generation**: Sequential (1 request at a time to vLLM — it handles batching internally).
- **Judge calls**: Sequential with rate-limiting sleep (stay under 15 RPM for free tier).
- **Sandbox execution**: Can be parallelized (CPU-bound, no GPU), but sequential is fine for
  164 HumanEval problems.
- **Metric computation**: Batched (BERTScore works best in batches of 32+).

### Airflow DAG (for scheduled full evals)

```python
# dags/eval_full.py
from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG("eval_full", schedule_interval=None, ...) as dag:
    # Manually triggered — not scheduled
    eval_chat = BashOperator(
        task_id="eval_chat_full",
        bash_command=(
            "python -m experiments.scripts.eval.run_eval "
            "--task chat --tier full "
            "--dataset hotpotqa --split validation "
            "--judge google/gemini-2.0-flash"
        ),
    )
    eval_summarize = BashOperator(
        task_id="eval_summarize_full",
        bash_command=(
            "python -m experiments.scripts.eval.run_eval "
            "--task summarize --tier full "
            "--dataset arxiv-summarization --split validation "
            "--judge google/gemini-2.0-flash"
        ),
    )
    eval_code = BashOperator(
        task_id="eval_code_full",
        bash_command=(
            "python -m experiments.scripts.eval.run_eval "
            "--task code --tier full "
            "--dataset humaneval --split test"
        ),
    )
    # All three can run in parallel (if resources allow) or sequentially
    [eval_chat, eval_summarize, eval_code]
```

---

## 12. Implementation Plan

### Phase 1: Foundation (database + eval runner skeleton)

1. Add `EvalRun`, `EvalMetric`, `EvalExample` models to `src/shared/db/models.py`.
2. Create the `experiments/scripts/eval/` package:
   - `run_eval.py` — CLI arg parsing, main loop skeleton.
   - `metrics.py` — ROUGE-L and BERTScore computation.
   - `config.py` — `EvalConfig` pydantic model.
3. Implement basic eval flow for **chat** task (no judge, automatic metrics only).
4. Write regression split preparation script.
5. Verify: run a chat regression eval, see results in Postgres.

### Phase 2: LLM-as-Judge + adapter loading

6. Implement `judge.py` — Gemini API client with structured output, rate limiting, fallback.
7. Extend `AdapterSyncer` with `sync_version()`.
8. Add `eval-vllm` profile to docker-compose.
9. Wire judge scoring into the eval loop.
10. Verify: run chat eval with judge, see relevance/correctness scores in DB.

### Phase 3: Code evaluation sandbox

11. Implement `sandbox.py` with firejail isolation.
12. Implement `humaneval.py` — dataset loading + prompt construction.
13. Wire code eval into `run_eval.py`.
14. Verify: run code eval, see pass@1 in DB.

### Phase 4: Streamlit migration + benchmarks page

15. Migrate `app.py` to multi-page structure (`pages/1_Chat.py`).
16. Add eval read endpoints to gateway (`/v1/eval/runs`, etc.).
17. Build `pages/2_Benchmarks.py` — run list, detail, comparison, trends.
18. Add `GatewayClient` methods for eval endpoints.
19. Verify: full loop — run eval → see results in UI.

### Phase 5: RAG eval + sweeps

20. Add retrieval-only eval mode (Recall@k, nDCG@k without generation).
21. Implement sweep config parsing and grid execution.
22. Add groundedness scoring to the judge.
23. Build sweep comparison views in the UI.

### Phase 6: Airflow integration + polish

24. Create `dags/eval_full.py` DAG.
25. Configure Airflow connection to app's Postgres.
26. Add post-training regression hook.
27. MLflow metric cross-posting.

---

## Appendix A: New Dependencies

```toml
# pyproject.toml — new extras group
[project.optional-dependencies]
eval = [
    "rouge-score",
    "bert-score",
    "google-genai",       # Google AI Studio SDK (for Gemini judge)
    "groq",               # Groq SDK (fallback judge)
    "datasets",           # HuggingFace datasets (HumanEval loading)
]
```

`firejail` is a system package, installed via `apt install firejail` in the eval environment
(Dockerfile or host).

## Appendix B: Environment Variables

```bash
# .env additions for eval
GATEWAY_JUDGE_API_KEY=<google-ai-studio-api-key>
GATEWAY_JUDGE_FALLBACK_API_KEY=<groq-api-key>
EVAL_VLLM_BASE_URL=http://eval-vllm:8000    # points to eval-vllm service
EVAL_ADAPTERS_DIR=./assets/eval_adapters     # staging dir for eval adapter downloads
```

## Appendix C: File Tree (new files)

```
experiments/
  conf/
    eval_rag_default.yaml       # default RAG config for eval
    eval_sweep.yaml             # sweep grid definition
  scripts/
    eval/
      __init__.py
      run_eval.py               # CLI entry point
      config.py                 # EvalConfig pydantic model
      judge.py                  # LLM-as-judge client
      metrics.py                # ROUGE-L, BERTScore
      sandbox.py                # firejail code execution
      humaneval.py              # HumanEval dataset interface
      prepare_regression_splits.py

src/
  shared/
    db/
      models.py                 # +EvalRun, EvalMetric, EvalExample
  gateway/
    api/
      v1/
        eval.py                 # read-only eval endpoints
  ui/
    app.py                      # refactored: shared init only
    pages/
      1_Chat.py                 # extracted chat page
      2_Benchmarks.py           # new benchmarks dashboard

dags/
  eval_full.py                  # Airflow DAG for on-demand full eval

infra/
  compose/
    docker-compose.yaml         # +eval-vllm service (profile: eval)
```
