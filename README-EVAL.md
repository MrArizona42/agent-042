# Evaluation & Benchmarking — Design Document

This document describes the architecture, data model, UI, and implementation plan for the
evaluation and benchmarking subsystem of agent-042.

---

## Table of Contents

1. [Goals and Non-Goals](#1-goals-and-non-goals)
2. [Evaluation Taxonomy and Datasets](#2-evaluation-taxonomy-and-datasets)
3. [Pinning Configs — What Exactly Am I Evaluating?](#3-pinning-configs--what-exactly-am-i-evaluating)
4. [Database Schema](#4-database-schema)
5. [Hydra-Driven Eval Runner](#5-hydra-driven-eval-runner)
6. [Parameter Sweeps](#6-parameter-sweeps)
7. [LoRA Adapter Loading for Eval](#7-lora-adapter-loading-for-eval)
8. [LLM-as-Judge — Gemini 2.0 Flash](#8-llm-as-judge--gemini-20-flash)
9. [Regression vs Full Eval](#9-regression-vs-full-eval)
10. [Integrating Evaluation with MLflow Experiment Tracking](#10-integrating-evaluation-with-mlflow-experiment-tracking)
11. [Code Generation Evaluation (Sandboxed Execution)](#11-code-generation-evaluation-sandboxed-execution)
12. [Streamlit Multi-Page Migration](#12-streamlit-multi-page-migration)
13. [Implementation Plan](#13-implementation-plan)

---

## 1. Goals and Non-Goals

### Goals

- Every eval run is **fully reproducible**: the exact model, adapter, RAG index, dataset split,
  embedding model, chunking strategy, and judge config are recorded.
- Results are stored in **PostgreSQL** (the same `agent042` database used by the app), queryable
  by structured config dimensions.
- A dedicated **Streamlit benchmarks page** shows results, comparisons, and drill-downs.
- Two eval tiers: **regression** (fast, ~100–500 examples) and **full eval**
  (on-demand, hours-long, full dataset splits).
- Code generation eval runs generated code in a **sandboxed environment**.
- The system can evaluate **any** adapter version from MLflow, not just the currently loaded
  champion.
- The eval runner is **Hydra-driven** — consistent with the training pipeline, reproducible
  configs, override-friendly CLI.

### Non-Goals

- Real-time online evaluation (user-facing A/B testing) — out of scope for now.
- Evaluating third-party commercial models — we only evaluate our own vLLM-served models.
- A separate eval vLLM instance — single-server setup uses the production vLLM with adapter
  swapping.

---

## 2. Evaluation Taxonomy and Datasets

The project has three task types, each requiring different metrics and methodologies.

### Tasks, metrics, and datasets

| Task | Primary metrics | Judge method | Dataset |
|---|---|---|---|
| **Chat (QA)** | Relevance (1–5), Correctness (1–5), BERTScore, ROUGE-L | LLM-as-judge + automatic | HotpotQA, Natural Questions |
| **Summarization** | Faithfulness (1–5), Coverage (1–5), BERTScore, ROUGE-L | LLM-as-judge + automatic | ArXiv-summarization |
| **Code generation** | Executable rate, Test pass rate (pass@1) | Sandboxed execution | HumanEval |

Additionally, **RAG-specific metrics** are evaluated independently of generation:

| Metric | What it measures | Dataset |
|---|---|---|
| Recall@k | Fraction of queries with ≥1 relevant doc in top-k | MS MARCO, BEIR-SciFact, BEIR-NFCorpus |
| nDCG@k | Ranking quality weighted by position | Same |
| Groundedness | Fraction of answer claims supported by retrieved docs | Evaluated alongside chat/summarization runs |

### Dataset sizes (as fetched in `experiments/notebooks/prefetch_assets.ipynb`)

Full datasets currently on disk:

| Dataset | Split | Rows | Disk size | Purpose |
|---|---|---|---|---|
| `ccdv/arxiv-summarization` | train | 203,037 | 6.9 GB | LoRA training (summarization) |
| `ccdv/arxiv-summarization` | validation | 6,436 | (incl. above) | Eval — summarization |
| `nvidia/OpenCodeInstruct` | train | 5,000,000 | 18 GB | LoRA training (code) |
| `openai/openai_humaneval` | test | 164 | < 1 MB | Eval — code (small enough for regression too) |
| `natural_questions` | validation (5%) | 392 | 169 MB | Eval — QA |
| `hotpot_qa` (distractor) | validation (5%) | 370 | 2.3 MB | Eval — QA |
| `ms_marco` (v1.1) | validation (5%) | 502 | 2.1 MB | Eval — retrieval |
| `BeIR/scifact` | corpus | 5,183 | 7.6 MB | Eval — retrieval |
| `BeIR/nfcorpus` | corpus | 3,633 | 5.7 MB | Eval — retrieval |

> **Note on subsets:** HotpotQA, Natural Questions, and MS MARCO were downloaded as 5% subsets
> in the notebook. The **full** validation splits are much larger: HotpotQA ~7,400 rows,
> Natural Questions ~7,800 rows, MS MARCO validation ~100k rows.
> For full eval, either download the complete splits or use the currently available subsets
> as the "full" tier, and prepare smaller fixed regression splits from them.

### Regression vs full split sizes

| Dataset | Regression split | Full split |
|---|---|---|
| HotpotQA | 100–200 (from current 370) | 370 (current 5%) or full 7.4k |
| Natural Questions | 100–200 (from current 392) | 392 (current 5%) or full 7.8k |
| ArXiv-summarization | 200–500 (from 6,436 val) | 6,436 |
| HumanEval | all 164 (small enough) | all 164 |
| MS MARCO | 100–200 (from current 502) | 502 (current 5%) or full ~100k |
| BEIR-SciFact | all 5,183 (corpus eval) | all 5,183 |
| BEIR-NFCorpus | all 3,633 (corpus eval) | all 3,633 |

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
  dataset_split: str           # "validation" | "test" | "regression_200"
  dataset_dvc_hash: str | None # exact content hash from DVC
  task: str                    # "chat" | "summarize" | "code"
  judge_model: str | None      # "gemini-2.0-flash" | None (for automatic-only metrics)
  bert_score_model: str        # "roberta-large" — pinned so scores stay comparable across runs

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

## 5. Hydra-Driven Eval Runner

### Why Hydra?

The training pipeline already uses Hydra (`experiments/scripts/train_hydra.py` +
`experiments/conf/`). Using Hydra for eval too gives us:

- **Consistency**: same config language for training and evaluation.
- **CLI overrides**: `python run_eval.py task=chat tier=regression adapter.version=5` — no
  argparse boilerplate, every config field is overridable.
- **Config composition**: base config + task-specific overrides + RAG overrides, composed
  via Hydra defaults.
- **Reproducibility**: Hydra auto-saves the resolved config to the run output directory.
  Combined with the JSONB snapshot in Postgres, the exact config is preserved in two places.
- **Sweep support**: Hydra's `--multirun` with sweep syntax is the standard way to run
  parameter grid searches (see [§6](#6-parameter-sweeps)).

### Config structure

```
experiments/
  conf/
    config.yaml               # existing (training entrypoint)
    eval_config.yaml           # new: eval entrypoint
    eval/
      task/
        chat.yaml              # task-specific defaults for chat
        summarize.yaml         # task-specific defaults for summarization
        code.yaml              # task-specific defaults for code
      rag/
        default.yaml           # baseline RAG config
        no_rag.yaml            # rag_enabled: false
        arxiv.yaml             # KB=arxiv + recommended params
        pytorch_docs.yaml      # KB=pytorch_docs + code-aware chunking
      judge/
        gemini.yaml            # Gemini 2.0 Flash config
        none.yaml              # no judge (automatic metrics only)
      metrics/
        default.yaml           # BERTScore model and ROUGE settings
```

### `eval_config.yaml` — top-level eval config

```yaml
defaults:
  - eval/task: chat
  - eval/rag: default
  - eval/judge: gemini
  - eval/metrics: default
  - _self_

# ── Model ──
model:
  base_model: /models/Qwen/Qwen3-0.6B
  vllm_base_url: http://localhost:8000

# ── Adapter (nullable — omit or set to null for base model) ──
adapter:
  name: null            # e.g. "lora-summarization"
  version: null         # e.g. 3 (MLflow version number) or "champion"
  # Resolved at runtime:
  # mlflow_run_id: ...

# ── Eval ──
tier: regression        # regression | full
dataset:
  name: hotpotqa
  split: validation
  max_examples: 200     # null = use all
  seed: 42              # for reproducible subsampling

# ── Generation ──
generation:
  temperature: 0.1
  top_p: 0.95
  max_tokens: 512

# ── Output ──
db_url: ${oc.env:GATEWAY_AGENT042_DB_URL}

# ── Automatic metrics ──
metrics:
  bert_score_model: roberta-large  # see eval/metrics/default.yaml

hydra:
  run:
    dir: experiments/logs/eval-logs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

### `eval/task/chat.yaml`

```yaml
task: chat
dataset:
  name: hotpotqa
  split: validation
  max_examples: 200
metrics:
  - relevance
  - correctness
  - rouge_l
  - bert_score  # model pinned in eval/metrics/default.yaml
```

### `eval/task/summarize.yaml`

```yaml
task: summarize
dataset:
  name: arxiv-summarization
  split: validation
  max_examples: 500
metrics:
  - faithfulness
  - coverage
  - rouge_l
  - bert_score  # model pinned in eval/metrics/default.yaml
```

### `eval/task/code.yaml`

```yaml
task: code
dataset:
  name: humaneval
  split: test
  max_examples: null  # all 164
metrics:
  - executable_rate
  - pass_at_1
judge:
  enabled: false      # code uses sandbox execution, not LLM judge
```

### `eval/rag/default.yaml`

```yaml
rag:
  enabled: true
  knowledge_base: arxiv
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  chunking_strategy: fixed_token
  chunk_size: 512
  chunk_overlap: 50
  retrieval_top_k: 5
  score_threshold: 0.35
  reranking_strategy: none
```

### `eval/metrics/default.yaml`

```yaml
metrics:
  bert_score_model: roberta-large
  # BERTScore uses its own internal encoder — NOT the RAG embedding model.
  # roberta-large is the library default; it is well-calibrated and widely used
  # as a reference point in the literature (~1.4 GB one-time download).
  #
  # Alternative: microsoft/deberta-xlarge-mnli — stronger correlation with human
  # judgments per the BERTScore paper, but ~2.3 GB and noticeably slower.
  #
  # IMPORTANT: BERTScore values are NOT comparable across different model choices.
  # Always fix bert_score_model and record it in EvalConfig so that results from
  # different runs remain comparable.
```

### `eval/judge/gemini.yaml`

```yaml
judge:
  enabled: true
  model: gemini-2.0-flash
  api_key: ${oc.env:GEMINI_API_KEY}
  max_rpm: 14            # stay under 15 RPM free tier
  timeout: 30.0
  structured_output: true
```

### Entry point

```python
# experiments/scripts/eval/run_eval.py
@hydra.main(config_path="../../conf", config_name="eval_config", version_base=None)
def main(cfg: DictConfig) -> None:
    eval_config = build_eval_config(cfg)
    run_evaluation(eval_config)
```

### CLI examples

```bash
# Regression eval for chat task (uses defaults from eval_config.yaml + eval/task/chat.yaml)
python experiments/scripts/eval/run_eval.py

# Full eval for summarization with specific adapter
python experiments/scripts/eval/run_eval.py \
  eval/task=summarize \
  tier=full \
  dataset.max_examples=null \
  adapter.name=lora-summarization \
  adapter.version=3

# Code eval (no judge, no RAG)
python experiments/scripts/eval/run_eval.py \
  eval/task=code \
  eval/rag=no_rag

# Override any parameter
python experiments/scripts/eval/run_eval.py \
  eval/rag=arxiv \
  rag.retrieval_top_k=10 \
  generation.temperature=0.3
```

### Execution flow

```
1. Hydra resolves config (composition + CLI overrides)
2. Build EvalConfig from resolved OmegaConf
3. Resolve versions:
   - Query MLflow for adapter_version → get mlflow_run_id
   - Snapshot Qdrant collection (if RAG enabled)
   - Read dataset DVC hash
4. Create eval_runs row in Postgres (status='running')
5. Prepare inference:
   - If specific adapter requested → sync to production vLLM (see §7)
   - Resolve which model/adapter name to pass in vLLM requests
6. Load dataset (subsample to max_examples with seed if set)
7. For each example (with progress bar):
   a. If RAG enabled: retrieve context via RAG service
   b. Build prompt (system + optional context + user input)
   c. Call vLLM for generation
   d. Compute automatic metrics (ROUGE-L, BERTScore)
   e. If task == "code": run sandbox execution
   f. If judge enabled: call LLM-as-judge (with rate limiting)
   g. Write eval_examples row
8. Compute aggregate metrics → write eval_metrics rows
9. Update eval_runs: status='completed', finished_at=now()
10. Optionally log metrics to MLflow (see §10)
11. Print summary to stdout
```

---

## 6. Parameter Sweeps

### The combinatorial problem

Even modest options produce dozens of RAG combinations:

```
KBs:        arxiv, pytorch_docs                    = 2
Chunking:   fixed_token, code, section_aware       = 3
Retrieval:  dense, sparse (BM25), hybrid           = 3
Reranking:  none, cross_encoder                    = 2
Top-k:      3, 5, 10                               = 3
```

That's 2 × 3 × 3 × 2 × 3 = **108 combinations** per task, per adapter version. Running all of
them on every eval is infeasible.

### Hydra multirun for sweeps

Hydra's `--multirun` flag with comma-separated values is the natural way to sweep:

```bash
# Single-dimension ablation: sweep top_k while holding everything else at baseline
python experiments/scripts/eval/run_eval.py --multirun \
  rag.retrieval_top_k=3,5,10,20

# Two-dimension sweep: chunking × reranking
python experiments/scripts/eval/run_eval.py --multirun \
  rag.chunking_strategy=fixed_token,section_aware \
  rag.reranking_strategy=none,cross_encoder

# Full grid of some interesting dimensions
python experiments/scripts/eval/run_eval.py --multirun \
  rag.retrieval_top_k=3,5,10 \
  rag.chunking_strategy=fixed_token,section_aware,code \
  eval/rag=arxiv,pytorch_docs
```

Each combination becomes a separate Hydra run with its own output directory and its own
`eval_runs` row in Postgres. The UI can then compare them by grouping on the config dimensions
that differ.

### Sweep strategy

1. **Baseline first**: Run every task with the default RAG config. This is your anchor point.

2. **Single-dimension ablations**: Sweep one parameter at a time against the baseline. This
   isolates the effect of each parameter and is feasible even on a single GPU:
   - `rag.retrieval_top_k=3,5,10,20` (4 runs)
   - `rag.chunking_strategy=fixed_token,code,section_aware` (3 runs)
   - `rag.reranking_strategy=none,cross_encoder` (2 runs)

3. **Targeted grids**: Based on ablation results, run 2–3 dimension grids on the most
   impactful parameters.

4. **Always use regression splits for sweeps** — keep each run fast so the total wall time
   stays manageable.

### Hydra sweep plugins

For more advanced sweeps (Bayesian optimization, random search), Hydra supports plugins like
[Optuna Sweeper](https://hydra.cc/docs/plugins/optuna_sweeper/). This can be added later if
grid search proves too expensive. The config structure stays the same.

---

## 7. LoRA Adapter Loading for Eval

### Background

The production vLLM instance runs with `--enable-lora --max-loras 4` and loads adapters from
`assets/adapters/lora-modules.json`, which is generated by `AdapterSyncer.sync()`. Currently
it syncs only `champion` aliases.

On a single-server setup with one GPU, a dedicated eval vLLM instance is infeasible — it would
compete for the same GPU memory. Instead, we load the target adapter into the **production
vLLM**.

### How vLLM LoRA loading works

vLLM can serve multiple LoRA adapters simultaneously (up to `--max-loras`). The adapter is
selected **per request** via the `model` field in the OpenAI-compatible API:

```json
{
  "model": "lora-summarization",
  "messages": [...]
}
```

If the adapter is listed in `lora-modules.json` and loaded, vLLM routes the request to that
adapter. The base model is always available as-is.

### Eval flow for a specific adapter version

1. **Extend `AdapterSyncer`** with a `sync_version(model_name, version)` method:
   - Downloads the specific adapter version from MLflow (not just champion).
   - Saves to `assets/adapters/<model_name>-v<version>/`.
   - Appends an entry to `lora-modules.json`.

2. **vLLM runtime LoRA loading**: vLLM supports loading new adapters at runtime via the
   `VLLM_ALLOW_RUNTIME_LORA_UPDATING=true` environment variable and the
   `POST /v1/load_lora_adapter` API endpoint. This means we can load an eval adapter without
   restarting the server:

   ```bash
   curl -X POST http://localhost:8000/v1/load_lora_adapter \
     -d '{"lora_name": "lora-summarization-v3", "lora_path": "/adapters/lora-summarization-v3"}'
   ```

3. **The eval runner** (`run_eval.py`):
   - Reads `adapter.name` and `adapter.version` from Hydra config.
   - If `adapter.version` is set: calls `sync_version()`, then
     `POST /v1/load_lora_adapter` to load it into the running vLLM.
   - If `adapter.version` is `"champion"` or `null`: resolves the current champion version
     and uses the already-loaded adapter.
   - Sets the `model` field in all vLLM requests to the adapter name.
   - Records the exact version number and MLflow run ID in the eval config.

4. **Cleanup**: After the eval run completes, the eval runner can optionally unload the eval
   adapter via `POST /v1/unload_lora_adapter` to free GPU memory. This is optional — vLLM
   manages LoRA memory within the `--max-loras` budget.

### Important: eval during production hours

Since we share the production vLLM, running a heavy eval with many requests will affect
latency for real users. Best practice:
- Run full evals during off-hours or when no users are active.
- Regression evals (200 examples) are fast enough to run anytime.

---

## 8. LLM-as-Judge — Gemini 2.0 Flash

### The circularity problem

Using the same model that generated the answer as the judge inflates scores. We need an
external, stronger judge.

### Why Gemini 2.0 Flash

- **Free tier**: 15 RPM, 1M tokens/day via Google AI Studio — sufficient for regression
  (200–500 examples ≈ 100k–200k judge tokens).
- **Structured output**: Gemini supports `response_mime_type: "application/json"` with a
  schema, avoiding fragile output parsing.
- **Quality**: Strong enough to judge small open-source models reliably.
- **Cost for full eval**: Even beyond free tier, $0.10/1M input tokens makes full evals cheap.

### Judge prompt design

For each eval task, a structured prompt asks the judge to score on specific rubrics:

**Chat/QA rubric:**
```
You are evaluating an AI assistant's answer to a question.

Question: {question}
Reference answer: {reference}
Model answer: {generated}

Score on two dimensions (1-5 each):
- Relevance: Does the answer address the question directly?
- Correctness: Are the factual claims accurate given the reference?

Return JSON: {"relevance": <int>, "correctness": <int>}
```

**Summarization rubric:**
```
You are evaluating a summary of a scientific article.

Original article (excerpt): {source}
Model summary: {generated}

Score on two dimensions (1-5 each):
- Faithfulness: Does the summary introduce unsupported claims? (5 = perfectly faithful)
- Coverage: Does it cover the main points of the source? (5 = comprehensive)

Return JSON: {"faithfulness": <int>, "coverage": <int>}
```

**Groundedness (RAG-specific):**
```
Given these retrieved documents and this answer, what fraction of
the answer's claims are supported by the documents?

Retrieved documents: {retrieved_docs}
Answer: {generated}

Return JSON: {"groundedness": <float>}   (0.0–1.0)
```

### Implementation

A `JudgeClient` class in `experiments/scripts/eval/judge.py`:

```python
class GeminiJudge:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", max_rpm: int = 14):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.min_interval = 60.0 / max_rpm

    def score_qa(self, question, answer, reference) -> dict:
        """Returns {"relevance": int, "correctness": int}"""

    def score_summary(self, source, summary) -> dict:
        """Returns {"faithfulness": int, "coverage": int}"""

    def score_groundedness(self, answer, retrieved_docs) -> dict:
        """Returns {"groundedness": float}"""
```

- Rate limiting: sleeps between calls to stay under `max_rpm`.
- Retries on 429/5xx with exponential backoff.
- Structured JSON output enforced via Gemini's `response_mime_type`.

### Configuration

```yaml
# experiments/conf/eval/judge/gemini.yaml
judge:
  enabled: true
  model: gemini-2.0-flash
  api_key: ${oc.env:GEMINI_API_KEY}
  max_rpm: 14
  timeout: 30.0
  structured_output: true
```

### Environment variable

```bash
# .env (or experiments/.env)
GEMINI_API_KEY=<your-google-ai-studio-api-key>
```

---

## 9. Regression vs Full Eval

### Two tiers, different purposes

| | Regression | Full eval |
|---|---|---|
| **Purpose** | "Did we break anything?" | "How good is this config overall?" |
| **Dataset size** | 100–200 fixed examples | Full dataset split (hundreds to thousands) |
| **Runtime** | Minutes | Hours |
| **When to run** | After training, before promotion; as sanity check | On-demand, before major releases, for sweep analysis |
| **Trigger** | CLI command, post-training hook | CLI command |
| **Judge calls** | Yes (200 max — well within free tier) | Yes (batched, may hit rate limits) |
| **Stored in** | Same `eval_runs` table, `tier = 'regression'` | Same table, `tier = 'full'` |

### Regression dataset creation

Fixed, reproducible subsets, subsampled by seed in the eval config:

```yaml
# Regression run — the default
tier: regression
dataset:
  name: hotpotqa
  split: validation
  max_examples: 200
  seed: 42
```

The `max_examples + seed` approach means we don't need separate DVC-tracked regression splits.
The eval runner does `ds.shuffle(seed=seed).select(range(max_examples))` at load time. The
seed is recorded in the eval config JSONB, so the exact subset is reproducible.

For a **full** eval:

```yaml
tier: full
dataset:
  name: hotpotqa
  split: validation
  max_examples: null   # no subsampling
```

### Where eval code lives

Eval requires live infrastructure (vLLM, Qdrant, embeddings service). It is **not** in
`tests/` — that directory is for unit/integration tests that mock services and don't need GPU.

```
experiments/
  scripts/
    eval/
      __init__.py
      run_eval.py               # Hydra entry point
      config.py                 # EvalConfig pydantic model
      judge.py                  # Gemini judge client
      metrics.py                # ROUGE-L, BERTScore computation
      sandbox.py                # code execution sandbox
      humaneval.py              # HumanEval dataset loader
```

---

## 10. Integrating Evaluation with MLflow Experiment Tracking

### The training → eval feedback loop

The training pipeline (`train_hydra.py`) already:
1. Trains a LoRA adapter.
2. Logs training metrics (loss, learning rate, val loss) to an MLflow run.
3. Registers the adapter as a new version in MLflow Model Registry.
4. Stores artifacts (adapter weights) in S3.

Evaluation should **close the loop** by attaching eval metrics to the same lineage, so that
in the MLflow UI you can see training metrics, eval metrics, and the adapter version all in one
place.

### Strategy: two-level MLflow logging

#### Level 1: Log eval metrics to the training run

When the eval runner knows the `adapter_mlflow_run_id` (resolved from MLflow Model Registry),
it can log eval metrics **to that same run** as post-hoc metrics:

```python
import mlflow

with mlflow.start_run(run_id=adapter_mlflow_run_id):
    mlflow.log_metrics({
        "eval/rouge_l": 0.42,
        "eval/bert_score": 0.78,
        "eval/relevance_mean": 3.8,
        "eval/correctness_mean": 4.1,
        "eval/pass_at_1": 0.65,
    })
    mlflow.log_params({
        "eval/tier": "regression",
        "eval/dataset": "hotpotqa",
        "eval/dataset_split": "validation",
        "eval/max_examples": 200,
        "eval/judge_model": "gemini-2.0-flash",
    })
```

This means the MLflow experiment view shows training runs with their associated eval scores
side by side — you can sort by `eval/rouge_l` to find the best adapter.

**Limitation**: MLflow params are write-once. If you run eval multiple times on the same
training run (e.g., regression then full), use a `step` parameter or eval-specific param keys
like `eval_full/rouge_l` vs `eval_regression/rouge_l`.

#### Level 2: Dedicated eval experiment in MLflow

Additionally, create a separate MLflow experiment called `eval` where each eval run is its own
MLflow run. This gives a dedicated view for comparing eval configs:

```python
mlflow.set_experiment("eval")
with mlflow.start_run(run_name=f"eval-{task}-{adapter_name}-v{adapter_version}"):
    mlflow.log_params(eval_config.dict())  # full config as params
    mlflow.log_metrics(aggregate_metrics)
    mlflow.log_artifact(hydra_config_path)  # the resolved Hydra config YAML
```

This experiment is useful for:
- Comparing eval runs across different adapter versions.
- Comparing different RAG configs on the same adapter.
- Sweep results visualization (MLflow has built-in parallel coordinates plots).

#### When each level applies

| Scenario | Level 1 (training run) | Level 2 (eval experiment) |
|---|---|---|
| Post-training regression | Yes — attach to the training run | Yes |
| Standalone full eval | Only if adapter_mlflow_run_id known | Yes |
| RAG-only eval (no adapter) | No (no training run) | Yes |
| Sweep runs | No (too many) | Yes — each sweep point is a run |

### Post-training automation

After `train_hydra.py` completes successfully, it can auto-trigger a regression eval:

```python
# In train_hydra.py (or a wrapper script)
def main(cfg):
    save_dir, logs_dir = run_training(cfg)

    # Auto-trigger regression eval if configured
    if cfg.get("auto_eval", False):
        subprocess.run([
            "python", "experiments/scripts/eval/run_eval.py",
            f"adapter.name={cfg.mlflow.registered_model_name}",
            "adapter.version=latest",
            "tier=regression",
        ], check=True)
```

Or more practically, as a separate step in a Makefile or shell script:

```bash
# Makefile
train-and-eval:
	python experiments/scripts/train_hydra.py $(TRAIN_ARGS)
	python experiments/scripts/eval/run_eval.py \
		adapter.name=$(ADAPTER_NAME) \
		adapter.version=latest \
		tier=regression
```

### Gating promotion on eval results

The eval runner can enforce a quality gate before promoting an adapter to `champion`:

```python
# In run_eval.py, after computing metrics:
if cfg.get("gate_promotion", False):
    current_champion = get_champion_metrics(adapter_name)
    if new_metrics["rouge_l"] < current_champion["rouge_l"] * 0.95:
        log.warning("ROUGE-L dropped >5% vs champion. Skipping promotion.")
        return
    promote_adapter(adapter_name, adapter_version, alias="champion")
```

This keeps the `champion` alias stable — a new adapter only gets promoted if it meets or
exceeds the current champion's eval scores.

---

## 11. Code Generation Evaluation (Sandboxed Execution)

### Why it's different

For code tasks, LLM-as-judge and ROUGE are unreliable proxies. The only trustworthy signal is:
does the generated code **run** and **pass the test cases**?

HumanEval provides 164 problems (all on disk, < 1 MB), each with:
- A function signature + docstring (the prompt).
- A set of test assertions (the ground truth).

### Sandbox design

Generated code must run in an **isolated environment** to prevent:
- File system damage (malicious or buggy `os.remove`, `shutil.rmtree`).
- Network access (data exfiltration).
- Excessive resource consumption (infinite loops, memory bombs).

### Implementation: subprocess with resource limits

For a self-hosted system, the simplest safe approach is a subprocess with `firejail` isolation:

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
                timeout=timeout + 5,
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
- Lightweight: ~1ms overhead per invocation.

### HumanEval integration

```python
# experiments/scripts/eval/humaneval.py
from datasets import load_from_disk

def load_humaneval(dataset_path: str):
    ds = load_from_disk(dataset_path)
    return [
        {
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "test": row["test"],
            "entry_point": row["entry_point"],
        }
        for row in ds["train"]  # saved as "train" split in prefetch notebook
    ]
```

The eval runner sends each `prompt` to the model, receives generated code, concatenates it with
`test`, and feeds it to `execute_code()`. Metrics:
- `executable_rate = sum(executable) / total`
- `pass_at_1 = sum(tests_passed) / total`

All 164 problems are small enough to always run in full — no regression/full split needed.

---

## 12. Streamlit Multi-Page Migration

### Current state

`src/ui/app.py` is a single-file Streamlit app. Adding a benchmarks page requires migration to
Streamlit's [multi-page app structure](https://docs.streamlit.io/get-started/multipage-apps).

### Target structure

```
src/ui/
  app.py              # entrypoint: shared init (auth, client, settings, page config)
  client.py           # GatewayClient (unchanged, extended with eval methods)
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

**1. Run list & filtering**
- Table of recent eval runs with columns: date, task, tier, adapter, dataset, status.
- Filters: by task, by tier, by adapter, by date range.
- Click a run to see details.

**2. Run detail view**
- Full config display (collapsible JSON).
- Aggregate metrics table.
- Per-example drill-down: input, reference, generated output, scores.
- For code tasks: execution status, error messages.
- For RAG runs: retrieved documents with scores.

**3. Comparison view**
- Select 2+ runs to compare.
- Side-by-side metric deltas (absolute and %).
- Highlight which config dimensions differ between runs.
- Chart: metric values across runs (bar chart or line over time).

**4. Trends view**
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

## 13. Implementation Plan

### Dependencies

New Python packages (add to `pyproject.toml` under a new `eval` extras group):

```toml
[project.optional-dependencies]
eval = [
    "rouge-score",
    "bert-score",
    "google-genai",       # Google AI Studio SDK (Gemini judge)
    "datasets",           # HuggingFace datasets (HumanEval loading)
]
```

System package: `firejail` — install via `apt install firejail` on the host.

Environment variables:

```bash
GEMINI_API_KEY=<google-ai-studio-api-key>
```

### New file tree

```
experiments/
  conf/
    eval_config.yaml                # Hydra entrypoint for eval
    eval/
      task/
        chat.yaml
        summarize.yaml
        code.yaml
      rag/
        default.yaml
        no_rag.yaml
        arxiv.yaml
        pytorch_docs.yaml
      judge/
        gemini.yaml
        none.yaml
  scripts/
    eval/
      __init__.py
      run_eval.py                   # Hydra-driven CLI entry point
      config.py                     # EvalConfig pydantic model
      judge.py                      # Gemini judge client
      metrics.py                    # ROUGE-L, BERTScore (roberta-large)
      sandbox.py                    # firejail code execution
      humaneval.py                  # HumanEval dataset loader

src/
  shared/
    db/
      models.py                     # +EvalRun, EvalMetric, EvalExample
  gateway/
    api/
      v1/
        eval.py                     # read-only eval endpoints
  ui/
    app.py                          # refactored: shared init only
    pages/
      1_Chat.py                     # extracted chat page
      2_Benchmarks.py               # new benchmarks dashboard
```

### Phases

#### Phase 1: Foundation — database + Hydra eval skeleton

1. Add `EvalRun`, `EvalMetric`, `EvalExample` models to `src/shared/db/models.py`.
2. Create Hydra config structure (`eval_config.yaml` + `eval/task/`, `eval/rag/`, `eval/judge/`).
3. Create `experiments/scripts/eval/` package:
   - `run_eval.py` — Hydra entry point, main loop skeleton.
   - `config.py` — `EvalConfig` pydantic model.
   - `metrics.py` — ROUGE-L and BERTScore computation (`roberta-large` by default, controlled by `metrics.bert_score_model`).
4. Implement basic eval flow for **chat** task (automatic metrics only, no judge).
5. Verify: run a chat regression eval, see results in Postgres.

#### Phase 2: LLM-as-Judge + adapter loading

6. Implement `judge.py` — `GeminiJudge` class with structured output and rate limiting.
7. Extend `AdapterSyncer` with `sync_version()` method.
8. Add vLLM runtime LoRA loading (call `POST /v1/load_lora_adapter` from eval runner).
9. Wire judge scoring into the eval loop.
10. Verify: run chat eval with judge, see relevance/correctness scores in DB.

#### Phase 3: Code evaluation sandbox

11. Implement `sandbox.py` with `firejail` isolation.
12. Implement `humaneval.py` — load from `assets/datasets/humaneval/`.
13. Wire code eval path into `run_eval.py`.
14. Verify: run code eval, see pass@1 in DB.

#### Phase 4: MLflow integration

15. Implement Level 1 logging (attach eval metrics to the training run).
16. Implement Level 2 logging (dedicated `eval` MLflow experiment).
17. Add `gate_promotion` flag and quality-gated promotion logic.
18. Verify: eval metrics visible in MLflow UI alongside training metrics.

#### Phase 5: Streamlit migration + benchmarks page

19. Migrate `app.py` to multi-page structure (`pages/1_Chat.py`).
20. Add eval read endpoints to gateway (`/v1/eval/runs`, etc.).
21. Build `pages/2_Benchmarks.py` — run list, detail, comparison, trends.
22. Extend `GatewayClient` with eval endpoint methods.
23. Verify: full loop — run eval → see results in UI, compare runs.

#### Phase 6: RAG eval + sweeps

24. Add retrieval-only eval mode (Recall@k, nDCG@k without generation).
25. Add groundedness scoring to the judge.
26. Run initial sweeps with `--multirun`.
27. Build sweep comparison views in the UI.
