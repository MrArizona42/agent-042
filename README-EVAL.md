# Evaluation & Benchmarking

## 1. Overview

### 1.1 Tasks, metrics, and datasets

Each row below is a separate **eval-suite** = unique `(task, dataset, metric)`.
One Airflow DAG = one eval-suite = one metric.

| Task | Dataset | Metric | Method |
|---|---|---|---|
| Chat (QA) | HotpotQA (validation) | Relevance (1–5) | LLM-as-Judge |
| Chat (QA) | HotpotQA (validation) | Correctness (1–5) | LLM-as-Judge |
| Chat (QA) | HotpotQA (validation) | BERTScore | Automatic |
| Chat (QA) | HotpotQA (validation) | ROUGE-L | Automatic |
| Chat (QA) | Natural Questions (validation) | Relevance (1–5) | LLM-as-Judge |
| Chat (QA) | Natural Questions (validation) | Correctness (1–5) | LLM-as-Judge |
| Chat (QA) | Natural Questions (validation) | BERTScore | Automatic |
| Chat (QA) | Natural Questions (validation) | ROUGE-L | Automatic |
| Summarization | ArXiv-summarization (validation) | Faithfulness (1–5) | LLM-as-Judge |
| Summarization | ArXiv-summarization (validation) | Coverage (1–5) | LLM-as-Judge |
| Summarization | ArXiv-summarization (validation) | BERTScore | Automatic |
| Summarization | ArXiv-summarization (validation) | ROUGE-L | Automatic |
| Code generation | HumanEval (test, 164 examples) | pass@1 | Sandboxed execution |
| Code generation | HumanEval (test, 164 examples) | Executable rate | Sandboxed execution |
| RAG + Chat | HotpotQA (validation) | Groundedness (1–5) | LLM-as-Judge |
| RAG + Chat | Natural Questions (validation) | Groundedness (1–5) | LLM-as-Judge |
| RAG + Code | HumanEval (test) | Groundedness (1–5) | LLM-as-Judge |
| Retrieval-only | MS MARCO (validation) | Recall@10 | Automatic |
| Retrieval-only | MS MARCO (validation) | nDCG@10 | Automatic |
| Retrieval-only | BEIR-SciFact (corpus) | Recall@10 | Automatic |
| Retrieval-only | BEIR-SciFact (corpus) | nDCG@10 | Automatic |
| Retrieval-only | BEIR-NFCorpus (corpus) | Recall@10 | Automatic |
| Retrieval-only | BEIR-NFCorpus (corpus) | nDCG@10 | Automatic |

### 1.2 Storage granularity

Each eval-suite produces **one row per (rag_alias, lora_alias) pair** in `eval_runs`, because
each suite targets exactly one metric.  For example, `eval_chat_hotpotqa_rouge_l` with
`rag_aliases=["champion","challenger"]` and `lora_aliases=["champion"]` produces:
1 metric × 2 rag_aliases × 1 lora_alias = **2 rows**.

### 1.3 LLM-as-Judge

Gemini 2.0 Flash via Google AI Studio API. Supports structured JSON output.
Free tier: 15 RPM, 1M tokens/day.

### 1.4 Code execution sandbox

HumanEval requires executing untrusted generated Python code.

Execution runs in **ephemeral Docker containers** with resource limits:

* Image: minimal Python (e.g., `python:3.11-slim`) — no extra packages beyond stdlib.
* Limits per sample: 1 CPU, 512 MB RAM, 30 s timeout, no network (`--network=none`).
* The eval runner uses the Docker SDK for Python (`docker` package) to create, start, and
  remove a container per code sample.
* The generated code is written to a temp file and bind-mounted read-only into the container.
* Exit code 0 + expected stdout → pass. Non-zero or timeout → fail.

This fits the existing Docker Compose infrastructure. In k8s, the same approach translates to
ephemeral Jobs with resource limits and no network policy — no architectural change needed.

The eval runner container requires access to the Docker socket (`/var/run/docker.sock` mounted
in the Compose service definition).

---

## 2. Eval architecture

### 2.1 Two types of evaluation

The system has two fundamentally different eval types with different execution paths:

**Generation evals** (Chat, Summarization, Code, RAG+Chat, RAG+Code):
- The eval runner calls the **gateway API** (`POST /v1/chat/completions`).
- The gateway is the single source of truth for alias resolution, RAG retrieval, and inference.
- The eval runner never imports `RAG/` or calls vLLM directly.

**Retrieval-only evals** (MS MARCO, BEIR-SciFact, BEIR-NFCorpus):
- These use **benchmark-provided corpora and relevance judgments** — they do NOT query production
  KBs (`arxiv`, `pytorch_docs`).
- The eval runner uses the `RAG/` library directly (not the gateway API) to:
  1. Build a temporary Qdrant collection from the benchmark corpus, using the same build config
     (chunking strategy, embedding model) as the production collection being evaluated.
  2. Run benchmark queries through the embedding + retrieval pipeline.
  3. Compare retrieved results against gold relevance labels.
- The build config is read from the `_meta` sentinel point of the production collection
  (identified by `kb_name + rag_alias`). This ensures the retrieval eval measures the same
  architecture that production uses.
- Temporary benchmark collections are named `eval_{kb}_{dataset}_{rag_alias}_{timestamp}` and
  deleted after the eval completes.

### 2.2 Gateway API extensions for eval

**Returning RAG context for Groundedness evaluation:**

The gateway API response must include the retrieved RAG chunks alongside the generated answer.
This is needed for Groundedness metrics (LLM-as-Judge evaluates whether the answer is supported
by the retrieved context).

The retrieved chunks are returned in a `rag_context` field in the response:

```json
{
    "choices": [{"message": {"role": "assistant", "content": "..."}}],
    "rag_context": [
        {"content": "chunk text...", "score": 0.87, "source": "arxiv_champion"},
        {"content": "chunk text...", "score": 0.82, "source": "arxiv_champion"}
    ]
}
```

When `rag_sources` is provided in the request, the response includes `rag_context`.
When RAG is disabled, `rag_context` is omitted or null.

**LoRA adapter selection:**

vLLM with `--enable-lora` accepts LoRA adapter names through the standard `model` field in the
OpenAI-compatible API. The eval runner selects a LoRA adapter by setting `model` to the adapter
name registered in vLLM (e.g., `lora-summarize`, `lora-code`).

The adapter name is resolved via MLflow Model Registry:

```
lora_alias="champion" → MLflow: get model version by alias → adapter_name="lora-chat", version=3
→ API request: model="lora-chat"
```

For base model evaluation (no LoRA), `lora_alias` is set to `"none"` and `model` uses the
default base model from settings.

### 2.3 Airflow DAG structure

One Airflow DAG = one eval-suite = one `(task, dataset, metric)`.

Each metric gets its own DAG.  This is intentional: LLM-as-Judge metrics may
be unavailable or slow, while automatic metrics are fast and always available.
Separating them at the DAG level gives maximum scheduling flexibility.

DAG naming: ``eval_{task}_{dataset}_{metric}`` for generation evals,
``eval_retrieval_{kb}_{dataset}_{metric}`` for retrieval-only evals.

DAG examples: `eval_chat_hotpotqa_rouge_l`, `eval_chat_hotpotqa_relevance`,
`eval_code_humaneval_pass_at_1`, `eval_retrieval_arxiv_beir_scifact_recall_at_10`,
`eval_summarization_arxiv_faithfulness`.

RAG-specific retrieval evals include the KB in the DAG name since each
retrieval dataset must be indexed using the config of that KB.

Each DAG accepts `rag_aliases` and `lora_aliases` parameters to compare
different RAG and LoRA configurations within one eval run.

DAG steps:

```
1. prepare_config       — resolve aliases, read _meta, snapshot collection, build run config
2. generate_predictions — call gateway API (or build temp collection for retrieval-only)
3. compute_metrics      — calculate the single metric for this suite
4. log_to_db            — write one row per (rag_alias, lora_alias) to eval_runs
5. cleanup              — delete temp collections (retrieval-only evals)
```

### 2.4 Eval-run arguments

Each eval-run takes arguments for cross-configuration comparison:

* `rag_aliases: list[str]` — RAG alias roles (e.g., `["champion", "challenger"]`).
* `lora_aliases: list[str]` — LoRA adapter alias roles (e.g., `["champion", "challenger"]`).

Defaults: `rag_aliases=["champion"]`, `lora_aliases=["champion"]`.

When multiple values are provided, the runner forms the **Cartesian product** and evaluates
each `(rag_alias, lora_alias)` pair independently.

**KB is fixed per eval-suite, not a parameter:**

| Eval suite | Knowledge base | Why |
|---|---|---|
| `eval_chat_hotpotqa_{metric}` | `arxiv` | Chat uses arxiv KB |
| `eval_chat_nq_{metric}` | `arxiv` | Chat uses arxiv KB |
| `eval_code_humaneval_{metric}` | `pytorch_docs` | Code uses pytorch_docs KB |
| `eval_summarization_arxiv_{metric}` | N/A | Summarization never uses RAG |
| `eval_retrieval_arxiv_*_{metric}` | `arxiv` | Tests arxiv build config |
| `eval_retrieval_pytorch_*_{metric}` | `pytorch_docs` | Tests pytorch_docs build config |

The `rag_alias` argument (e.g., `champion`, `challenger`) selects which alias role of the
suite's fixed KB to use. The eval runner constructs the Qdrant alias name as
`{kb_name}_{rag_alias}` — consistent with RAG-IMPROVEMENTS.md alias resolution logic.

For suites where RAG is not applicable (Summarization), `rag_aliases` is ignored.

### 2.5 Retrieval-only eval details

Retrieval-only eval suites follow the same pattern as generation suites: each suite has a
**fixed KB** and iterates over the `rag_aliases` list.

Suite naming: `eval_retrieval_{kb}_{dataset}` — e.g., `eval_retrieval_arxiv_beir_scifact`,
`eval_retrieval_pytorch_msmarco`.

For each `rag_alias` in the list, the runner:

```
1. Resolve {kb_name}_{rag_alias} → get production collection name.
2. Read _meta → extract build_config (chunking_strategy, embedding_model, etc.).
3. Build temporary collection eval_{kb}_{dataset}_{rag_alias}_{timestamp}
   from benchmark corpus using the extracted build_config.
4. Run benchmark queries → compute Recall@k, nDCG@k against gold labels.
5. Log results to eval_runs (with rag_alias and knowledge_base recorded).
6. Delete temporary collection.
```

This means `--rag-aliases champion,challenger` runs the full benchmark **twice** — once
replicating the champion collection's config, once with the challenger's — producing a direct
comparison of retrieval quality between two different build configs for the same KB.

`lora_aliases` is ignored for retrieval-only suites (no generation involved).

---

## 3. Database schema

One row per `(task, dataset, metric, rag_alias, lora_alias)` combination (one row per
alias pair because each eval-suite targets exactly one metric). The full system
configuration is captured for reproducibility.

```sql
CREATE TABLE eval_runs (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at           TIMESTAMPTZ,
    status                TEXT NOT NULL DEFAULT 'running',  -- running | completed | failed

    task                  TEXT NOT NULL,                     -- chat | summarize | code | retrieval
    dataset_name          TEXT NOT NULL,
    metric_name           TEXT NOT NULL,
    metric_value          DOUBLE PRECISION NOT NULL,

    -- Model
    base_model            TEXT NOT NULL,
    adapter_name          TEXT,                              -- e.g., lora-chat, lora-code
    adapter_version       INTEGER,
    adapter_mlflow_run_id TEXT,                              -- read from MLflow Model Registry
    lora_alias            TEXT,                              -- champion | challenger | none

    -- RAG
    rag_enabled           BOOLEAN NOT NULL DEFAULT false,
    rag_alias             TEXT,                              -- champion | challenger | null
    knowledge_base        TEXT,                              -- arxiv | pytorch_docs | null
    qdrant_collection     TEXT,                              -- resolved collection name
    embedding_model       TEXT,
    chunking_strategy     TEXT,
    chunk_size            INTEGER,
    chunk_overlap         INTEGER,
    retrieval_top_k       INTEGER,
    score_threshold       DOUBLE PRECISION,
    qdrant_snapshot_id    TEXT,                              -- snapshot taken before eval
    dataset_dvc_hash      TEXT,                              -- from .dvc file at eval start
    reranking_strategy    TEXT,                              -- none | cross_encoder | llm

    -- Judge & metrics config
    judge_model           TEXT,
    bert_score_model      TEXT,

    -- Generation params
    temperature           DOUBLE PRECISION,
    max_tokens            INTEGER,

    extra                 JSONB NOT NULL DEFAULT '{}',

    error_message         TEXT
);

CREATE INDEX idx_eval_runs_task ON eval_runs (task);
CREATE INDEX idx_eval_runs_dataset ON eval_runs (dataset_name);
CREATE INDEX idx_eval_runs_adapter ON eval_runs (adapter_name, adapter_version);
CREATE INDEX idx_eval_runs_created ON eval_runs (created_at DESC);
CREATE INDEX idx_eval_runs_base_model ON eval_runs (base_model);
CREATE INDEX idx_eval_runs_rag_alias ON eval_runs (rag_alias);
CREATE INDEX idx_eval_runs_lora_alias ON eval_runs (lora_alias);
CREATE INDEX idx_eval_runs_extra ON eval_runs USING gin (extra);
```

The `extra` JSONB field stores additional information that may expand without migrations.

`rag_alias` and `lora_alias` store role names (`champion`, `challenger`, `none`), not resolved
Qdrant alias names. The resolved collection name is in `qdrant_collection`.

RAG build config fields (`embedding_model`, `chunking_strategy`, `chunk_size`, `chunk_overlap`)
are populated by reading the `_meta` sentinel point from the target collection at eval start.

---

## 4. Auto-promotion comparison logic

Used by the `pytorch_docs_rag_update` DAG (see RAG-IMPROVEMENTS.md, Section 5.1) to decide
whether to auto-promote a newly built collection.

### 4.1 Comparison query

```sql
SELECT metric_value
FROM eval_runs
WHERE task = 'retrieval'
  AND dataset_name = 'beir_scifact'
  AND metric_name = 'nDCG@10'
  AND rag_alias = 'champion'
  AND knowledge_base = :kb_name
  AND status = 'completed'
ORDER BY created_at DESC
LIMIT 1;
```

### 4.2 Decision logic

```
if no previous champion score exists:
    → auto-promote unconditionally (first-ever build)

delta = (new_score - champion_score) / champion_score

if -0.05 <= delta <= 0.20:
    → auto-promote
elif delta < -0.05:
    → hold, log as regression
elif delta > 0.20:
    → hold, log as anomaly (suspiciously large improvement)
```

---

## 5. Eval runner location and CLI

Code location: `experiments/scripts/eval/`

```
experiments/scripts/eval/
    __init__.py
    runner.py              — main eval runner logic
    metrics/
        __init__.py
        automatic.py       — BERTScore, ROUGE-L, Recall@k, nDCG@k
        llm_judge.py       — Gemini API calls for Relevance, Correctness, etc.
        code_exec.py       — sandboxed HumanEval execution
    retrieval_bench.py     — temporary collection builder for retrieval-only evals
```

CLI:

```bash
# Chat eval — single metric (automatic)
python -m experiments.scripts.eval.runner \
    --task chat --dataset hotpotqa --metric rouge_l

# Chat eval — LLM-as-judge metric
python -m experiments.scripts.eval.runner \
    --task chat --dataset hotpotqa --metric relevance

# Chat eval — LLM-as-judge, comparing champion vs challenger RAG + LoRA matrix
python -m experiments.scripts.eval.runner \
    --task chat --dataset hotpotqa --metric correctness \
    --rag-aliases champion,challenger \
    --lora-aliases champion,challenger

# Retrieval-only eval (arxiv KB config vs BEIR-SciFact benchmark)
python -m experiments.scripts.eval.runner \
    --task retrieval --kb arxiv --dataset beir_scifact --metric recall_at_10 \
    --rag-aliases champion,challenger

# Retrieval-only eval (pytorch_docs KB config vs MS MARCO benchmark)
python -m experiments.scripts.eval.runner \
    --task retrieval --kb pytorch_docs --dataset msmarco --metric ndcg_at_10 \
    --rag-aliases champion,challenger

# Code eval — pass@1
python -m experiments.scripts.eval.runner \
    --task code --dataset humaneval --metric pass_at_1

# Summarization — LLM-as-judge metric (no RAG, LoRA comparison)
python -m experiments.scripts.eval.runner \
    --task summarize --dataset arxiv_summarization --metric faithfulness \
    --lora-aliases champion,challenger
```

---

## 6. Implementation stages

### Stage 1: Base LLM (no RAG, no LoRA)

Suites: `eval_chat_hotpotqa_{metric}`, `eval_chat_nq_{metric}`,
`eval_summarization_arxiv_{metric}`, `eval_code_humaneval_{metric}`.
All use `rag_aliases=["none"]`, `lora_aliases=["none"]`. Establishes baseline metrics.

### Stage 2: Base LLM + RAG

Add RAG-enabled generation suites and retrieval-only suites.
Compare `rag_aliases=["champion"]` vs `["none"]` to measure RAG impact.
Groundedness metrics computed when RAG is enabled.

### Stage 3: Base LLM + RAG + LoRA

Add LoRA dimension: `lora_aliases=["champion","none"]` to measure adapter impact.
Full matrix: `(rag_alias, lora_alias)` Cartesian product.

### Stage 4: Agent with orchestrator

TBD — depends on orchestrator architecture.

---

## 7. Implementation Summary (Stages 1–3)

Stages 1–3 of the evaluation workflow have been implemented. Below is an
overview of what was added and how to start using it.

### 7.1 What was implemented

| Component | Files | Description |
|---|---|---|
| **DB model** | `src/shared/db/models.py` | `EvalRun` SQLAlchemy ORM model with all columns from the schema in Section 3. |
| **Migration** | `src/shared/db/eval_runs.sql` | Raw SQL script to create the `eval_runs` table and indexes. Run against the `agent042` PostgreSQL database. |
| **Eval settings** | `src/shared/config.py` | `EvalSettings` class (env prefix `EVAL_`) with judge model, BERTScore model, gateway URL, code-exec settings, etc. |
| **Gateway API** | `src/gateway/services/processing.py`, `src/gateway/services/rag_service.py` | The chat completions response now includes a `rag_context` field when RAG is used. `RAGService` gained `retrieve_documents()` and `format_documents()` methods while preserving full backward compatibility with `retrieve_context()`. |
| **Eval runner** | `experiments/scripts/eval/runner.py` | CLI entry point with `--task`, `--dataset`, `--metric`, `--kb`, `--rag-aliases`, `--lora-aliases`. Each invocation computes exactly **one metric** (one eval-suite). Alias lists form the Cartesian product for cross-configuration comparison. |
| **Automatic metrics** | `experiments/scripts/eval/metrics/automatic.py` | ROUGE-L, BERTScore (via `bert-score`), Recall@k, nDCG@k. |
| **LLM-as-Judge** | `experiments/scripts/eval/metrics/llm_judge.py` | Gemini 2.0 Flash scoring for Relevance, Correctness, Faithfulness, Coverage, Groundedness. Rate-limited to stay under the free-tier 15 RPM cap. |
| **Code execution** | `experiments/scripts/eval/metrics/code_exec.py` | Sandboxed HumanEval execution in ephemeral Docker containers (no network, 512 MB RAM, 30 s timeout). |
| **Retrieval bench** | `experiments/scripts/eval/retrieval_bench.py` | Temporary collection builder that reads `_meta` from production collections and indexes benchmark corpora with the same config. |
| **Airflow DAGs** | `dags/eval_dags.py` | One DAG per eval-suite = per `(task, dataset, metric)`. Stage 1 DAGs (`eval_chat_hotpotqa_{metric}`, `eval_chat_nq_{metric}`, `eval_summarization_arxiv_{metric}`, `eval_code_humaneval_{metric}`) default to `none`/`none` aliases. Stage 2 DAGs (`eval_retrieval_arxiv_beir_scifact_{metric}`, `eval_retrieval_arxiv_beir_nfcorpus_{metric}`, `eval_retrieval_pytorch_msmarco_{metric}`) target retrieval-only evals. Stage 3 is served by the same DAGs with different `params`. |
| **Tests** | `tests/eval/test_eval_workflow.py` | 31 unit tests covering DB model, settings, automatic metrics, LLM judge, code exec, runner config, gateway rag_context, and migration SQL. |

### 7.2 How to get started

**1. Create the database table**

```bash
psql "$AGENT042_DB_URL" -f src/shared/db/eval_runs.sql
```

**2. Set environment variables**

```bash
# Required for LLM-as-Judge
export EVAL_GOOGLE_AI_API_KEY="your-google-ai-studio-key"

# Required for database logging
export EVAL_DB_URL="postgresql://user:pass@localhost:5432/agent042"

# Optional overrides
export EVAL_GATEWAY_URL="http://localhost:9001"
export EVAL_JUDGE_MODEL="gemini-2.0-flash"
export EVAL_SAMPLE_LIMIT="50"    # limit samples for quick testing
```

Inside Docker Compose and the bundled Jupyter container, `EVAL_GATEWAY_URL`
is already injected as `http://gateway:9000`. Export it only when you want to
target a different endpoint.

**3. Run evaluations from the CLI**

```bash
# Stage 1 — base model baselines (one metric per invocation)
python -m experiments.scripts.eval.runner \
    --task chat --dataset hotpotqa --metric rouge_l

python -m experiments.scripts.eval.runner \
    --task chat --dataset hotpotqa --metric relevance

python -m experiments.scripts.eval.runner \
    --task code --dataset humaneval --metric pass_at_1

# Stage 2 — add RAG
python -m experiments.scripts.eval.runner \
    --task chat --dataset hotpotqa --metric bertscore_f1 \
    --rag-aliases champion

# Stage 3 — full RAG + LoRA matrix
python -m experiments.scripts.eval.runner \
    --task chat --dataset hotpotqa --metric correctness \
    --rag-aliases champion,challenger \
    --lora-aliases champion,none
```

**4. Run evaluations from Airflow**

Trigger any eval DAG from the Airflow UI or CLI. Override `params` to
select different alias combinations:

```bash
airflow dags trigger eval_chat_hotpotqa_rouge_l \
    --conf '{"rag_aliases": "champion,challenger", "lora_aliases": "champion,none"}'
```

**5. Run the tests**

```bash
PYTHONPATH=src:. python -m pytest tests/eval/ -v
```
