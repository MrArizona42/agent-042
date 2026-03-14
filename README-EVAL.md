# Evaluation & Benchmarking

## 1. Overview

### 1.1 Tasks, metrics, and datasets

| Task | Dataset | Metrics | Method |
|---|---|---|---|
| Chat (QA) | HotpotQA (validation) | Relevance (1–5), Correctness (1–5) | LLM-as-Judge |
| Chat (QA) | HotpotQA (validation) | BERTScore, ROUGE-L | Automatic |
| Chat (QA) | Natural Questions (validation) | Relevance (1–5), Correctness (1–5) | LLM-as-Judge |
| Chat (QA) | Natural Questions (validation) | BERTScore, ROUGE-L | Automatic |
| Summarization | ArXiv-summarization (validation) | Faithfulness (1–5), Coverage (1–5) | LLM-as-Judge |
| Summarization | ArXiv-summarization (validation) | BERTScore, ROUGE-L | Automatic |
| Code generation | HumanEval (test, 164 examples) | Executable rate, pass@1 | Sandboxed execution |
| RAG + Chat | HotpotQA (validation) | Groundedness (1–5) | LLM-as-Judge |
| RAG + Chat | Natural Questions (validation) | Groundedness (1–5) | LLM-as-Judge |
| RAG + Code | HumanEval (test) | Groundedness (1–5) | LLM-as-Judge |
| Retrieval-only | MS MARCO (validation) | Recall@k, nDCG@k | Automatic |
| Retrieval-only | BEIR-SciFact (corpus) | Recall@k, nDCG@k | Automatic |
| Retrieval-only | BEIR-NFCorpus (corpus) | Recall@k, nDCG@k | Automatic |

### 1.2 Storage granularity

Each eval run produces **one row per metric** in `eval_runs`. For example, `eval_chat_hotpotqa`
with `rag_aliases=["champion","challenger"]` and `lora_aliases=["champion"]` produces:
4 metrics × 2 rag_aliases × 1 lora_alias = **8 rows**.

### 1.3 LLM-as-Judge

Gemini 2.0 Flash via Google AI Studio API. Supports structured JSON output.
Free tier: 15 RPM, 1M tokens/day.

<!-- USER-DECISION: eval subset size
     HotpotQA validation has ~7k examples. At 15 RPM, one LLM-as-judge metric takes ~8 hours.
     Options:
     (a) Sample N examples per eval run (e.g., 200) — fast, but noisier.
     (b) Use paid tier for higher RPM — faster, but costs money.
     (c) Full dataset, accept long runtime — accurate, but blocks DAG slots for hours.
     Pick subset size or rate limit strategy and update this section. -->

### 1.4 Code execution sandbox

HumanEval requires executing untrusted generated Python code.

<!-- USER-DECISION: sandbox mechanism
     Options:
     (a) Docker container with resource limits (CPU/memory/timeout per sample, no network).
     (b) Restricted Python subprocess with resource.setrlimit + timeout.
     (c) Dedicated sandboxing tool (e.g., bubblewrap, firejail).
     Pick a mechanism and specify timeout per sample (e.g., 30s). -->

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
- Temporary benchmark collections are named `eval_{dataset}_{timestamp}` and deleted after the
  eval completes.

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
name registered in vLLM (e.g., `lora-summarization`, `lora-code`).

The adapter name is resolved via MLflow Model Registry:

```
lora_alias="champion" → MLflow: get model version by alias → adapter_name="lora-chat", version=3
→ API request: model="lora-chat"
```

For base model evaluation (no LoRA), `lora_alias` is set to `"none"` and `model` uses the
default base model from settings.

### 2.3 Airflow DAG structure

One Airflow DAG = one eval-suite. Each suite evaluates one `(task, dataset)` pair across a matrix
of configurations.

DAG examples: `eval_chat_hotpotqa`, `eval_retrieval_beir_scifact`, `eval_code_humaneval`,
`eval_summarization_arxiv`.

DAG steps:

```
1. prepare_config       — resolve aliases, read _meta, snapshot collection, build run config
2. generate_predictions — call gateway API (or build temp collection for retrieval-only)
3. compute_metrics      — calculate all metrics for this (task, dataset) pair
4. log_to_db            — write one row per (metric, rag_alias, lora_alias) to eval_runs
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
| `eval_chat_hotpotqa` | `arxiv` | Chat uses arxiv KB |
| `eval_chat_nq` | `arxiv` | Chat uses arxiv KB |
| `eval_code_humaneval` | `pytorch_docs` | Code uses pytorch_docs KB |
| `eval_summarization_arxiv` | N/A | Summarization never uses RAG |
| `eval_retrieval_*` | Reads config from target KB's collection `_meta` | See Section 2.5 |

The `rag_alias` argument (e.g., `champion`, `challenger`) selects which alias role of the
suite's fixed KB to use. The eval runner constructs the Qdrant alias name as
`{kb_name}_{rag_alias}` — consistent with RAG-IMPROVEMENTS.md alias resolution logic.

For suites where RAG is not applicable (Summarization), `rag_aliases` is ignored.

### 2.5 Retrieval-only eval details

For retrieval-only eval suites (`eval_retrieval_msmarco`, `eval_retrieval_beir_scifact`,
`eval_retrieval_beir_nfcorpus`), the `rag_alias` determines **which production collection's
build config to replicate** for the benchmark:

```
1. Resolve {kb_name}_{rag_alias} → get production collection name.
2. Read _meta → extract build_config (chunking_strategy, embedding_model, etc.).
3. Build temporary collection eval_{dataset}_{timestamp} from benchmark corpus
   using the extracted build_config.
4. Run benchmark queries → compute Recall@k, nDCG@k against gold labels.
5. Log results to eval_runs (with rag_alias recorded).
6. Delete temporary collection.
```

<!-- USER-DECISION: retrieval eval KB mapping
     Current assumption: retrieval evals always replicate arxiv KB config (since arxiv
     is the primary chat KB and retrieval quality matters most there).
     If you want retrieval evals to also test pytorch_docs config, add a kb parameter
     to retrieval eval suites or create separate suites per KB. -->

---

## 3. Database schema

One row per `(task, dataset, metric, rag_alias, lora_alias)` combination. The full system
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
# Chat eval with default aliases
python -m experiments.scripts.eval.runner \
    --task chat --dataset hotpotqa

# Chat eval comparing champion vs challenger (RAG + LoRA matrix)
python -m experiments.scripts.eval.runner \
    --task chat --dataset hotpotqa \
    --rag-aliases champion,challenger \
    --lora-aliases champion,challenger

# Retrieval-only eval
python -m experiments.scripts.eval.runner \
    --task retrieval --dataset beir_scifact \
    --rag-aliases champion,challenger

# Summarization (no RAG, LoRA comparison)
python -m experiments.scripts.eval.runner \
    --task summarize --dataset arxiv_summarization \
    --lora-aliases champion,challenger
```

---

## 6. Implementation stages

### Stage 1: Base LLM (no RAG, no LoRA)

Suites: `eval_chat_hotpotqa`, `eval_chat_nq`, `eval_summarization_arxiv`, `eval_code_humaneval`.
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
