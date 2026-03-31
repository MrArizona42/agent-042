# Plan: CLI Migration, RAG Modularity & Eval Pipeline Refactor

Status: **Done**
Branch: `refactor/lora-swap-optimization` (continuation)

> **Note:** The CLI scripts `scripts/manage_registry.py` and `scripts/manage_rag.py`
> referenced in §1.2 and §1.3 have been deleted. Registry and RAG operations are now
> performed via operation notebooks (`experiments/training/lora_ops.ipynb`,
> `experiments/rag/rag_ops.ipynb`) or the underlying Python APIs.

---

## 1. Migrate CLI Scripts to Python Fire

Replace argparse boilerplate with `python-fire` across all CLI scripts
for a unified CLI convention.  Keep Hydra for `train_hydra.py`.

### 1.1 Add `fire` dependency

- Add `fire` to `[project.optional-dependencies]` in `pyproject.toml`
  (under the `cli` extra or the main deps group).

### 1.2 Migrate `scripts/manage_registry.py` (8 subcommands)

Current state: ~334 lines, argparse with `register`, `list`, `versions`,
`promote`, `demote`, `download`, `production`, `sync`.

Steps:
- [x] Replace `main()` argparse dispatch with a `ManageRegistry` Fire class.
- [x] Each `cmd_*` function becomes a method on the class.
- [x] Positional args in argparse become method positional params; optional
      flags become keyword params with defaults.
- [x] Remove the `dispatch = {…}` dict and `parser.add_subparsers` block.
- [x] Verify: `python scripts/manage_registry.py list`,
      `python scripts/manage_registry.py sync --vllm-url …`.

### 1.3 Migrate `scripts/manage_rag.py` (3 subcommands)

Current state: ~140 lines, argparse with `list`, `inspect`, `promote`.

Steps:
- [x] Replace `main()` with a `ManageRag` Fire class.
- [x] `_cmd_list` → `list()`, `_cmd_inspect` → `inspect(kb, alias)`,
      `_cmd_promote` → `promote(kb, from_alias, to_alias)`.
- [x] Global `--qdrant-host` / `--qdrant-port` become `__init__` params
      with defaults from `get_settings()`.
- [x] Remove argparse.

### 1.4 Migrate `src/shared/model_registry.py` `_cli()` (2 subcommands)

Current state: ~70 lines, argparse with `sync`, `list`.

Steps:
- [x] Replace internal `_cli()` with a `RegistryCli` Fire class.
- [x] `sync` and `list` become methods.
- [x] Verify: `python -m shared.model_registry sync --vllm-url …`.

### 1.5 Migrate `experiments/scripts/eval/runner.py` (flat flags)

Current state: ~1770 lines, flat argparse with `--task`, `--dataset`,
`--metric`, `--kb`, `--rag-aliases`, `--lora-aliases`.

Steps:
- [x] Replace `main()` argparse with `fire.Fire(run_eval_cli)` where
      `run_eval_cli` is a thin wrapper whose params mirror the current flags.
- [x] Add validation for `task` choices (`chat`, `summarize`, `code`,
      `retrieval`) at the top of the function — replaces argparse `choices=`.
- [x] Comma-separated `--rag-aliases` / `--lora-aliases` become `str` params
      split inside the function (Fire passes them as strings).

### 1.6 Migrate `experiments/scripts/rag_data/build_*.py` to Fire (after §2 split)

After the split (§2), each build script has flat flags.  Migrate both.

Steps:
- [x] `build_arxiv_index.py`: replace argparse with `fire.Fire(build_arxiv)`.
- [x] `build_pytorch_docs_index.py`: replace argparse with
      `fire.Fire(build_pytorch_docs)`.
- [x] Validate `--chunking-strategy` choices in-function.

### 1.7 Scripts left on their current framework (no change)

| Script | Reason |
|--------|--------|
| `experiments/scripts/train_hydra.py` | Uses Hydra — not applicable. |

---

## 2. Split `build_vector_index.py` into Per-Strategy Scripts

Current state: 571 lines, two fundamentally different build strategies
(`build_chat_index` incremental/upsert, `build_code_index` atomic-replace)
mashed into one file with `--task chat|code|both` dispatch and 12 argparse
flags.  Shared chunking/embedding/batching loops are duplicated.

### 2.1 Create `build_arxiv_index.py` (incremental strategy)

- [x] Move `build_chat_index()` + its helpers into
      `experiments/scripts/rag_data/build_arxiv_index.py`.
- [x] CLI via `fire.Fire(build_arxiv)` — function params replace argparse
      flags: `kb`, `alias`, `arxiv_file`, `chunking_strategy`, `chunk_size`,
      `chunk_overlap`, `qdrant_host`, `qdrant_port`, `embeddings_url`.
- [x] Verify: runs standalone with same behaviour as before.

### 2.2 Create `build_pytorch_docs_index.py` (atomic-replace strategy)

- [x] Move `build_code_index()` + its helpers into
      `experiments/scripts/rag_data/build_pytorch_docs_index.py`.
- [x] CLI via `fire.Fire(build_pytorch_docs)` — function params replace
      argparse flags: `kb`, `alias`, `pytorch_docs_file`,
      `chunking_strategy`, `chunk_size`, `chunk_overlap`, `qdrant_host`,
      `qdrant_port`, `embeddings_url`.

### 2.3 Promote shared helpers to `rag` modules (if needed)

- [x] If both scripts share a non-trivial batched embed+upsert loop,
      extract a `rag.build_utils.batched_embed_and_upsert()` helper.
- [x] Otherwise, keep duplication minimal — the `rag.chunking`,
      `rag.embeddings`, `rag.vector_store` modules already provide the
      building blocks.

### 2.4 Delete the monolithic script

- [x] Remove `build_vector_index.py`.

### 2.5 Update Airflow DAGs

- [x] Update `dags/arxiv_rag_update.py` to call `build_arxiv_index.py`.
- [x] Update `dags/pytorch_docs_rag_update.py` to call
      `build_pytorch_docs_index.py`.

---

## 3. Eval Pipeline: Drop `compute_automatic_metrics` Indirection

Current state: `compute_automatic_metrics()` in `automatic.py` is a thin
dispatcher — the runner already passes `metric=metric`, so the function
just does `if metric == "rouge_l"` / `elif metric.startswith("bertscore")`
and calls the individual function.  Unnecessary layer.

### 3.1 Call metric functions directly from the runner

- [x] In `_evaluate_generation()`: replace the
      `compute_automatic_metrics(…, metric=metric)` call with direct calls
      to `compute_rouge_l()` or `compute_bertscore()` depending on
      the `metric` value.
- [x] In `_compute_generation_metric()` (two-phase path): same change.

### 3.2 Remove the aggregator

- [x] Delete `compute_automatic_metrics()` from `automatic.py`.
- [x] Public API of `automatic.py` becomes: `compute_rouge_l`,
      `compute_bertscore`, `compute_recall_at_k`, `compute_ndcg_at_k`.

---

## 4. Eval Pipeline: Unify One-Phase / Two-Phase Paths

Current state: `runner.py` has two parallel eval flows:

- **One-phase** (`run_eval` → `_evaluate_generation` / `_evaluate_code` /
  `_evaluate_retrieval`): predict + compute metrics in one call.
- **Two-phase** (`fetch_predictions` + `calculate_metrics` →
  `_compute_generation_metric` / `_compute_code_metric` /
  `_compute_retrieval_metric`): separate predict and compute for Celery.

Both paths duplicate ~400 lines of nearly identical gateway-call,
metric-dispatch, and DB-logging logic.

### 4.1 Make one-phase call two-phase internally

- [x] Rewrite `run_eval()` to:
      1. Call `fetch_predictions(…)` to get predictions.
      2. Call `calculate_metrics(metric=metric, prediction_data=…)` to
         compute the metric and log to DB.
- [x] Ensure existing CLI behaviour is unchanged.

### 4.2 Delete one-phase-only helpers

- [x] Remove `_evaluate_generation()`, `_evaluate_code()`,
      `_evaluate_retrieval()` — their logic now lives in the two-phase
      functions (`_fetch_*` + `_compute_*`).

### 4.3 Estimated line reduction

Removing the one-phase duplicates should cut runner.py by ~400 lines
(from ~1770 to ~1350).

---

## 5. Eval Pipeline: Extract Dataset Loader

Current state: `_load_dataset_samples()` is ~130 lines of complex
format-specific parsing (BEIR qrels, HumanEval, NQ annotations, MSMARCO
passages) embedded in `runner.py`.  It has no dependency on eval logic.

### 5.1 Create `experiments/scripts/eval/datasets.py`

- [x] Move `_load_dataset_samples()`, `_DATASET_LOCAL`, `DATASETS_DIR`,
      and `_PROJECT_ROOT` into a new `datasets.py` module.
- [x] Export `load_dataset_samples()` (drop the underscore — now public).
- [x] Import it in `runner.py`.

---

## 6. Eval Pipeline: Minor Optimizations

### 6.1 Retrieval: return both metrics in one pass

Current state: `_evaluate_retrieval` / `_compute_retrieval_metric`
computes both `recall_at_10` and `ndcg_at_10` on every call but only
returns the one matching `metric`, discarding the other.  Building the
temp collection is the expensive step.

- [x] When both `recall_at_10` and `ndcg_at_10` are requested for the
      same suite, compute once and return both rows.
- [x] Alternatively, in the two-phase flow `fetch_predictions` already
      stores `query_results`; `_compute_retrieval_metric` is cheap — this
      is a low priority since the expensive collection build is only done
      once per `fetch_predictions` call.  **Skipped — already cheap.**

### 6.2 Gateway call loop extraction (optional)

`_evaluate_generation` and `_evaluate_code` share the same loop:
iterate samples → `_call_gateway()` → extract content → count failures.
A small `_collect_predictions(samples, gateway_url, …)` helper could
DRY this up.

- [x] Evaluate whether this is worth the abstraction cost. Skip if the
      two loops diverge enough (code eval has prompt-specific message
      formatting).  **Skipped — loops diverge enough.**

---

## Implementation Order

| Step | Section | Est. Complexity |
|------|---------|-----------------|
| 1 | §1.1 — Add `fire` dependency | trivial |
| 2 | §1.2 — Migrate `manage_registry.py` | medium |
| 3 | §1.3 — Migrate `manage_rag.py` | small |
| 4 | §1.4 — Migrate `model_registry.py _cli()` | small |
| 5 | §2.1–2.2 — Split build scripts (with Fire CLI) | medium |
| 6 | §2.3 — Extract shared helpers if needed | small |
| 7 | §2.4–2.5 — Delete old script, update DAGs | small |
| 8 | §1.5 — Migrate `runner.py` to Fire | small |
| 9 | §3.1–3.2 — Drop `compute_automatic_metrics` | small |
| 10 | §5.1 — Extract dataset loader | small |
| 11 | §4.1–4.2 — Unify one-phase/two-phase eval | medium |
| 12 | §6.1–6.2 — Minor retrieval/gateway optimizations | small |

Steps 1–4 (Fire migration of management CLIs) are independent of steps 5–7 (RAG split).
Step 8 (runner.py Fire) can be done anytime after step 1.
Steps 9–12 (eval refactor) depend on each other and should be done in order.
