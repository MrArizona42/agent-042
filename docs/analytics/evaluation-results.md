# Evaluation Results

Evaluation results live in Postgres. They are the source of truth for offline
quality checks, while ClickHouse is the source for production inference event
analytics.

## Tables

`eval_runs` stores aggregate metric rows. One evaluation execution can create
several rows: for example one row per metric and per `(rag_alias, lora_alias)`
combination.

`eval_samples` stores per-sample details linked to an `eval_runs.id`. The same
sample can appear once for each aggregate metric row because the sample details
are attached to the metric result that was computed from them.

## Important `eval_runs` Fields

| Field | Meaning |
| --- | --- |
| `id` | Aggregate metric row id. Join key for `eval_samples.eval_run_id`. |
| `created_at`, `finished_at` | Eval timing. Use `created_at` for trend charts. |
| `status` | `running`, `completed`, or `failed`. |
| `task` | Eval task: `chat`, `summarize`, `code`, or `retrieval`. |
| `dataset_name` | Dataset key used by the eval runner. |
| `metric_name`, `metric_value` | Metric and aggregate score. |
| `base_model` | Gateway default model at eval time. |
| `adapter_name`, `adapter_version`, `lora_alias` | LoRA adapter metadata, when used. |
| `rag_enabled`, `rag_alias`, `knowledge_base` | RAG configuration used by the run. |
| `qdrant_alias`, `qdrant_collection`, `rag_manifest_id` | Resolved RAG collection metadata. |
| `embedding_model`, `chunking_strategy`, `retrieval_top_k` | Retrieval configuration metadata when available. |
| `judge_backend`, `judge_model` | LLM-as-judge backend used for judge metrics. |
| `temperature`, `max_tokens` | Generation settings used during prediction fetch. |
| `eval_verdict` | Threshold result: `pass`, `warn`, `fail`, or `unscored`. |
| `extra` | JSONB bag for task-specific metadata and RAG observability details. |
| `error_message` | Failure details for failed runs. |

## Important `eval_samples` Fields

| Field | Meaning |
| --- | --- |
| `eval_run_id` | Parent aggregate metric row. |
| `sample_idx` | Ordinal position in the loaded dataset. |
| `sample_id` | Dataset-native id, when available. |
| `input` | Question, prompt, or query. |
| `output` | Model output, generated code, or `null` for retrieval-only evals. |
| `reference` | Gold answer, expected output, or test code. |
| `detail` | JSONB task-specific details. |

Common `detail` shapes:

- generation/chat: may include `rag_context`;
- code: includes `passed`, `exit_code`, and `stderr`;
- retrieval: includes `retrieved_ids` and `relevance`.
- RAG benchmark: includes retrieved chunk/document provenance, expected qrels,
  reference answers, evidence refs, generation facts, prompt identity, and
  timing diagnostics.

## RAG Benchmark Persistence

RAG benchmark inputs are prepared as source-instance artifacts, not as result
reports:

```text
assets/rag_data/source_instances/<benchmark_source_instance_id>/benchmark/
  cases.jsonl
  labels.jsonl
  metadata.json
```

Benchmark results live only in Postgres:

- aggregate metrics go to `eval_runs`;
- per-case observations go to `eval_samples.detail`;
- runs must record the benchmark source instance id, KB id, explicit alias,
  resolved Qdrant alias/collection, manifest id, adapter id/version, artifact
  digests, and prompt identity when generation is involved.

RAG benchmark suite names:

```text
retrieval_quality
context_quality
generation_quality
```

Retrieval labels are normalized as `qrels[]` with `entity_type` set to
`document` or `chunk`. Flat relevant id lists are derived from qrels rather than
stored as a second source of truth.

## Metric Families

Automatic generation metrics:

- `rouge_l`;
- `bertscore_precision`;
- `bertscore_recall`;
- `bertscore_f1`.

Retrieval metrics:

- `recall_at_<k>`;
- `ndcg_at_<k>`;
- `mrr_at_<k>`.

Code metrics:

- `pass_at_1`.

LLM-as-judge metrics:

- `relevance`;
- `correctness`;
- `faithfulness`;
- `coverage`;
- `groundedness`.

Judge metrics use a 1-5 score scale. Threshold verdicts are only meaningful
when the eval context configured thresholds for the metric.

## Common Queries

Latest completed runs:

```sql
select
    id,
    created_at,
    task,
    dataset_name,
    metric_name,
    metric_value,
    eval_verdict,
    knowledge_base,
    rag_alias,
    lora_alias,
    qdrant_collection
from eval_runs
where status = 'completed'
order by created_at desc
limit 50;
```

Metric trend by dataset and RAG alias:

```sql
select
    date_trunc('day', created_at) as day,
    task,
    dataset_name,
    metric_name,
    coalesce(rag_alias, 'none') as rag_alias,
    avg(metric_value) as avg_metric
from eval_runs
where status = 'completed'
group by day, task, dataset_name, metric_name, coalesce(rag_alias, 'none')
order by day desc, task, dataset_name, metric_name, rag_alias;
```

Champion/challenger comparison:

```sql
select
    task,
    dataset_name,
    metric_name,
    knowledge_base,
    coalesce(rag_alias, 'none') as rag_alias,
    coalesce(lora_alias, 'none') as lora_alias,
    avg(metric_value) as avg_metric,
    count(*) as rows
from eval_runs
where status = 'completed'
group by
    task,
    dataset_name,
    metric_name,
    knowledge_base,
    coalesce(rag_alias, 'none'),
    coalesce(lora_alias, 'none')
order by task, dataset_name, metric_name, knowledge_base, rag_alias, lora_alias;
```

Failed or warning aggregate rows:

```sql
select
    id,
    created_at,
    task,
    dataset_name,
    metric_name,
    metric_value,
    eval_verdict,
    error_message
from eval_runs
where status <> 'completed'
   or eval_verdict in ('warn', 'fail')
order by created_at desc
limit 100;
```

Sample details for one run:

```sql
select
    sample_idx,
    sample_id,
    input,
    output,
    reference,
    detail
from eval_samples
where eval_run_id = '<EVAL_RUN_ID>'
order by sample_idx;
```

Retrieval no-hit samples:

```sql
select
    r.id as eval_run_id,
    r.created_at,
    r.dataset_name,
    r.knowledge_base,
    r.rag_alias,
    s.sample_idx,
    s.sample_id,
    s.input,
    s.detail -> 'retrieved_ids' as retrieved_ids,
    s.detail -> 'relevance' as relevance
from eval_runs r
join eval_samples s on s.eval_run_id = r.id
where r.task = 'retrieval'
  and r.status = 'completed'
  and jsonb_array_length(coalesce(s.detail -> 'retrieved_ids', '[]'::jsonb)) = 0
order by r.created_at desc, s.sample_idx
limit 100;
```

Code eval failures:

```sql
select
    r.id as eval_run_id,
    r.created_at,
    r.dataset_name,
    r.lora_alias,
    s.sample_id,
    s.input,
    s.output,
    s.detail ->> 'stderr' as stderr
from eval_runs r
join eval_samples s on s.eval_run_id = r.id
where r.task = 'code'
  and s.detail ->> 'passed' = 'false'
order by r.created_at desc
limit 50;
```

RAG observability metadata saved in `extra`:

```sql
select
    id,
    created_at,
    task,
    dataset_name,
    metric_name,
    extra -> 'rag' as rag_observability
from eval_runs
where extra ? 'rag'
order by created_at desc
limit 50;
```

## Current Gaps For Phase 2

- Citation metadata is not yet stored per sample.
- Retrieval eval has document ids and relevance labels, but not final
  user-facing citation correctness.
- LLM-as-judge prompts exist in code but should be versioned explicitly before
  judge metrics become promotion gates.
- Production feedback is not yet joined to eval samples.
- Request-level production events and offline eval rows do not yet share a
  first-class experiment or variant id.

## Related Docs

- [Observability, Evaluation, And Analytics Workflow](observability-evaluation-workflow.md)
- [ClickHouse Analytics](clickhouse-analytics.md)
- [Improvement Plan](../planning/improvements.md)
- [Failure Analysis Notebook](../../experiments/eval/failure_analysis.ipynb)
