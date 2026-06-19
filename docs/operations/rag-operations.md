# RAG Operations

This is the operator-facing workflow for RAG builds on the server. The main
entrypoint is the `rag-ops` Compose service, wrapped by `scripts/rag_ops.sh`.
Airflow uses the same CLI through the `rag_lifecycle` DAG.

For the conceptual model and runtime architecture, see section 5 of
`docs/architecture/system-design.md`. This document focuses on operator
commands and server workflow.

## Operator Contract

The supported production flow is:

```text
validate catalog and manifests
  -> build corpus source instances
  -> materialize a physical collection with an explicit alias profile
  -> point the challenger alias at that collection
  -> prepare and run attached benchmarks against challenger
  -> inspect Postgres results and gateway behavior
  -> promote the same collection to champion
```

LlamaIndex owns document, node, index, retrieval, query-engine, and evaluator
mechanics. Project code owns catalog identity, source adapters, alias policy,
collection attestations, benchmark labels, orchestration, and DB persistence.
There is no supported legacy vector-store, retriever, `collection_meta`
sentinel, `[[sources]]`, or KB-local source-id path.

## Prerequisites

Before running a lifecycle command, verify:

- `catalog.toml` and `runtime.toml` are from the deployed release;
- `assets/rag_data` is writable by the `rag-ops` container;
- Qdrant, embeddings, and reranker services are healthy;
- vLLM is healthy when running generation or judge benchmarks;
- `GATEWAY_AGENT042_DB_URL` points to the eval Postgres database;
- the configured embedding model and dimension match collections that will be
  queried;
- the configured judge model appears in the selected OpenAI-compatible
  backend, and external judges declare `eval.judge.context_window`.

For a local environment, install the relevant dependency surfaces:

```bash
uv sync --extra rag --extra gateway --extra airflow-worker
```

## Naming

- KB id: logical knowledge base id from `catalog.toml`, for example
  `ml_papers_core` or `pytorch_reference`.
- Source adapter: catalog-declared behavior for loading a source instance.
  Source adapters live in `[[source_adapters]]`; benchmark-capable adapters
  live in `[[benchmark_adapters]]`.
- Source instance id: globally meaningful source id, for example
  `ml_papers_core.papers` or `pytorch_reference.docs`.
- Source role: `role = "corpus"` participates in normal KB builds;
  `role = "benchmark"` is prepared with `prepare-benchmark` and is excluded
  from normal materialization.
- Source manifest: curated input list or adapter-specific config at the
  conventional path
  `assets/rag_data/source_instances/<source_instance_id>/manifest.toml`.
- Alias config: per-KB retrieval profile in the catalog, for example
  `champion` or `challenger`; it controls `top_k`, threshold, strategy, and
  reranker settings.
- Physical collection: Qdrant collection built from a bundle, named
  `rag__<kb_id>__<timestamp>`. It intentionally does not include alias names.
- Qdrant alias: runtime pointer named `rag__<kb_id>__<alias>`.
- Artifact manifest: full build provenance JSON under
  `assets/rag_data/knowledge_bases/<kb>/manifests/`.
- Qdrant attestation: compact collection metadata used so runtime can validate
  alias targets. Builds store it at `.result.config.metadata.attestation`.
  Collections without this metadata must be rebuilt before use.
- Benchmark artifacts: normalized `corpus.jsonl`, `cases.jsonl`,
  `labels.jsonl`, and `metadata.json` under
  `assets/rag_data/source_instances/<benchmark_source_instance_id>/benchmark/`.

The catalog no longer supports legacy `[[sources]]` entries or KB-local source
selectors. Operator commands use global `--source-instance <id>` values.

## Artifact Layout

```text
assets/rag_data/
  source_instances/<source_instance_id>/
    manifest.toml
    raw/
    extracted/
    chunks/
    benchmark/
      corpus.jsonl
      cases.jsonl
      labels.jsonl
      metadata.json

  knowledge_bases/<kb_id>/
    manifests/<collection_name>.json
    metadata/build_runs/<build_run_id>.json
```

`raw/`, `extracted/`, and `chunks/` are resumable caches containing native
LlamaIndex data. They are not alternate project document/chunk contracts.
Benchmark results do not live here; Postgres is their only result store.

## Server CLI

Run commands from the deployment root on the server:

Choose one audit id and one candidate name for the complete manual run:

```bash
export BUILD_RUN_ID="manual-pytorch-$(date -u +%Y%m%d-%H%M%S)"
export COLLECTION="rag__pytorch_reference__$(date -u +%Y%m%d_%H%M%S)"
```

Validate configuration, source-instance selection, manifests, and adapter
factories without changing external state:

```bash
bash current/scripts/rag_ops.sh python -m rag.sources.cli plan \
  --catalog catalog.toml \
  --kb pytorch_reference \
  --source-instance pytorch_reference.docs \
  --rag-data-root assets/rag_data
```

Always run `plan` before a full rebuild or after changing `catalog.toml`.

Build one or more corpus source instances:

```bash
bash current/scripts/rag_ops.sh python -m rag.sources.cli build-source \
  --catalog catalog.toml \
  --source-instance pytorch_reference.docs \
  --rag-data-root assets/rag_data \
  --build-run-id "$BUILD_RUN_ID" \
  --persist-build-run
```

Materialize a candidate from existing native node artifacts. Give production
runs an explicit unique collection name so subsequent inspection and promotion
cannot accidentally target a different build:

```bash
bash current/scripts/rag_ops.sh python -m rag.sources.cli materialize \
  --catalog catalog.toml \
  --kb pytorch_reference \
  --source-instance pytorch_reference.docs \
  --alias-config challenger \
  --collection "$COLLECTION" \
  --rag-data-root assets/rag_data \
  --build-run-id "$BUILD_RUN_ID" \
  --persist-build-run
```

Prepare a benchmark source instance:

```bash
bash current/scripts/rag_ops.sh python -m rag.sources.cli prepare-benchmark \
  --catalog catalog.toml \
  --source-instance pytorch_reference.qa_benchmark \
  --rag-data-root assets/rag_data
```

Promote a verified collection behind an alias:

```bash
bash current/scripts/rag_ops.sh python -m rag.sources.cli promote-alias \
  --catalog catalog.toml \
  --kb pytorch_reference \
  --alias challenger \
  --collection "$COLLECTION" \
  --rag-data-root assets/rag_data \
  --build-run-id "$BUILD_RUN_ID" \
  --persist-build-run
```

Inspect persisted lifecycle state:

```bash
bash current/scripts/rag_ops.sh python -m rag.sources.cli status \
  --kb pytorch_reference \
  --rag-data-root assets/rag_data

bash current/scripts/rag_ops.sh python -m rag.sources.cli show-build-run \
  --kb pytorch_reference \
  --build-run-id "$BUILD_RUN_ID" \
  --rag-data-root assets/rag_data
```

Use `--document-id` and `--limit` for smoke builds. Use force flags only when
deliberately invalidating a cache layer: `--force-fetch`, `--force-extract`,
`--force-chunk`, and `--force-recreate` progress from least to most expensive.

## Build Rules

- `build-source` fetches source documents, extracts native LlamaIndex
  `Document` objects, and parses them into native `TextNode` artifacts. Cache
  artifacts are immutable unless force flags are passed.
- `build-source` takes one or more global `--source-instance` values and derives
  the KB from those source instances. All selected source instances must belong
  to the same KB.
- `build-source --source-instance <id>` rejects `role = "benchmark"` targets.
  Use `prepare-benchmark` for benchmark source instances.
- `materialize --alias-config <alias>` uses that alias profile as build input.
  It indexes native nodes through LlamaIndex and does not assign or move a
  Qdrant alias.
- `promote-alias --alias <alias>` points `rag__<kb>__<alias>` at an attested
  physical collection.
- Dense alias configs can query dense or hybrid collections.
- Hybrid alias configs require hybrid collections.
- Challenger collections should normally be built and inspected before
  champion promotion.
- LlamaIndex-built collections are serving-compatible after inspection;
  runtime reopens them through the alias and validates collection metadata
  before retrieval.
- Benchmark execution must receive an explicit alias. A benchmark source
  instance is attached to exactly one KB through `source_instance.knowledge_base`;
  the alias supplies the KB runtime/build profile for that run.

## Candidate Promotion Workflow

1. Run `plan`.
2. Run `build-source` for every changed corpus source instance.
3. Run `materialize --alias-config challenger --collection <candidate>`.
4. Verify the artifact manifest and Qdrant attestation agree.
5. Point `challenger` at the candidate.
6. Run retrieval, context, and generation benchmarks declared for that KB.
7. Smoke-test gateway retrieval against `challenger`.
8. Point `champion` at the exact same physical collection only after the
   candidate passes.

Alias promotion is intentionally separate from materialization. Building a
collection never changes serving traffic by itself.

## Benchmark Workflow

A benchmark source instance is `role = "benchmark"`, belongs to exactly one
KB, and declares one or more suites in `benchmark.suites`:

```text
retrieval_quality
context_quality
generation_quality
```

Prepare normalized corpus/case/label artifacts whenever its manifest or labels
change:

```bash
bash current/scripts/rag_ops.sh python -m rag.sources.cli prepare-benchmark \
  --catalog catalog.toml \
  --source-instance pytorch_reference.qa_benchmark \
  --rag-data-root assets/rag_data
```

Run it against an explicit live alias:

```bash
bash current/scripts/rag_ops.sh python -m rag.evaluation.cli \
  --catalog catalog.toml \
  --source-instance pytorch_reference.qa_benchmark \
  --alias challenger \
  --rag-data-root assets/rag_data
```

The selected alias supplies current chunking, embedding, retrieval, threshold,
and reranking parameters. If `corpus.jsonl` is populated, the runner builds a
temporary benchmark collection with that profile and deletes it in `finally`.
If it is empty, cases query the attached live KB collection directly.

Retrieval quality uses LlamaIndex binary hit rate, MRR, precision, recall, AP,
and NDCG, plus project graded NDCG for document/chunk qrels. Context quality
uses LlamaIndex context relevancy. Generation quality uses answer relevancy,
faithfulness, and correctness when reference answers exist.

### Judge Configuration

Generation and judge clients are separate. The generation client uses the
runtime vLLM model. The judge uses `[eval.judge]` from `runtime.toml`:

```toml
[eval.judge]
backend = "local_vllm" # allowed: "local_vllm", "openai_compatible"
model = "/models/Qwen/Qwen3-0.6B"
base_url = ""
timeout = 60.0
request_delay_seconds = 0.0
# context_window = 128000 # required for backend = "openai_compatible"
```

Both clients use LlamaIndex `OpenAILike`, so self-hosted model names are valid.
Before judge runs, confirm the configured model is listed by the selected
backend's `/v1/models` endpoint.

### Result Verification

Every aggregate metric is one `eval_runs` row; every per-case observation is
an `eval_samples` row. A successful run must record the explicit alias,
physical collection, manifest id, benchmark artifact digests, prompt identity
for generation, and actual judge backend/model for judged metrics.

Example SQL:

```sql
select
    r.created_at,
    r.dataset_name,
    r.knowledge_base,
    r.rag_alias,
    r.qdrant_collection,
    r.metric_name,
    r.metric_value,
    r.judge_backend,
    r.judge_model,
    count(s.id) as sample_count
from eval_runs r
left join eval_samples s on s.eval_run_id = r.id
where r.dataset_name = '<benchmark-source-instance-id>'
group by r.id
order by r.created_at desc, r.metric_name;
```

Treat missing sample rows, missing artifact identity, unexpected judge
identity, or leftover temporary collections as failed acceptance checks even
when aggregate scores were written.

## Airflow

Airflow schedules corpus lifecycle work only. RAG benchmark execution is
currently an explicit operator action through `rag.evaluation.cli`.

Use `rag_lifecycle` for the same lifecycle:

- `kb`: KB id, for example `ml_papers_core`.
- `source_instance`: source instance id, for example `ml_papers_core.papers`.
  Leave empty to build all corpus source instances for the selected KB.
- `alias_config`: build profile, usually `challenger` for test builds.
- `promote_alias`: optional runtime alias to repoint after materialization.
  Leave empty for build-only runs.
- `document_ids` and `limit`: scoped smoke builds.
- `force_fetch`, `force_extract`, `force_chunk`, `force_recreate`: explicit
  cache/collection invalidation controls.
- `sync_dvc`: when true, DVC-sync generated artifacts before promotion.
- `dvc_artifacts`: optional comma-separated artifact directories to sync. The
  DAG resolves source-instance artifact names such as `extracted`, `chunks`,
  and `benchmark` under `source_instances/<source_instance_id>/`; KB-scoped
  names such as `manifests` and `metadata` resolve under
  `knowledge_bases/<kb>/`.
- `dvc_base_branch`, `dvc_bot_branch`: Git branch controls for the temp-clone
  DVC sync PR.
- `build_run_id`: optional stable audit id. When omitted, Airflow derives one
  from the DAG run id and passes it through build, materialize, and promotion.
- `dry_run`: create and optionally persist the lifecycle request/plan without
  fetching, extracting, chunking, materializing into Qdrant, or promoting an
  alias.

Each persisted build run is written under
`<rag_data_root>/knowledge_bases/<kb>/metadata/build_runs/<build_run_id>.json`. The record
captures the catalog digest, selected source manifest digests, source adapter
versions, build profile digest, per-stage results, final collection name, and
promotion status. Use it as the first restore/debug handle before touching DVC
or Qdrant aliases.

DVC policy:

- Source instance `manifest.toml` files stay in Git because they are curated
  operator input.
- Generated source-instance artifacts, benchmark normalized artifacts, KB
  manifests, and KB metadata directories are DVC candidates.
- Raw cache (`raw/`) is server-local by default. This avoids DVC-tracking raw
  PDFs unless fully offline rebuilds become a requirement.
- When `sync_dvc=true`, `rag_lifecycle` runs DVC sync after materialization and
  before alias promotion. A failed DVC sync prevents promotion by task ordering.

## Inspection

Use the Qdrant API/dashboard and `rag.sources.cli` for direct observability:

- list collections and aliases;
- compare expected catalog aliases with live Qdrant aliases;
- inspect collection attestation metadata and sample points;
- identify old physical collections not behind any alias;
- create snapshots through Qdrant's snapshot API;
- remove stale collections through the guarded collection-cleanup workflow.

Inspect one collection's attestation from the server:

```bash
curl -s "$QDRANT_URL/collections/$COLLECTION" \
  | jq '.result.config.metadata.attestation'
```

The response must contain `manifest_id`, `kb_id`, `collection_name`,
`embedding_model`, `retrieval_capability`, and `chunk_count`. Compare it with
`assets/rag_data/knowledge_bases/<kb>/manifests/<collection>.json`. A missing
attestation is not repaired in place; rebuild the collection.

## Runtime Observability

`rag.runtime.RagRuntime` returns result-level observability alongside native
`NodeWithScore` results:

- `provenance`: one row per resolved KB/alias with Qdrant alias, physical
  collection, manifest id, retrieval strategy/capability, hit count, no-hit
  flag, score summary, and source timings.
- `timings_ms`: total runtime retrieval latency.
- `diagnostics`: requested/resolved/skipped source counts, total hit count, and
  no-hit flag.

The gateway logs these diagnostics after retrieval. This is the first feedback
surface for Grafana/log-based runtime panels; it does not change alias
promotion behavior.

## Rollback

Rollback is another alias promotion. Point the alias back to a previous
attested physical collection:

```bash
bash current/scripts/rag_ops.sh python -m rag.sources.cli promote-alias \
  --kb pytorch_reference \
  --alias champion \
  --collection rag__pytorch_reference__20260604_180000
```

Before rollback, inspect the target collection attestation and make sure its
retrieval capability is compatible with the alias config.

## Failure Recovery

| Failure | Operator action |
| --- | --- |
| Catalog, manifest, or adapter validation fails | Fix configuration; rerun `plan`. Do not use force flags. |
| Fetch/extraction/chunking fails | Inspect the source build summary; retry only the failed cache layer. |
| Materialization fails | Leave aliases unchanged; remove the unattached partial collection after inspection. |
| Attestation or manifest mismatch | Do not promote. Rebuild from a clean physical collection name. |
| Challenger benchmark fails | Keep champion unchanged; preserve DB observations for comparison. |
| Judge model is unavailable | Fix `[eval.judge]` or model deployment; do not interpret missing judged metrics as passes. |
| Benchmark process is interrupted | Confirm its temporary `eval__*` collection was removed before rerunning. |
| Champion regresses after promotion | Promote the previously attested collection back to champion. |

## Legacy Collection Migration

Collections created by the retired custom store may contain a
`collection_meta` point instead of real collection metadata. The runtime does
not support them. For each affected alias:

1. Identify its current physical collection.
2. Rebuild corpus caches if necessary.
3. Materialize a new LlamaIndex collection using the intended alias profile.
4. Verify `.result.config.metadata.attestation` and the artifact manifest.
5. Test it behind challenger.
6. Promote it to champion.
7. Delete the legacy collection only after rollback is no longer required.

Never copy the old sentinel payload into collection metadata as a shortcut;
the rebuilt manifest and node schema are part of the migration.

## Release Acceptance

Before deploying a RAG code or catalog change, run the local contract suite:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --group lint ruff check \
  src/rag src/app_config/runtime src/gateway tests/rag tests/gateway

UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=src uv run --group test pytest \
  tests/rag tests/gateway tests/eval \
  tests/api/test_processing_request_contract.py \
  tests/api/test_rag_lifecycle.py::TestRAGServiceResolution -q
```

Then repeat the promotion workflow against deployed Qdrant, Postgres,
embeddings, reranker, vLLM, and judge services. Mocked tests are necessary but
do not replace one real challenger run for each declared benchmark suite.

## Promotion Checklist

- `plan` passes.
- Source build reports no unexpected failures.
- Candidate manifest and Qdrant attestation match.
- Challenger alias resolves to the intended physical collection.
- Retrieval benchmark rows and samples are complete.
- Context/generation benchmark rows use the intended judge identity.
- Gateway challenger smoke test returns expected source metadata.
- No temporary benchmark collections remain.
- Previous champion collection remains available for rollback.
- Build-run id and candidate collection name are recorded in the change log.
