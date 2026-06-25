# RAG Operations

This is the operator-facing workflow for RAG KB releases on the server. The
main entrypoint is the `rag` CLI (`rag.cli.app`), run through the `ops`
Compose service via `ops/rag_ops.sh`. Airflow uses the same application
service (`AliasService`) directly through the `rag_alias_apply` DAG, not the
CLI process.

For the conceptual model and runtime architecture, see section 5 of
`docs/architecture/system-design.md`. This document focuses on operator
commands and server workflow.

## Operator Contract

`catalog.toml` is desired state. Postgres (`rag_releases`,
`rag_alias_deployments`) is applied state. The supported production flow is:

```text
edit catalog.toml (build/retrieve config for an alias)
  -> rag alias diff       (desired vs. applied, no side effects)
  -> rag alias apply      (builds/reuses a release, activates the alias)
  -> rag benchmark run    (run attached benchmarks against the alias)
  -> inspect Postgres eval_runs/eval_samples and gateway behavior
  -> rag alias apply for the default alias (after benchmarks pass)
```

LlamaIndex owns document, node, index, retrieval, query-engine, and evaluator
mechanics. Project code owns catalog identity, source adapters, content-
addressed releases, alias diff/apply reconciliation, benchmark labels,
orchestration, and DB persistence. There is no supported legacy vector-store,
retriever, `collection_meta` sentinel, `[[sources]]`, KB-local source-id path,
or build-run/promotion CLI.

A Qdrant alias is a mirror of applied state, not the source of truth: the
active `rag_alias_deployments` row decides what serves traffic, and a release
is only ever deleted from Qdrant after being marked retired in Postgres.

## Prerequisites

Before running an alias command, verify:

- `catalog.toml` and `runtime.toml` are from the deployed release;
- `assets/rag_data` is writable by the `ops` container;
- Qdrant, embeddings, and reranker services are healthy;
- vLLM is healthy when running generation or judge benchmarks;
- `GATEWAY_AGENT042_DB_URL` points at the control-plane/eval Postgres database
  (`rag_release_builds`, `rag_releases`, `rag_alias_deployments`, `eval_runs`,
  `eval_samples`);
- the configured embedding model and dimension match collections that will be
  queried;
- the configured judge model appears in the selected OpenAI-compatible
  backend, and external judges declare `eval.judge.context_window`.

For a local environment, install the relevant dependency surfaces and apply
the control-plane SQL migrations:

```bash
uv sync --extra rag --extra gateway --extra airflow-worker
bash bootstrap/apply_agent042_db_migrations.sh
```

This isn't specific to the `rag` CLI: the gateway's eval/chat persistence and
the Airflow `rag_alias_apply` DAG depend on the same control-plane schema.
Release-based server deploys apply it automatically as part of
`deploy_release.sh`; local checkouts still need to run it by hand after
pulling a change that adds a new migration file.

## Naming

- KB id: logical knowledge base id from `catalog.toml`, for example
  `ml_papers_core` or `pytorch_reference`.
- Source adapter: catalog-declared behavior for loading a source instance.
  Source adapters live in `[[source_adapters]]`; benchmark-capable adapters
  live in `[[benchmark_adapters]]`.
- Source instance id: globally meaningful source id, for example
  `ml_papers_core.papers` or `pytorch_reference.docs`.
- Source role: `role = "corpus"` participates in normal KB builds;
  `role = "benchmark"` is normalized for benchmarking and excluded from
  release builds.
- Source manifest: curated input list or adapter-specific config at the
  conventional path
  `assets/rag_data/source_instances/<source_instance_id>/manifest.toml`.
- Alias: per-KB named pointer in the catalog, for example `champion` or
  `challenger`; each declares a nested `build` profile (chunking, dense/sparse
  encoder) and a `retrieve` profile (strategy, top_k, threshold, reranker).
  `default_alias` on the KB names the one alias subject to the evaluation-
  coverage gate.
- Release: an immutable, content-addressed build result. Its id
  (`ragrel_<kb>_<16hex>`) and Qdrant collection name
  (`rag__<kb>__<16hex>`) are derived from a fingerprint of the build config,
  source declaration, transformation, and source snapshot -- not a
  timestamp. The same release is reused across aliases/KBs whenever its
  fingerprint matches.
- Alias deployment: the Postgres row recording which release is active for a
  (kb_id, alias) pair. This, not the Qdrant alias, is the runtime serving
  source of truth.
- Release manifest: full build provenance JSON under
  `assets/rag_data/knowledge_bases/<kb>/releases/<release_id>.json`. Immutable
  once written.

The catalog no longer supports legacy `[[sources]]` entries, KB-local source
selectors, or flat alias fields. Operator commands use global
`<source_instance_id>` values and nested `build`/`retrieve` alias config.

## Artifact Layout

```text
assets/rag_data/
  source_instances/<source_instance_id>/
    manifest.toml
    raw/
    extracted/
    chunks/<transformation_digest>/
    benchmark/
      corpus.jsonl
      cases.jsonl
      labels.jsonl
      metadata.json

  knowledge_bases/<kb_id>/
    releases/<release_id>.json
```

`raw/`, `extracted/`, and `chunks/` are resumable caches containing native
LlamaIndex data. They are not alternate project document/chunk contracts.
Benchmark results do not live here; Postgres is their only result store.

## Server CLI

Run commands from the deployment root on the server. Output is JSON to
stdout when not a TTY (always JSON inside `ops`); logs go to stderr.

Validate the catalog: schema, alias build/retrieve compatibility, and
references. Always run this after editing `catalog.toml`:

```bash
bash ops/rag_ops.sh python -m rag.cli.app catalog validate
```

Compare desired (catalog) vs. applied (Postgres) state for one alias. No
side effects; exits `1` when drift is found:

```bash
bash ops/rag_ops.sh python -m rag.cli.app alias diff pytorch_reference challenger
```

Show diff for every alias declared on a KB:

```bash
bash ops/rag_ops.sh python -m rag.cli.app alias status pytorch_reference
```

Make an alias match its catalog declaration. This resolves or builds the
release the diff implies, then activates the deployment. Building fetches,
extracts, and chunks source content only on cache miss; pass
`--refresh-sources` to force a re-fetch even when no drift would otherwise
trigger a rebuild:

```bash
bash ops/rag_ops.sh python -m rag.cli.app alias apply pytorch_reference challenger
```

The default alias (`champion`, typically) refuses to silently build and
activate an unevaluated release. Use the bootstrap overrides only for a new
KB's first release or a genuine emergency, and record the action:

```bash
bash ops/rag_ops.sh python -m rag.cli.app alias apply pytorch_reference champion \
  --allow-build-default --allow-unevaluated
```

Disambiguate when multiple releases match the desired build/source state:

```bash
bash ops/rag_ops.sh python -m rag.cli.app alias apply pytorch_reference challenger \
  --release ragrel_pytorch_reference_<fingerprint>
```

Inspect releases:

```bash
bash ops/rag_ops.sh python -m rag.cli.app release list --kb pytorch_reference
bash ops/rag_ops.sh python -m rag.cli.app release show ragrel_pytorch_reference_<fingerprint>
```

Expert source diagnostics (not part of the normal workflow -- `alias apply`
resolves and builds sources on its own):

```bash
bash ops/rag_ops.sh python -m rag.cli.app source inspect pytorch_reference.docs
bash ops/rag_ops.sh python -m rag.cli.app source rebuild pytorch_reference.docs
```

## Build Rules

- `alias apply` derives a release's build config (chunking, dense/sparse
  encoder) from the target alias's catalog declaration. It only fetches,
  extracts, and chunks source content when the cache is missing or
  `--refresh-sources` is passed.
- A release is content-addressed: the same build config plus the same
  source declaration (and, for an exact reuse hit, the same source snapshot)
  always resolves to the same `release_id` and Qdrant collection name. `alias
  apply` reuses an existing matching release instead of rebuilding.
- Dense alias `retrieve.strategy` configs can query dense or hybrid releases.
  Hybrid/sparse `retrieve.strategy` configs require a release with a sparse
  encoder; `alias apply` refuses to activate an incompatible combination.
- Challenger aliases should normally be applied and benchmarked before the
  default alias is applied to the same release.
- Provider identity (embedding/sparse/reranker model + dimension) is
  validated against the live provider service before build and before
  retrieval; a mismatch is a refused apply, not a silent rebuild.

## Candidate Release Workflow

1. Edit `catalog.toml`'s `challenger` alias build/retrieve config.
2. Run `rag catalog validate`.
3. Run `rag alias diff pytorch_reference challenger` to confirm the intended
   drift.
4. Run `rag alias apply pytorch_reference challenger`.
5. Run the benchmarks declared for that KB against `challenger`.
6. Smoke-test gateway retrieval against `challenger`.
7. Apply the exact same release to `champion` only after the candidate
   passes:
   `rag alias apply pytorch_reference champion --release <release_id>`.

Building/activating a non-default alias never changes default-alias serving
traffic by itself.

## Benchmark Workflow

A benchmark source instance is `role = "benchmark"`, belongs to exactly one
KB, and declares one or more suites in `benchmark.suites`:

```text
retrieval_quality
context_quality
generation_quality
```

`rag benchmark run` ensures normalized corpus/case/label artifacts are
prepared (re-normalizing automatically when the source manifest or labels
changed) before running:

```bash
bash ops/rag_ops.sh python -m rag.cli.app benchmark run \
  pytorch_reference.qa_benchmark --alias challenger
```

Run every benchmark source instance attached to a KB:

```bash
bash ops/rag_ops.sh python -m rag.cli.app benchmark run \
  --kb pytorch_reference --alias challenger
```

List/show recorded runs:

```bash
bash ops/rag_ops.sh python -m rag.cli.app benchmark list --kb pytorch_reference
bash ops/rag_ops.sh python -m rag.cli.app benchmark show <eval_run_id>
```

The alias supplies current build (chunking, encoder) and retrieval
(strategy, top_k, threshold, reranker) parameters via its active release. If
`corpus.jsonl` is populated, the runner builds a temporary, disposable
benchmark collection with that profile and deletes it in `finally`. If it is
empty, cases query the alias's live release collection directly.

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

Both clients use LlamaIndex `OpenAILike`, so self-hosted model names are
valid. Before judge runs, confirm the configured model is listed by the
selected backend's `/v1/models` endpoint.

### Result Verification

Every aggregate metric is one `eval_runs` row; every per-case observation is
an `eval_samples` row. A successful run records the explicit alias, RAG
release id, alias deployment id, build/retrieval config digests, physical
collection, benchmark artifact digests, prompt identity for generation, and
actual judge backend/model for judged metrics.

Example SQL:

```sql
select
    r.created_at,
    r.dataset_name,
    r.knowledge_base,
    r.rag_alias,
    r.rag_release_id,
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

Treat missing sample rows, missing release/digest identity, unexpected judge
identity, or leftover temporary collections as failed acceptance checks even
when aggregate scores were written.

## Airflow

Airflow's `rag_alias_apply` DAG calls `AliasService.apply()` directly through
the same factories the CLI uses -- not by shelling out to a CLI process.

Parameters:

- `kb_id`, `alias`: the target to reconcile.
- `release_id`: optional, disambiguates an ambiguous reusable release.
- `refresh_sources`: force a re-fetch of source content even when no drift
  would otherwise trigger a rebuild.
- `allow_unevaluated`, `allow_build_default`: the same default-alias
  bootstrap overrides as the CLI. Use sparingly and record the run.
- `sync_dvc`, `dvc_base_branch`, `dvc_bot_branch`: when `sync_dvc=true`, a
  follow-up task DVC-syncs the KB's source-instance artifact directories
  after the alias apply task succeeds.

DVC policy:

- Source instance `manifest.toml` files stay in Git because they are curated
  operator input.
- Generated source-instance artifacts (`extracted/`, `chunks/`, `benchmark/`)
  are DVC candidates.
- Raw cache (`raw/`) is server-local by default and intentionally not
  DVC-tracked by this DAG.
- A failed DVC sync does not undo the alias apply that already happened; it
  is a follow-up data-retention task, not a release gate.

A separate `rag_collection_cleanup` DAG runs `@daily` and is described under
Inspection below.

## Inspection

Use the Qdrant API/dashboard, `rag release`/`rag alias status`, and direct
Postgres queries for observability:

- compare `rag alias status <kb>` against live Qdrant aliases;
- list releases for a KB and their `retired_at` state;
- identify physical collections with no corresponding `rag_releases` row
  (always investigate before deleting -- this DAG never deletes them itself,
  since a release row is written only after a build finishes);
- create snapshots through Qdrant's snapshot API before any manual deletion.

`rag_collection_cleanup` decides liveness from Postgres, not from whether a
Qdrant alias happens to point at a collection: it protects every collection
referenced by an active or pending deployment, retains the newest few
superseded deployments per (kb_id, alias) as a rollback buffer, marks a
release retired in Postgres before deleting its collection, and leaves
release manifests on disk untouched after retirement.

Inspect one collection's release-v2 attestation from the server:

```bash
curl -s "$QDRANT_URL/collections/$COLLECTION" \
  | jq '.result.config.metadata.attestation'
```

## Runtime Observability

`rag.runtime.RagRuntime` returns result-level observability alongside native
`NodeWithScore` results:

- `provenance`: one row per resolved KB/alias with Qdrant alias, physical
  collection, release id, retrieval strategy/capability, hit count, no-hit
  flag, score summary, and source timings.
- `timings_ms`: total runtime retrieval latency.
- `diagnostics`: requested/resolved/skipped source counts, total hit count,
  and no-hit flag.

The gateway logs these diagnostics after retrieval. This is the first
feedback surface for Grafana/log-based runtime panels; it does not change
alias apply behavior. The gateway raises `RagDatabaseUnavailableError` (not a
generic startup failure) when the control-plane database is unreachable.

## Rollback

Rollback is another alias apply, pointed at a previous release:

```bash
bash ops/rag_ops.sh python -m rag.cli.app alias apply pytorch_reference champion \
  --release ragrel_pytorch_reference_<previous-fingerprint>
```

Before rollback, run `rag release show <release_id>` and confirm its
retrieval capability is compatible with the alias's `retrieve.strategy`.

## Failure Recovery

| Failure | Operator action |
| --- | --- |
| Catalog validation fails | Fix configuration; rerun `rag catalog validate`. |
| `alias diff` reports a provider identity mismatch | Fix the embedding/sparse/reranker deployment or catalog model declaration; do not apply through it. |
| Fetch/extraction/chunking fails during `alias apply` | Inspect the error; retry, optionally with `--refresh-sources`. |
| Release build fails | The build attempt is recorded as `failed` in `rag_release_builds`; no deployment is changed. Fix and retry `alias apply`. |
| `alias apply` refuses an incompatible retrieval strategy | Fix `retrieve.strategy` or use a release built with the required encoder. |
| Default-alias apply is refused without `--allow-*` flags | This is the evaluation-coverage gate working as intended; benchmark the release on a non-default alias first. |
| Challenger benchmark fails | Keep champion unchanged; preserve DB observations for comparison. |
| Judge model is unavailable | Fix `[eval.judge]` or model deployment; do not interpret missing judged metrics as passes. |
| Benchmark process is interrupted | Confirm its temporary `eval__*` collection was removed before rerunning. |
| Champion regresses after apply | `alias apply` the previous release back to champion. |

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

Then repeat the candidate release workflow against deployed Qdrant,
Postgres, embeddings, reranker, vLLM, and judge services. Mocked tests are
necessary but do not replace one real challenger apply and benchmark run for
each declared suite.

## Apply Checklist

- `rag catalog validate` passes.
- `rag alias diff` showed the expected drift before applying.
- `rag alias apply` reports `action` of `built_release` or `reused_release`
  with no provider identity mismatch.
- Challenger alias resolves to the intended release.
- Retrieval benchmark rows and samples are complete.
- Context/generation benchmark rows use the intended judge identity.
- Gateway challenger smoke test returns expected source metadata.
- No temporary benchmark collections remain.
- Previous champion release remains available (not yet retired) for
  rollback.
- The applied release id is recorded in the change log.
