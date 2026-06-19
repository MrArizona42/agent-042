# RAG Operations

This is the operator-facing workflow for RAG builds on the server. The main
entrypoint is the `rag-ops` Compose service, wrapped by `scripts/rag_ops.sh`.
Airflow uses the same CLI through the `rag_lifecycle` DAG.

For the conceptual model and runtime architecture, see section 5 of
`docs/architecture/system-design.md`. This document focuses on operator
commands and server workflow.

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
  alias targets. New builds store it at
  `.result.config.metadata.attestation`; existing legacy collections may still
  expose a `collection_meta` sentinel until they are rebuilt or retired.
- Benchmark artifacts: normalized `cases.jsonl`, `labels.jsonl`, and
  `metadata.json` under
  `assets/rag_data/source_instances/<benchmark_source_instance_id>/benchmark/`.

The catalog no longer supports legacy `[[sources]]` entries or KB-local source
selectors. Operator commands use global `--source-instance <id>` values.

## Server CLI

Run commands from the deployment root on the server:

```bash
bash current/scripts/rag_ops.sh python -m rag.sources.cli build-source \
  --catalog catalog.toml \
  --source-instance pytorch_reference.docs \
  --rag-data-root assets/rag_data
```

Build a collection from existing chunk artifacts:

```bash
bash current/scripts/rag_ops.sh python -m rag.sources.cli materialize \
  --catalog catalog.toml \
  --kb pytorch_reference \
  --source-instance pytorch_reference.docs \
  --alias-config challenger \
  --rag-data-root assets/rag_data
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
  --kb pytorch_reference \
  --alias challenger \
  --collection rag__pytorch_reference__20260605_120000
```

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
- During the Phase 3/4 migration boundary, do not promote LlamaIndex-built
  collections to serving aliases. The legacy runtime cannot consume their
  vector/payload layout; Phase 4 installs the matching runtime path.
- Benchmark execution must receive an explicit alias. A benchmark source
  instance is attached to exactly one KB through `source_instance.knowledge_base`;
  the alias supplies the KB runtime/build profile for that run.

## Airflow

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

Use `experiments/rag/rag_ops.ipynb` for direct Qdrant observability:

- list collections and aliases;
- compare expected catalog aliases with live Qdrant aliases;
- inspect collection attestation metadata and sample points;
- identify old physical collections not behind any alias;
- create snapshots or run danger-zone cleanup cells deliberately.

The notebook is not a build entrypoint. Builds use CLI or Airflow.

## Runtime Observability

`rag.runtime.RagRuntime` returns result-level observability alongside hits:

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
