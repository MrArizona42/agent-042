# RAG Operations

This is the operator-facing workflow for RAG builds on the server. The main
entrypoint is the `rag-ops` Compose service, wrapped by `scripts/rag_ops.sh`.
Airflow uses the same CLI through the `rag_lifecycle` DAG.

## Naming

- KB id: logical knowledge base id from `src/shared/catalog.toml`, for example
  `ml_papers_core` or `pytorch_reference`.
- Source type: connector/extractor family, for example `arxiv_paper` or
  `html_docs`.
- Source instance id: KB-local source id, for example `papers` or `docs`.
- Source manifest: curated input list under `assets/rag_data/<kb>/sources.toml`.
- Alias config: per-KB retrieval profile in the catalog, for example
  `champion` or `challenger`; it controls `top_k`, threshold, strategy, and
  reranker settings.
- Physical collection: Qdrant collection built from a bundle, named
  `rag__<kb_id>__<timestamp>`. It intentionally does not include alias names.
- Qdrant alias: runtime pointer named `rag__<kb_id>__<alias>`.
- Artifact manifest: full build provenance JSON under
  `assets/rag_data/<kb>/manifests/`.
- Qdrant attestation: compact metadata stored in the collection `_meta` point
  so runtime can validate alias targets.

## Server CLI

Run commands from the deployment root on the server:

```bash
bash current/scripts/rag_ops.sh python -m rag.sources.cli build-source \
  --catalog src/shared/catalog.toml \
  --kb pytorch_reference \
  --source docs \
  --rag-data-root assets/rag_data
```

Build a collection from existing chunk artifacts:

```bash
bash current/scripts/rag_ops.sh python -m rag.sources.cli materialize \
  --catalog src/shared/catalog.toml \
  --kb pytorch_reference \
  --source docs \
  --alias-config challenger \
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

- `build-source` fetches/extracts/chunks source documents. Cache artifacts are
  immutable unless force flags are passed.
- `materialize --alias-config <alias>` uses that alias profile as build input.
  It does not assign or move a Qdrant alias.
- `promote-alias --alias <alias>` points `rag__<kb>__<alias>` at an attested
  physical collection.
- Dense alias configs can query dense or hybrid collections.
- Hybrid alias configs require hybrid collections.
- Challenger collections should normally be built and inspected before
  champion promotion.

## Airflow

Use `rag_lifecycle` for the same lifecycle:

- `kb`: KB id, for example `ml_papers_core`.
- `source`: source instance id, for example `papers`.
- `alias_config`: build profile, usually `challenger` for test builds.
- `promote_alias`: optional runtime alias to repoint after materialization.
  Leave empty for build-only runs.
- `document_ids` and `limit`: scoped smoke builds.
- `force_fetch`, `force_extract`, `force_chunk`, `force_recreate`: explicit
  cache/collection invalidation controls.

## Inspection

Use `experiments/rag/rag_ops.ipynb` for direct Qdrant observability:

- list collections and aliases;
- compare expected catalog aliases with live Qdrant aliases;
- inspect attestation metadata and sample points;
- identify old physical collections not behind any alias;
- create snapshots or run danger-zone cleanup cells deliberately.

The notebook is not a build entrypoint. Builds use CLI or Airflow.

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
