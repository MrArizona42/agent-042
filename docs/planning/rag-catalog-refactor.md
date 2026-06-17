# RAG Catalog Refactor Proposal

This note captures the intended catalog direction after the RAG pipeline
isolation work. It is a design proposal, not the currently implemented schema.

The goal is to make `catalog.toml` describe a typed graph of runtime knowledge
bases, source instances, source adapters, and benchmark-capable source
instances without hiding behavior in central Python registries.

## Design Decisions

- Retire `source_type`. Adapter identity is the machine-facing behavior
  selector.
- Replace hidden default adapter registration with declarative
  `[[source_adapters]]` and `[[benchmark_adapters]]`.
- Keep source and benchmark adapter declarations separate in the catalog, even
  if their Python implementations live in the same adapters package.
- Use only `id` and `description` as descriptive strings on catalog instances.
  Avoid `label`, `name`, and selection-specific description fields.
- Make source instance IDs globally meaningful, for example
  `pytorch_reference.docs`.
- Require each source instance manifest to live at
  `assets/rag_data/source_instances/<source_instance_id>/manifest.toml`.
  The catalog should derive this path from source instance id instead of storing
  arbitrary manifest paths.
- Treat benchmarks as source instances with additional benchmark capability.
  They are attached to the knowledge base they evaluate.
- Use `role = "corpus"` and `role = "benchmark"` to keep normal KB rebuilds
  from indexing benchmark cases and labels.
- A benchmark adapter implements the normal source lifecycle contract and also
  benchmark preparation.
- `build-source` builds normal corpus source instances.
- `prepare-benchmark` prepares benchmark artifacts such as cases, labels,
  qrels, evidence refs, reference answers, and rubrics.
- `materialize` builds Qdrant collections from corpus source instances only by
  default.
- Benchmark runs must receive an explicit alias at execution time. They must
  not silently inherit `default_alias`.
- Benchmark generation settings should reuse the project-wide generation
  benchmark contracts. RAG-specific benchmark outputs should add retrieval
  provenance, KB id, alias, collection name, and grounding data.

## Current `tasks.adapter` Question

The current catalog has task-level `adapter = { enabled = false }`. In code this
is used by Gateway for task-specific LoRA/model adapter selection, not for RAG
source ingest. Keeping the name `adapter` in the same catalog that introduces
source and benchmark adapters is confusing.

For the next schema, keep this configuration in `catalog.toml` but rename it to
`lora_adapter`. The field controls task-level LoRA/model adapter selection and
must stay distinct from source and benchmark adapters.

The example below omits `lora_adapter` values for brevity. When present, they
belong under `[[tasks]]`, not under source or benchmark adapter declarations.

## Suggested `catalog.toml`

```toml
schema_version = 3

# Adapter factories are Python callables returning an adapter object.
# Source adapters must implement the normal source lifecycle contract.
[[source_adapters]]
id = "generic.http_html"
version = "1"
description = "Fetches HTTP HTML pages and extracts readable text sections."
factory = "rag.adapters:make_http_html_adapter"

[[source_adapters]]
id = "generic.arxiv_pdf"
version = "1"
description = "Resolves arXiv paper ids to PDFs, fetches them, and extracts text."
factory = "rag.adapters:make_arxiv_pdf_adapter"

# Benchmark adapters extend the source lifecycle contract with benchmark
# preparation. Their code can live in the same adapters package as source
# adapters.
[[benchmark_adapters]]
id = "benchmark.pytorch_qa"
version = "1"
description = "Loads QA examples, reference answers, and expected evidence for PyTorch docs."
factory = "rag.adapters:make_pytorch_qa_benchmark_adapter"

[[benchmark_adapters]]
id = "benchmark.beir_scifact"
version = "1"
description = "Loads BEIR SciFact corpus records, queries, and qrels."
factory = "rag.adapters:make_beir_scifact_benchmark_adapter"


[[tasks]]
id = "chat"
description = "Open-ended ML/DL/AI/LLM research discussion, conceptual explanation, comparison, brainstorming, planning, and general Q&A that is not mainly code debugging and not a request to summarize provided text."
knowledge_bases = ["ml_papers_core"]

[[tasks]]
id = "code"
description = "Programming help for ML systems: writing code, debugging tracebacks, refactoring, explaining APIs, fixing integration issues, and reasoning about implementation details."
knowledge_bases = ["pytorch_reference"]

[[tasks]]
id = "summarize"
description = "Summarize or condense user-provided content into a shorter form such as TL;DR, bullets, outline, recap, or structured summary without relying on external knowledge retrieval."
knowledge_bases = []


[[knowledge_bases]]
id = "ml_papers_core"
description = "Curated full-text ML/AI papers for literature-grounded discussion."
update_strategy = "replace"
default_alias = "champion"
aliases.champion = { top_k = 5, score_threshold = 0.35, retrieval_strategy = "dense", reranker_multiplier = 1 }
aliases.challenger = { top_k = 5, score_threshold = 0.01, retrieval_strategy = "hybrid", reranker = "cross-encoder/ms-marco-MiniLM-L-6-v2", reranker_multiplier = 4 }

[[knowledge_bases]]
id = "pytorch_reference"
description = "PyTorch documentation for coding assistance."
update_strategy = "replace"
default_alias = "champion"
aliases.champion = { top_k = 5, score_threshold = 0.35, retrieval_strategy = "dense", reranker_multiplier = 1 }
aliases.challenger = { top_k = 5, score_threshold = 0.01, retrieval_strategy = "hybrid", reranker = "cross-encoder/ms-marco-MiniLM-L-6-v2", reranker_multiplier = 4 }


# role allowed values: "corpus", "benchmark".
# role = "corpus" is included in normal KB builds/materialization.
# role = "benchmark" is excluded from normal KB builds and used by
# prepare-benchmark/run-benchmark.
[[source_instances]]
id = "ml_papers_core.papers"
description = "Curated arXiv ML/AI papers."
role = "corpus"
knowledge_base = "ml_papers_core"
adapter = { id = "generic.arxiv_pdf", version = "1" }

[[source_instances]]
id = "pytorch_reference.docs"
description = "Official PyTorch documentation pages."
role = "corpus"
knowledge_base = "pytorch_reference"
adapter = { id = "generic.http_html", version = "1" }


# benchmark.contains allowed values:
# "queries", "qrels", "answers", "scores", "evidence_text",
# "relevant_doc_ids", "relevant_chunk_ids", "rubrics".
[[source_instances]]
id = "ml_papers_core.scifact_benchmark"
description = "SciFact benchmark cases and labels for evaluating ml_papers_core."
role = "benchmark"
knowledge_base = "ml_papers_core"
adapter = { id = "benchmark.beir_scifact", version = "1" }

benchmark = {
  contains = ["queries", "qrels", "relevant_doc_ids"],
  metrics = ["recall_at_k", "mrr", "ndcg"]
}

[[source_instances]]
id = "pytorch_reference.qa_benchmark"
description = "Question-answering benchmark cases and expected evidence for PyTorch documentation."
role = "benchmark"
knowledge_base = "pytorch_reference"
adapter = { id = "benchmark.pytorch_qa", version = "1" }

benchmark = {
  contains = ["queries", "answers", "relevant_doc_ids", "evidence_text"],
  metrics = ["recall_at_k", "answer_groundedness"]
}
```

## Artifact Layout

Source artifacts should be keyed by globally unique source instance id.
Knowledge-base directories should hold collection-level outputs, not
source-instance-local raw/extracted/chunk artifacts.

Benchmark artifacts should live next to the benchmark source instance they
belong to, not in a separate `rag_benchmarks` namespace.

Suggested generated layout:

```text
assets/rag_data/
  source_instances/
    <source_instance_id>/
      manifest.toml
      raw/
      extracted/
      chunks/
      benchmark/
        cases.jsonl
        labels.jsonl
        metadata.json

  knowledge_bases/
    <kb_id>/
      manifests/
      metadata/
        build_runs/
```

Postgres is required for benchmark results:

- aggregate metric rows go to `eval_runs`;
- per-case observations go to `eval_samples.detail`;
- RAG-specific provenance and diagnostics live in JSON fields when they do not
  deserve first-class columns yet.

Benchmark runs should fail early when database persistence is not configured.
The artifact tree stores benchmark inputs and normalized cases/labels, not
benchmark results.

`build-source` and `materialize` should ignore `role = "benchmark"` by default.
`prepare-benchmark` should write normalized benchmark artifacts. `run-benchmark`
should require:

- a benchmark source instance id;
- an explicit alias;
- prepared benchmark artifacts, or a clear error instructing the operator to run
  `prepare-benchmark`.

The evaluated KB is `source_instance.knowledge_base`.

The manifest path is always derived from source instance id:

```text
assets/rag_data/source_instances/<source_instance_id>/manifest.toml
```

Do not keep arbitrary `manifest = "..."` paths in the final schema.

## Normalized Contracts

Keep the source lifecycle contract centered on existing RAG objects, then add
benchmark-specific normalized contracts:

- `SourceDocument`: buildable corpus document selected by an adapter.
- `BenchmarkCase`: query/question/input case, split, optional messages, and
  metadata.
- `BenchmarkLabel`: labels for a case, such as qrels, evidence refs, relevant
  document IDs, relevant chunk IDs, reference answers, scores, or rubrics.
- `BenchmarkRunObservation`: per-case output of a benchmark run, including KB
  id, alias, collection name, retrieved hits, optional generated answer, and
  metric inputs/outputs.

Labels should be optional so the same benchmark source instance model can
support retrieval benchmarks, answer benchmarks, scored datasets, and unlabeled
smoke/regression sets.

## Adapter Capability Shape

Use separate protocols for behavior and explicit capability metadata for clear
planning/error messages.

Recommended contracts:

```python
AdapterCapability = Literal["source", "benchmark"]


class SourceAdapter(Protocol):
    adapter_id: str
    version: str
    capabilities: frozenset[AdapterCapability]

    def validate_manifest(self, manifest: Any) -> Any: ...
    def list_documents(self, manifest: Any) -> list[SourceDocument]: ...
    def fetcher(self) -> SourceFetcher: ...
    def extractor(self) -> SourceExtractor: ...


class BenchmarkAdapter(SourceAdapter, Protocol):
    capabilities: frozenset[AdapterCapability]

    def prepare_benchmark(self, manifest: Any) -> BenchmarkPreparedArtifacts: ...
```

The expected capability sets are:

```python
SourceAdapter.capabilities == frozenset({"source"})
BenchmarkAdapter.capabilities == frozenset({"source", "benchmark"})
```

Rules:

- `[[source_adapters]]` factories must return objects with `"source"` in
  `capabilities` and must satisfy `SourceAdapter`.
- `[[benchmark_adapters]]` factories must return objects with both `"source"`
  and `"benchmark"` in `capabilities` and must satisfy `BenchmarkAdapter`.
- `role = "corpus"` source instances may use only source-capable adapters.
- `role = "benchmark"` source instances must use benchmark-capable adapters.
- Capability metadata is not trusted by itself. Loader validation should also
  check that required methods are present and callable.

This avoids marker methods such as `supports_benchmark()` as the primary
contract. Marker methods can be useful for custom diagnostics, but the generic
loader should validate protocols and capabilities instead.

## RAG Generation Observation Extension

RAG generation benchmarks should reuse the project-wide generation benchmark
result model. They should not create a parallel RAG-only generation report.

The extension point is the existing `eval_runs.extra` and
`eval_samples.detail` JSONB fields. Do not add a separate RAG generation result
table unless JSONB query patterns become a proven bottleneck.

Run-level persistence:

```text
eval_runs
  task                    existing column; generation task name
  dataset_name            benchmark source instance id or stable benchmark id
  metric_name             aggregate metric name
  metric_value            aggregate metric value

  base_model              existing column
  adapter_name            existing column; LoRA adapter name when enabled
  adapter_version         existing column
  lora_alias              existing column

  rag_enabled             true for RAG generation benchmarks
  rag_alias               mandatory alias used for this benchmark run
  knowledge_base          source_instance.knowledge_base
  qdrant_alias            resolved runtime alias
  qdrant_collection       resolved physical collection
  rag_manifest_id         resolved collection manifest id
  retrieval_top_k         alias profile top_k
  score_threshold         alias profile score_threshold
  reranking_strategy      alias profile reranker or "none"

  temperature             existing column; generation run setting
  max_tokens              existing column; generation run setting
  eval_verdict            existing column
  extra                   JSONB run-level details
```

`eval_runs.extra` should include:

```text
extra.rag.benchmark_source_instance_id
extra.rag.benchmark_adapter_id
extra.rag.benchmark_adapter_version
extra.rag.benchmark_artifact_digests
extra.rag.retrieval_strategy
extra.rag.retrieval_capability
extra.rag.source_instance_ids
extra.rag.score_summary
extra.generation.prompt_template
extra.generation.prompt_template_digest
extra.generation.judge_profile
```

Per-sample persistence:

```text
eval_samples
  eval_run_id             existing FK to eval_runs.id
  sample_idx              existing column
  sample_id               BenchmarkCase.id
  input                   BenchmarkCase query/question/messages rendered as text
  output                  generated answer
  reference               reference answer or rubric summary when available
  detail                  JSONB per-case details
```

`eval_samples.detail.generation` should include per-case generation facts:

```text
prompt_digest
prompt_tokens
completion_tokens
latency_ms
finish_reason
error
```

`eval_samples.detail.rag` should include per-case RAG facts:

```text
retrieved[]:
  rank
  chunk_id
  document_id
  source_instance_id
  score
  title
  uri
  in_prompt
  prompt_rank
  text_digest

expected:
  relevant_doc_ids
  relevant_chunk_ids
  reference_answer_ids
  evidence_refs

diagnostics:
  hit_count
  no_hit
  context_tokens
  trimmed_chunk_count

timings_ms:
  resolve
  retrieve
  prompt_assembly
```

Run-level fields stay in `eval_runs`; per-case facts stay in
`eval_samples.detail`. This keeps DB persistence as the source of truth while
preserving enough per-case RAG context for failure analysis and groundedness
metrics.

## Implementation Plan

The migration should be staged so existing RAG builds keep working while the
new model is introduced. Each phase should end with tests passing and the
operator-facing CLI still usable.

### Phase 1: Catalog Schema And Compatibility Layer

Add the new schema models without removing current `[[sources]]` support.

Implementation tasks:

- Add catalog schema models for:
  - `SourceAdapterConfig`;
  - `BenchmarkAdapterConfig`;
  - `SourceInstanceConfig`;
  - `BenchmarkSourceConfig`;
  - `SourceInstanceRole`, with allowed values `"corpus"` and `"benchmark"`.
- Add validation rules:
  - every source instance id is globally unique;
  - every `source_instances[].knowledge_base` references an existing KB;
  - every `source_instances[].adapter` references either `[[source_adapters]]`
    or `[[benchmark_adapters]]`;
  - every source instance has a readable manifest at the derived conventional
    path `assets/rag_data/source_instances/<source_instance_id>/manifest.toml`;
  - `role = "benchmark"` requires a `benchmark` block;
  - `role = "corpus"` must not contain a `benchmark` block;
  - `benchmark.contains` values must come from the documented vocabulary;
  - task `knowledge_bases` entries reference existing KBs.
- Keep legacy `[[sources]]` parsing temporarily and normalize it into
  `SourceInstanceConfig` in memory:
  - legacy `(kb, id)` becomes new source instance id `<kb>.<id>`;
  - legacy `manifest` is used only to copy or validate the one-time migrated
    manifest at the conventional source-instance path;
  - legacy `ingest_adapter` becomes new `adapter`;
  - legacy entries default to `role = "corpus"`;
  - legacy `type` is ignored for behavior and retained only in transitional
    metadata if needed.
- Update `materialize_catalog()` / catalog indexes so callers can query:
  - tasks by id;
  - KBs by id;
  - source instances by global id;
  - corpus source instances by KB;
  - benchmark source instances by KB.
- Add tests for the new schema, legacy normalization, and invalid catalog
  references.

Acceptance criteria:

- Current `catalog.toml` still loads.
- A schema-version-3 sample using `[[source_instances]]` loads.
- Duplicate source instance ids are rejected.
- Schema-version-3 source instances derive manifest paths from id.
- Benchmark sources are visible through the new source-instance index but are
  excluded from normal corpus-source queries.

### Phase 2: Declarative Adapter Loading

Replace `DEFAULT_SOURCE_ADAPTERS` as the source of truth with catalog-declared
adapter factories.

Implementation tasks:

- Add a generic adapter loader that imports `factory = "module:function"` and
  calls it.
- Define adapter capability protocols:
  - source lifecycle capability: validate/list/fetch/extract;
  - benchmark preparation capability: prepare cases and labels.
- Load `[[source_adapters]]` and `[[benchmark_adapters]]` from catalog into an
  adapter registry keyed by `(id, version)`.
- Make built-in adapters available through factory functions, for example:
  - `rag.adapters:make_http_html_adapter`;
  - `rag.adapters:make_arxiv_pdf_adapter`.
- Update source build planning to use the catalog-loaded registry.
- Keep `DEFAULT_SOURCE_ADAPTERS` only as a transitional fallback for legacy
  schema tests, then remove it in a later cleanup phase.
- Add tests for:
  - factory import errors;
  - missing adapter declarations;
  - adapter capability mismatch;
  - benchmark source using an adapter without benchmark capability.

Acceptance criteria:

- New catalog adapter declarations are the normal path.
- Adding a new adapter no longer requires editing a central default registry.
- Benchmark adapters can share implementation modules with source adapters.

### Phase 3: Source-Instance-Centered Artifact Paths

Introduce the new artifact layout and move checked-in source manifests into the
derived source-instance directories.

Implementation tasks:

- Add path helpers for source-instance artifacts:
  - `assets/rag_data/source_instances/<source_instance_id>/manifest.toml`;
  - `raw/`;
  - `extracted/`;
  - `chunks/`;
  - `benchmark/`.
- Add path helpers for KB-level outputs:
  - `assets/rag_data/knowledge_bases/<kb_id>/manifests/`;
  - `assets/rag_data/knowledge_bases/<kb_id>/metadata/build_runs/`.
- Thread source instance id through fetch, extract, chunk, bundle collection,
  materialization manifests, and build-run provenance.
- Move existing checked-in source manifests to the conventional source-instance
  paths.
- Rebuild generated raw/extracted/chunk artifacts in the new layout instead of
  preserving legacy generated artifact paths.
- Write new artifacts to the new layout.
- Update DVC sync path selection to operate on source instance paths and KB
  collection-output paths separately.
- Add migration notes or a helper command for moving checked-in manifests.

Acceptance criteria:

- New builds write source artifacts under `source_instances/<id>/...`.
- New collection manifests and build runs write under
  `knowledge_bases/<kb>/...`.
- Generated artifacts can be rebuilt from the new source-instance layout.

### Phase 4: Lifecycle Semantics For Source Roles

Make `role` control what lifecycle commands do by default.

Implementation tasks:

- Update source resolution:
  - `build-source --kb <kb>` selects only `role = "corpus"` by default;
  - `materialize --kb <kb>` collects only corpus source instance chunks by
    default;
  - explicit `--source-instance <id>` may target one source instance directly;
  - explicit benchmark source instance targets are rejected by `build-source`
    unless a deliberate benchmark preparation command is used.
- Add source-instance-oriented CLI arguments:
  - `--source-instance <id>` for exact global source instance selection;
  - keep legacy `--source <local-id>` temporarily when `--kb` is supplied.
- Update Airflow `rag_lifecycle` params to use source instance ids while
  keeping backward-compatible `source` handling during migration.
- Record selected source instance ids in `BuildRun`.
- Add tests for:
  - all-corpus build for a KB;
  - selected source instance build;
  - benchmark source excluded from normal build;
  - materialization excluding benchmark artifacts.

Acceptance criteria:

- A normal KB rebuild cannot accidentally index benchmark cases or labels.
- Operator commands can address source instances globally.

### Phase 5: Benchmark Preparation Contracts

Add benchmark preparation while reusing source instances and adapters.

Implementation tasks:

- Add normalized models:
  - `BenchmarkCase`;
  - `BenchmarkLabel`;
  - `BenchmarkRunObservation`.
- Add a `BenchmarkAdapter` capability protocol.
- Add `prepare-benchmark` stage:
  - input: benchmark source instance id;
  - adapter validates benchmark manifest;
  - adapter emits normalized cases and labels;
  - output artifacts go under
    `assets/rag_data/source_instances/<source_instance_id>/benchmark/`.
- Preserve optionality:
  - labels may be empty or absent;
  - answers, qrels, evidence refs, relevant doc ids, relevant chunk ids, scores,
    and rubrics are optional according to `benchmark.contains`.
- Add tests for:
  - adapter output validation;
  - missing benchmark block;
  - `benchmark.contains` mismatch;
  - artifact writing and reading.

Acceptance criteria:

- Benchmark preparation is independent from corpus chunk rebuilding.
- Labels can change without forcing a collection rebuild.
- A benchmark-capable source instance remains attached to the KB it evaluates.

### Phase 6: Benchmark Run Pipeline

Implement benchmark execution against an explicit live KB alias.

Implementation tasks:

- Add `run-benchmark` command:
  - requires `--source-instance <benchmark-id>`;
  - requires `--alias <alias>`;
  - resolves KB from `source_instance.knowledge_base`;
  - rejects aliases not declared on that KB;
  - rejects missing prepared benchmark artifacts with a clear
    `prepare-benchmark` instruction.
- Use `rag.runtime.RagRuntime` for retrieval so benchmark runs validate the
  same Qdrant alias, attestation, embedding dimension, and retrieval capability
  as production runtime.
- For retrieval benchmarks, compute metrics from retrieved hits and labels.
- For generation benchmarks, call the project-wide generation benchmark
  machinery and add RAG retrieval provenance to observations.
- Require configured database persistence before executing benchmark cases.
- Persist aggregate metric rows to `eval_runs`.
- Persist per-case observations to `eval_samples.detail`.
- Include in DB rows:
  - benchmark source instance id;
  - KB id;
  - mandatory alias;
  - resolved Qdrant alias;
  - physical collection name;
  - collection manifest id;
  - adapter id/version;
  - benchmark artifact digests;
  - generation settings when applicable.
- Add tests for mandatory alias, unknown alias, missing prepared artifacts,
  retrieval observations, and metric output.

Acceptance criteria:

- Benchmark results are always tied to an explicit alias and resolved
  collection.
- No benchmark run silently uses `default_alias`.
- Database rows are reproducible from catalog, benchmark artifacts, and live
  collection provenance.

### Phase 7: Cleanup And Schema Flip

Remove transitional compatibility once the new schema and layout are adopted.

Implementation tasks:

- Convert checked-in `catalog.toml` to `schema_version = 3`.
- Move checked-in source manifests into
  `assets/rag_data/source_instances/<source_instance_id>/manifest.toml`.
- Remove legacy `[[sources]]` support.
- Remove catalog `manifest` fields from source instances.
- Remove legacy `source_type` behavior and tests.
- Remove `DEFAULT_SOURCE_ADAPTERS`.
- Rename task-level LoRA/model adapter configuration from `adapter` to
  `lora_adapter`.
- Update operations docs, Airflow docs, and examples.

Acceptance criteria:

- New schema is the only supported catalog schema.
- Artifact paths match the source-instance / knowledge-base split.
- Source instance manifests are always derived from source instance id.
- The catalog no longer has overloaded adapter or source-type concepts.
