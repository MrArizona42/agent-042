# RAG LlamaIndex Transition Implementation Plan

This document is the implementation plan for moving the RAG mechanics to
LlamaIndex after the catalog refactor. The durable source-instance catalog,
benchmark, artifact, and DB result contracts now live in the main project docs:
`docs/architecture/system-design.md`, `docs/operations/rag-operations.md`, and
`docs/analytics/evaluation-results.md`. This file starts from that project
state and describes the target LlamaIndex contracts, data flows, artifacts, and
migration phases.

## Current Project State

The catalog refactor baseline is implemented. The current code uses the
source-instance-only catalog:

- `[[source_adapters]]`, `[[benchmark_adapters]]`, and `[[source_instances]]`
  exist in schema;
- `SourceInstanceConfig` has `id`, `description`, `role`, `knowledge_base`,
  `adapter`, and optional `benchmark`;
- tasks and knowledge bases use `id` and `description` as their descriptive
  strings;
- `role` is `"corpus"` or `"benchmark"`;
- benchmark source config currently declares only `suites`, with allowed values
  `"retrieval_quality"`, `"context_quality"`, and `"generation_quality"`;
- `SourceInstanceIndex` indexes declared `[[source_instances]]`;
- legacy `[[sources]]`, local `--source <id>` selectors, arbitrary manifest
  paths, and `DEFAULT_SOURCE_ADAPTERS` are removed.

Phases 0-3 are implemented. Source processing and collection materialization
now use LlamaIndex objects; runtime retrieval remains on the legacy path until
Phase 4:

```text
catalog source_instance
  -> catalog-declared adapter factory
  -> source manifest TOML
  -> LlamaIndex Document source descriptors with project metadata
  -> fetch raw artifact
  -> extracted artifact containing a LlamaIndex Document
  -> SentenceSplitter
  -> native TextNode artifact with deterministic UUID id_
  -> SourceNodeBundle
  -> materialize_kb_collection_llamaindex()
  -> VectorStoreIndex + LlamaIndex QdrantVectorStore
  -> Qdrant node points + collection metadata attestation
  -> IndexManifest JSON
  -> Qdrant alias promotion
  -> RagRuntime + custom Retriever
```

The active source/build path no longer depends on project `SourceDocument`,
`ExtractedDocument`, `DocumentSection`, or `Chunk` contracts. They remain only
in legacy compatibility modules/tests and are deleted in Phase 6. Current
durable project RAG contracts include:

- `CollectionAttestation`;
- `IndexManifest`;
- `RetrievalHit`.

Current evaluation contracts already live in `src/rag/evaluation/models.py`:

- `BenchmarkCase`;
- `Qrel`;
- `BenchmarkLabel`;
- `BenchmarkPreparedArtifacts`;
- `RetrievalEvalObservation`;
- `AnswerEvalObservation`;
- `PromotionDecision`.

New LlamaIndex collections store attestation in
`.result.config.metadata.attestation`; no sentinel point is written. Legacy
collections and the pre-Phase-4 runtime still understand the old
`type=collection_meta` sentinel. A deployed Qdrant smoke test confirmed
collection metadata round-trips through:

```text
PUT /collections/<collection>              metadata accepted
GET /collections/<collection>              .result.config.metadata returned
PATCH /collections/<collection>            metadata updated
```

Phase 4 must switch runtime retrieval before a LlamaIndex-built collection is
promoted to a serving alias. Phase 6 removes the remaining sentinel code after
legacy collections are rebuilt or retired.

Current LlamaIndex facts from the checkup:

- locked core package: `llama-index-core==0.14.22`;
- transient Qdrant package inspected: `llama-index-vector-stores-qdrant==0.10.1`;
- `QdrantVectorStore` supports caller-provided collection names, dense config,
  sparse config, named dense/sparse vectors, hybrid search, batch size, payload
  indexes, and sync/async clients;
- LlamaIndex `TextNode.id_` is used as the Qdrant point id, so it must be a
  valid Qdrant point id, usually UUID or integer;
- source node parsing derives deterministic UUID5 `TextNode.id_` values from
  human-readable chunk ids, and transitional materialization passes those ids
  through unchanged;
- LlamaIndex built-in retrieval metrics include `hit_rate`, `mrr`,
  `precision`, `recall`, `ap`, and `ndcg`, but they are binary-id metrics;
- `RetrieverEvaluator` does not pass qrel grades, scores, entity types, or full
  `NodeWithScore` objects into metrics.

## Target Ownership

Keep custom:

- catalog semantics: tasks, KBs, source instances, roles, adapters, benchmark
  suites;
- KB aliases and Qdrant alias promotion;
- physical collection naming;
- collection attestation validation;
- source/benchmark identity metadata;
- prompt identity and prompt version metadata;
- DB persistence through project eval tables/writer;
- graded-qrel retrieval scoring;
- benchmark case/label contracts.

Move to LlamaIndex:

- primary document/node objects;
- source reader integration where standard readers fit;
- node parsing/chunking;
- Qdrant dense/hybrid indexing and querying;
- retrieval result objects;
- query-engine response synthesis;
- prompt template rendering;
- binary retrieval metrics;
- context/generation evaluators.

Retire during transition:

- legacy `[[sources]]`;
- legacy local `--source <id>`;
- arbitrary source manifest paths;
- `DEFAULT_SOURCE_ADAPTERS`;
- `source_type` as a behavior selector;
- project `SourceDocument`, `ExtractedDocument`, `DocumentSection`, and `Chunk`
  as primary lifecycle contracts;
- custom `QdrantVectorStore` and custom `Retriever` as the main runtime path;
- top-level `src/rag_data_pipelines/`; useful dataset-specific helpers move
  under `rag.adapters.*` or small LlamaIndex reader wrappers;
- `collection_meta` sentinel points.

## Target Project Structure

The target structure should make one boundary obvious: project code owns
catalog, identity, policy, adapters, persistence, and orchestration; LlamaIndex
owns document/node/query/index mechanics wherever it can.

Target source tree:

```text
src/
  app_config/
    catalog/
      schema.py              # tasks, KBs, aliases, adapters, source instances
      source_instances.py    # source-instance resolution and manifest paths
      loader.py              # catalog loading/materialization
    runtime/                 # operator/runtime settings only

  rag/
    adapters/
      sources.py             # built-in source adapter factories
      benchmarks.py          # built-in benchmark adapter factories
      capabilities.py        # source/benchmark capability metadata

    contracts/
      metadata.py            # project metadata keys for LI Document/TextNode
      attestation.py         # CollectionAttestation
      prompts.py             # PromptIdentity and prompt version references

    sources/
      manifests.py           # source-instance manifest loading/validation
      build.py               # source build orchestration
      benchmark_prep.py      # benchmark artifact preparation
      readers.py             # small wrappers around LlamaIndex readers
      nodes.py               # metadata enrichment and node id policy

    indexing/
      collections.py         # physical collection names and alias validation
      llamaindex_qdrant.py   # LlamaIndex Qdrant index creation/loading wrapper
      materialize.py         # build nodes into Qdrant + collection metadata
      promotion.py           # Qdrant alias promotion

    runtime/
      resolver.py            # task/KB/alias resolution
      engines.py             # LlamaIndex query-engine construction
      prompts.py             # prompt-template registry wrapper
      service.py             # RagRuntime public facade

    evaluation/
      models.py              # BenchmarkCase, Qrel, BenchmarkLabel, observations
      adapters.py            # benchmark adapter protocol/factories
      evaluator.py           # ProjectRetrieverEvaluator + generation evaluators
      runner.py              # benchmark execution orchestration

  shared/
    db/
      eval_writer.py         # single DB writer for eval_runs/eval_samples
```

Important placement rules:

- Adapter implementations may live together under `rag/adapters/`, but adapter
  contracts must still advertise capabilities explicitly. A source-only adapter
  exposes source capability; a benchmark-capable adapter exposes both source and
  benchmark capability.
- `src/rag/sources/` remains orchestration around source instances and
  manifests. It should not re-grow project-owned `Document` / `Chunk` lifecycle
  types after LlamaIndex `Document` / `TextNode` become primary.
- `src/rag/indexing/` owns collection naming, Qdrant aliases, collection
  metadata attestation, and the project wrapper around LlamaIndex Qdrant
  storage. It should not contain dataset-specific adapter logic.
- `src/rag/runtime/` owns runtime resolution and query-engine construction. It
  should not know source manifest formats or benchmark dataset schemas.
- `src/rag/evaluation/` owns normalized benchmark contracts and execution.
  It should evolve the existing `models.py` contracts rather than introduce a
  parallel benchmark contract package.
- `src/shared/db/eval_writer.py` is the only place that writes RAG benchmark
  results to `eval_runs` / `eval_samples`; benchmark runners should not define
  private SQL table shapes.

Target artifact tree:

```text
assets/rag_data/
  source_instances/
    <source_instance_id>/
      manifest.toml          # curated, Git-tracked
      raw/                   # server-local cache by default
      extracted/             # transitional; removed when native LI artifacts replace it
      chunks/                # transitional; removed when native LI nodes replace it
      benchmark/
        cases.jsonl          # input cases
        labels.jsonl         # expected outputs/qrels/evidence
        metadata.json        # preparation provenance

  knowledge_bases/
    <kb_id>/
      manifests/             # build/index provenance
      metadata/
        build_runs/          # lifecycle audit records
```

Target docs:

```text
docs/
  architecture/system-design.md       # durable RAG model and boundaries
  operations/rag-operations.md        # operator commands and lifecycle
  analytics/evaluation-results.md     # DB result contract
  planning/rag-llamaindex-transition-plan.md
```

Patterns to avoid:

- no dataset-specific code in `src/rag/runtime/` or `src/rag/indexing/`;
- no separate top-level `src/rag_data_pipelines/` package after migration;
- no new hardcoded default adapter registry replacing catalog declarations;
- no benchmark reports under `reports/<run_id>` as a second result store;
- no arbitrary catalog manifest paths after the migration;
- no new project-native document/chunk hierarchy that competes with
  LlamaIndex objects.

## Target Catalog Contract

The catalog remains the project-owned graph. It should not describe prompt
wording, metric lists, or runtime provider internals.

Example:

```toml
schema_version = 3

[[source_adapters]]
id = "generic.http_html"
version = "1"
description = "Fetches HTTP HTML pages and extracts readable text sections."
factory = "rag.adapters.sources:make_http_html_adapter"

[[benchmark_adapters]]
id = "benchmark.pytorch_qa"
version = "1"
description = "Loads QA examples, reference answers, and expected evidence for PyTorch docs."
factory = "rag.adapters.benchmarks:make_pytorch_qa_benchmark_adapter"

[[tasks]]
id = "code"
description = "Programming help for ML systems and PyTorch implementation questions."
knowledge_bases = ["pytorch_reference"]

[[knowledge_bases]]
id = "pytorch_reference"
description = "PyTorch documentation for coding assistance."
update_strategy = "replace"
default_alias = "champion"
aliases.champion = { top_k = 5, score_threshold = 0.35, retrieval_strategy = "dense", reranker_multiplier = 1 }
aliases.challenger = { top_k = 5, score_threshold = 0.01, retrieval_strategy = "hybrid", reranker = "cross-encoder/ms-marco-MiniLM-L-6-v2", reranker_multiplier = 4 }

# role allowed values: "corpus", "benchmark".
[[source_instances]]
id = "pytorch_reference.docs"
description = "Official PyTorch documentation pages."
role = "corpus"
knowledge_base = "pytorch_reference"
adapter = { id = "generic.http_html", version = "1" }

[[source_instances]]
id = "pytorch_reference.qa_benchmark"
description = "Question-answering benchmark cases and expected evidence for PyTorch documentation."
role = "benchmark"
knowledge_base = "pytorch_reference"
adapter = { id = "benchmark.pytorch_qa", version = "1" }
benchmark = { suites = ["context_quality", "generation_quality"] }
```

Final schema rules:

- no legacy `[[sources]]`;
- no source instance `manifest` field;
- source manifest path is derived from source instance id;
- no catalog-level `contains` or `metrics`;
- benchmark execution always receives an explicit alias;
- task LoRA/model adapter config is named `lora_adapter`, not `adapter`.

## Target Runtime Contracts

### Metadata Keys

Every LlamaIndex `Document` and `TextNode` emitted by project adapters must
carry project identity in metadata.

Required `Document.metadata`:

```text
kb_id
source_instance_id
source_document_id
document_id
title
source_uri
adapter_id
adapter_version
manifest_id or manifest_digest
```

Required `TextNode.metadata`:

```text
kb_id
source_instance_id
source_document_id
document_id
chunk_id
title
source_uri
section_title
section_ordinal
section_level
ordinal
token_count
adapter_id
adapter_version
```

`TextNode.id_` is not the human-readable `chunk_id`. It is the Qdrant point id:

```text
TextNode.id_ = uuid5(PROJECT_QDRANT_POINT_NAMESPACE, chunk_id)
metadata["chunk_id"] = "<source document id>:chunk:<ordinal>"
```

This keeps Qdrant/LlamaIndex happy while preserving readable labels,
citations, logs, and benchmark references.

### Collection Attestation

`CollectionAttestation` remains the compact runtime validation contract:

```text
schema_version
manifest_id
kb_id
collection_name
embedding_model
sparse_encoder
retrieval_capability
chunk_count
created_at
```

Store it in Qdrant collection metadata:

```json
{
  "attestation": {
    "schema_version": 1,
    "manifest_id": "sha256:...",
    "kb_id": "pytorch_reference",
    "collection_name": "rag__pytorch_reference__20260618_181058",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "sparse_encoder": "fastembed/bm25",
    "retrieval_capability": "hybrid",
    "chunk_count": 12345,
    "created_at": "2026-06-18T18:10:58Z"
  }
}
```

Read path for deployed Qdrant:

```text
GET /collections/<collection>
  -> .result.config.metadata.attestation
```

Project manifest artifacts remain the external source of truth. Qdrant
attestation is the fast runtime copy used before promotion and retrieval.

### Prompt Identity

LlamaIndex may render prompts, but project code owns prompt identity.

Required prompt metadata:

```text
prompt_id
prompt_version
prompt_digest
prompt_params
```

Persist prompt identity in inference/eval DB rows before comparing
generation-quality benchmarks.

## Target Source Pipeline

Corpus source flow:

```text
catalog source instance
  -> source adapter wrapper
  -> conventional manifest
  -> LlamaIndex reader or custom adapter reader
  -> llama_index.core.Document[]
  -> LlamaIndex node parser
  -> llama_index.core.schema.TextNode[]
  -> optional native LlamaIndex persistence
  -> VectorStoreIndex + QdrantVectorStore
  -> Qdrant physical collection
  -> Qdrant collection metadata attestation
  -> IndexManifest JSON
  -> custom Qdrant alias promotion
```

Source adapter contract after transition:

```python
class SourceAdapter(Protocol):
    adapter_id: str
    version: str

    def validate_manifest(self, path: Path) -> SourceManifest:
        ...

    def load_documents(
        self,
        *,
        manifest: SourceManifest,
        source_instance: SourceInstanceConfig,
    ) -> list[Document]:
        ...
```

Adapters may delegate to LlamaIndex readers:

- `llama-index-readers-web` for HTTP/HTML;
- `llama-index-readers-file` for PDFs and files;
- `llama-index-readers-papers` for arXiv/paper loading.

Adapters still own project metadata enrichment. Readers do not know project KB
ids, source instance ids, benchmark roles, or manifest digests.

Node parser target:

```python
SentenceSplitter(
    chunk_size=<from runtime/build config>,
    chunk_overlap=<from runtime/build config>,
    id_func=<project deterministic uuid function>,
)
```

The parser must preserve metadata and emit nodes with Qdrant-valid ids.

## Target Materialization Pipeline

Dense collection:

```text
TextNode[]
  -> ProjectEmbedding(BaseEmbedding)
  -> QdrantVectorStore(collection_name=physical_name, dense_config=...)
  -> StorageContext.from_defaults(vector_store=...)
  -> VectorStoreIndex(nodes=nodes, storage_context=..., embed_model=...)
  -> write IndexManifest
  -> update Qdrant collection metadata with attestation
```

Hybrid collection:

```text
TextNode[]
  -> ProjectEmbedding(BaseEmbedding)
  -> QdrantVectorStore(
       collection_name=physical_name,
       enable_hybrid=True,
       dense_vector_name="dense",
       sparse_vector_name="sparse",
       sparse_doc_fn=<project sparse doc encoder>,
       sparse_query_fn=<project sparse query encoder>,
       sparse_config=...
     )
  -> VectorStoreIndex(...)
  -> write IndexManifest
  -> update Qdrant collection metadata with attestation
```

Project wrappers:

```python
class ProjectEmbedding(BaseEmbedding):
    """LlamaIndex embedding adapter over the current embedding service."""

class ProjectSparseEncoder:
    """Adapter exposing sparse_doc_fn and sparse_query_fn callables."""

class LlamaIndexQdrantCollection:
    """Small project wrapper for collection creation, metadata attestation,
    alias promotion support, and physical collection reopen."""
```

Qdrant aliases remain custom:

```text
physical collection: rag__<kb_id>__<timestamp>
alias:               rag__<kb_id>__<alias>
```

Promotion still validates attestation before updating the alias.

## Target Runtime Query Pipeline

Retrieval-only flow:

```text
Gateway/task routing
  -> selected KB + explicit/effective alias
  -> custom alias resolver
  -> physical collection name
  -> read Qdrant collection metadata attestation
  -> validate KB id, collection name, embedding model, vector dimension,
     retrieval capability
  -> reopen QdrantVectorStore by physical collection name
  -> VectorStoreIndex.from_vector_store(...)
  -> index.as_retriever(...)
  -> retrieve NodeWithScore[]
  -> optional project reranker node postprocessor
  -> map NodeWithScore to retrieval observation / source nodes
```

Query-engine flow:

```text
validated alias state
  -> VectorStoreIndex.from_vector_store(...)
  -> index.as_query_engine(
       llm=<OpenAI-compatible vLLM/gateway client>,
       similarity_top_k=<alias top_k or expanded rerank top_k>,
       vector_store_query_mode=<dense/hybrid>,
       node_postprocessors=[score threshold, reranker, provenance]
     )
  -> Response
  -> source_nodes -> citations/provenance
  -> persist prompt identity and RAG observations
```

Chat engines are deferred. The first migration uses query engines because
Gateway/session memory remains a project boundary.

Runtime result mapping:

```text
NodeWithScore.node.node_id      -> point_id
NodeWithScore.node.metadata     -> project ids and source metadata
NodeWithScore.score             -> retrieval score
metadata["chunk_id"]            -> cited chunk id
metadata["document_id"]         -> cited document id
metadata["source_instance_id"]  -> source provenance
metadata["source_uri"]          -> citation URL
```

`RetrievalHit` can be retired as a primary runtime contract after the gateway
and DB observation code consume LlamaIndex response/source-node data directly.
Until then, keep a mapper for compatibility.

## Target Benchmark Pipeline

Benchmark source instances remain attached to exactly one KB. They inherit
runtime/build parameters from the attached KB alias at benchmark execution
time. The operator must pass the alias explicitly.

Preparation flow:

```text
benchmark source instance
  -> benchmark-capable adapter
  -> LlamaIndex Document[] / TextNode[] for benchmark corpus when applicable
  -> BenchmarkCase[] input artifact
  -> BenchmarkLabel[] expected-output artifact
```

Execution flow:

```text
run-benchmark --source-instance <benchmark-id> --alias <alias>
  -> resolve attached KB from source_instance.knowledge_base
  -> read alias profile and parameter-source collection attestation
  -> build temporary benchmark collection with same LlamaIndex build profile
  -> run retrieval/query engine
  -> ProjectRetrieverEvaluator for retrieval_quality
  -> LlamaIndex evaluators for context_quality / generation_quality
  -> persist eval_runs and eval_samples.detail
  -> delete temporary benchmark collection
```

Retrieval-quality evaluation:

```python
class ProjectRetrieverEvaluator(RetrieverEvaluator):
    """Reuse LlamaIndex retriever and postprocessors, but evaluate project labels."""

    async def aevaluate_project(
        self,
        *,
        case: BenchmarkCase,
        label: BenchmarkLabel,
    ) -> ProjectRetrievalEvalResult:
        ...
```

Responsibilities:

- reuse LlamaIndex retriever;
- reuse LlamaIndex node postprocessors;
- run built-in binary metrics where applicable;
- pass `qrels[]`, relevance grades, entity types, scores, metadata, and
  `NodeWithScore[]` to project graded scorers;
- produce `RetrievalEvalObservation`.

Context/generation evaluation:

- `context_quality` maps to LlamaIndex context/relevancy evaluators;
- `generation_quality` maps to correctness, semantic similarity, faithfulness,
  answer relevancy, and guideline evaluators;
- judge LLM can use LlamaIndex OpenAI-compatible client pointed at the
  vLLM/gateway path;
- prompt identity is persisted with every run/sample.

## Target Artifacts

Keep project-owned artifacts:

```text
assets/rag_data/
  source_instances/
    <source_instance_id>/
      manifest.toml
      benchmark/
        cases.jsonl
        labels.jsonl
        metadata.json

  knowledge_bases/
    <kb_id>/
      manifests/
        <collection_name>.json
      metadata/
        build_runs/
```

Move document/node/index artifacts to native LlamaIndex persistence where
needed. Keep `raw/` only as a source cache; do not preserve project
`extracted/` and `chunks/` JSON as a permanent API unless a concrete
operational need appears.

Suggested LlamaIndex artifact area:

```text
assets/rag_data/
  source_instances/
    <source_instance_id>/
      llamaindex/
        documents/
        nodes/

  knowledge_bases/
    <kb_id>/
      llamaindex/
        <collection_name>/
          storage/
```

Artifact principle:

- catalog manifests and benchmark cases/labels are project artifacts;
- LlamaIndex documents/nodes/index state are LlamaIndex artifacts;
- Qdrant is the runtime vector-store source;
- Postgres is the benchmark result source of truth.

## Implementation Plan

### Phase 0: Freeze Legacy Cleanup As Prerequisite

Status: complete.

Goal: avoid building the LlamaIndex path on compatibility branches that are
already scheduled for removal.

Tasks:

- Convert checked-in `catalog.toml` to `schema_version = 3`.
- Move checked-in source manifests to
  `assets/rag_data/source_instances/<source_instance_id>/manifest.toml`.
- Remove legacy `[[sources]]` support from schema/index/materialization.
- Remove legacy local `--source <id>` CLI path.
- Remove arbitrary source manifest paths.
- Remove `DEFAULT_SOURCE_ADAPTERS`.
- Rename task-level model adapter config to `lora_adapter`.
- Stop using `source_type` as adapter behavior selector.

Acceptance:

- New catalog schema is the only supported schema.
- `build-source` and `prepare-benchmark` address source instances by global id.
- No LlamaIndex migration code has to branch on legacy source declarations.

### Phase 1: Dependencies And Provider Adapters

Status: complete.

Goal: install only the LlamaIndex packages needed in the images that need them.

Tasks:

- Add `llama-index-vector-stores-qdrant` to RAG build/runtime images.
- Add reader packages to build/source-preparation images only.
- Add `llama-index-llms-openai` where query/eval uses the OpenAI-compatible
  vLLM/gateway path.
- Implement `ProjectEmbedding(BaseEmbedding)` over the current embedding
  service.
- Implement sparse encoder callables for LlamaIndex Qdrant hybrid mode.
- Implement project reranker node postprocessor over the current reranker
  service.

Acceptance:

- Unit tests instantiate embedding/reranker wrappers without network calls.
- Integration smoke tests can create a LlamaIndex Qdrant vector store with
  dense and hybrid configs.

### Phase 2: Source Adapter And Object Model Migration

Status: complete.

Goal: make LlamaIndex `Document` and `TextNode` primary.

Tasks:

- Replace project `SourceDocument`, `ExtractedDocument`, `DocumentSection`, and
  `Chunk` usage in new source pipeline with LlamaIndex objects.
- Move built-in adapter factories toward `rag.adapters.sources` and
  `rag.adapters.benchmarks`; keep adapter protocols/capabilities explicit.
- Move the PyTorch redirect/placeholder helper to
  `rag.adapters.pytorch_docs`; remove the top-level `rag_data_pipelines`
  package.
- Update source adapters to emit `Document[]`.
- Add metadata enrichment helpers for required project keys.
- Add deterministic UUID node id helper:
  `TextNode.id_ = uuid5(namespace, chunk_id)`.
- Persist LlamaIndex `Document` and `TextNode` objects directly inside the
  transitional extraction/node artifact envelopes.
- Feed `SourceNodeBundle` into the existing materializer until Phase 3 replaces
  the collection writer; do not map nodes back into project `Chunk` objects.

Acceptance:

- Source build for a corpus source produces LlamaIndex documents/nodes with all
  required metadata.
- Human-readable `chunk_id` remains stable in metadata.
- Project `Document` / `Chunk` are no longer required by the new build path.

### Phase 3: LlamaIndex Qdrant Materialization

Status: complete.

Goal: replace custom Qdrant writes with LlamaIndex vector-store indexing.

Tasks:

- Implement `materialize_kb_collection_llamaindex()`.
- Use `QdrantVectorStore` with caller-provided physical collection name.
- Support dense and hybrid profiles.
- Write `IndexManifest` JSON as project provenance.
- Write `CollectionAttestation` into Qdrant collection metadata.
- Read attestation from `.result.config.metadata.attestation`.
- Keep alias promotion custom.
- Remove `collection_meta` sentinel writes from the new path.

Acceptance:

- Dense collection builds and retrieves through LlamaIndex.
- Hybrid collection builds and retrieves through LlamaIndex.
- Qdrant collection metadata contains attestation.
- Promotion validates collection metadata before alias update.
- No sentinel point appears in LlamaIndex-built collections.

### Phase 4: Runtime Retrieval And Query Engine

Goal: replace custom runtime retriever with LlamaIndex retriever/query engine
behind the existing catalog/alias boundary.

Tasks:

- Implement LlamaIndex runtime resolver that:
  - resolves KB alias to physical collection;
  - reads collection metadata attestation;
  - validates KB id, collection name, embedding model/dimension, retrieval
    capability;
  - reopens `VectorStoreIndex.from_vector_store()`.
- Implement retrieval adapter returning `NodeWithScore[]` and project
  observations.
- Implement query-engine adapter with project prompt wrapper.
- Persist prompt identity in runtime/eval observations.
- Keep chat engines out of scope.

Acceptance:

- Runtime can retrieve from champion/challenger aliases with LlamaIndex.
- Query engine returns answer plus source nodes.
- Citations/provenance map from source node metadata.
- Existing gateway behavior can be preserved through a compatibility mapper.

### Phase 5: Benchmark Execution And Evaluation

Goal: run RAG benchmarks through LlamaIndex retrieval/query/evaluator
abstractions while keeping project DB persistence.

Tasks:

- Implement `ProjectRetrieverEvaluator(RetrieverEvaluator)`.
- Use LlamaIndex built-in metrics for binary retrieval metrics.
- Implement project graded qrel scorers for relevance grades and
  document/chunk entity types.
- Wire LlamaIndex context/generation evaluators.
- Centralize DB writes through `src/shared/db/eval_writer.py`.
- Persist retrieval observations, generation observations, prompt identity,
  and benchmark artifact digests.

Acceptance:

- Retrieval-quality benchmark supports binary and graded qrels.
- Context/generation-quality benchmarks use LlamaIndex evaluator inputs.
- Every benchmark run requires explicit alias.
- Results are persisted to Postgres, not report files.

### Phase 6: Remove Old RAG Mechanics

Goal: finish the transition by deleting old runtime/build contracts and
compatibility mappers.

Tasks:

- Remove custom `QdrantVectorStore` from the main RAG path.
- Remove custom `Retriever` from the main RAG path.
- Remove project `SourceDocument`, `ExtractedDocument`, `DocumentSection`, and
  `Chunk` from active source/build contracts.
- Remove old raw/extracted/chunks artifact writers unless retained for a
  specific operational reason.
- Remove `collection_meta` sentinel read/write logic after all live collections
  have either been rebuilt or migrated.
- Update operations docs and Airflow DAGs.

Acceptance:

- New LlamaIndex path is the only supported RAG build/runtime path.
- Live collections use Qdrant collection metadata attestation.
- Tests no longer depend on legacy source schema or old document/chunk
  contracts.

## Residual Risks

- Native LlamaIndex persistence may not preserve every historical artifact
  detail. This is acceptable unless an operational use case proves otherwise.
- Chat engine adoption remains deliberately out of scope.
