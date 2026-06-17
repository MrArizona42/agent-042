# RAG Pipeline Isolation Plan

This plan should happen before the RAG experiment series. The project is meant
to be domain-free, so RAG lifecycle code must be separated from corpus-specific
data pipelines before benchmark and KB experiments multiply.

The goal is not to remove all domain-specific code. The goal is to put it behind
stable contracts so each corpus can preserve its natural metadata without
turning the production RAG engine into a PyTorch-, arXiv-, QASPER-, or
BEIR-shaped implementation.

## Target Structure

```text
src/
  app_config/
    catalog/
      application catalog schema, loading, and validation
    runtime/
      runtime config schema and loading
  rag/
    contracts/
      stable RAG data contracts and serialization models
    ingest/
      source adapter interfaces
      generic manifest envelope
      fetch/extract/chunk artifact lifecycle
      generic adapters for common source formats
    indexing/
      materialization build profiles
      Qdrant collection materialization
      collection manifests and alias promotion
    evaluation/
      eval runner, metrics, persistence, and promotion rules
    runtime/
      runtime retrieval contracts and services
  rag_data_pipelines/
    qasper/
    open_ragbench/
    beir/
    msmarco/
    pytorch_docs/
    arxiv_papers/
  shared/
    db, events, logging, telemetry, and other cross-cutting infrastructure

pipelines/
  rag/
    optional thin operator-facing wrappers or config bundles only

experiments/
  rag/
    notebooks
    reports
    failure analysis
    exploratory variants
```

## Boundary Rules

- `src/` owns production platform code:
  - RAG contracts such as `SourceDocument`, `ExtractedDocument`, `Chunk`, and
    `IndexManifest`;
  - source adapter interfaces and generic ingest lifecycle;
  - evaluation contracts such as normalized eval rows, qrels, evidence refs, and
    result records;
  - generic fetch/extract/chunk lifecycle;
  - Qdrant materialization, collection manifests, and alias promotion;
  - Qdrant collection and alias management;
  - runtime retrieval;
  - generic source adapters such as local files, HTTP HTML, HTTP PDF, JSONL, and
    possibly Hugging Face dataset loading.
- `src/app_config/` owns application-level config:
  - catalog schema, loading, and validation;
  - task, KB, source-instance, alias, and adapter-routing declarations;
  - runtime config schema and loading.
- `src/rag_data_pipelines/` owns production dataset pipeline code:
  - QASPER normalization;
  - Open RAG Benchmark import;
  - BEIR/MS MARCO import;
  - PyTorch docs source-list generation;
  - arXiv paper source-list generation;
  - mapping raw dataset fields to platform source documents and eval rows;
  - corpus-specific filtering, version pinning, and metadata preservation.
  The explicit name is intentional: this package owns RAG data preparation
  pipelines, not runtime RAG services and not generic pipeline machinery.
- `pipelines/rag/`, if it exists, owns only thin operator-facing wrappers,
  checked-in pipeline configs, or shell conveniences. It should not contain the
  production implementation when `src/rag_data_pipelines/` exists.
- `experiments/` owns non-production exploration:
  - notebooks;
  - plots and reports;
  - failure analysis;
  - ad hoc inspection;
  - parser or prompt variants before they are promoted into production code.

Production KB builds should not depend on notebooks or scratch scripts in
`experiments/`.

`src/shared/` should be kept for cross-cutting infrastructure only. Catalog and
runtime configuration should move to `src/app_config/` so `shared` does not
become the place where unrelated application concepts accumulate.

## Adding A New Collection

The operator-facing setup for a new collection should be small and explicit.
Most new collections should not require edits to generic ingest, indexing,
runtime, or gateway code.

Required operator inputs:

1. Add or update catalog entries.

   Edit `catalog.toml` to declare:

   - the knowledge base if it is new;
   - its aliases and retrieval profiles;
   - the task-to-KB routing relationship when runtime auto-selection should use
     it;
   - one or more source instances, each with `kb`, source instance `id`,
     current source `type`, manifest/config ref, ingest adapter id/version, and
     adapter settings.

   This is application configuration and should eventually be loaded through
   `src/app_config/catalog`.

   The current catalog shape does not conflict with this. It already uses
   `[[sources]]` entries shaped around `type`, `kb`, `id`, and `manifest`; the
   ingest adapter link can be an additive source-level field. Avoid overloading
   the existing task-level `adapter` field, which currently means LoRA/task
   routing rather than source ingestion.

   Preferred migration shape:

   ```toml
   [[sources]]
   type = "html_docs"                     # Existing source metadata; retained for compatibility.
   kb = "pytorch_reference"               # Existing owning KB id.
   id = "docs"                            # Existing KB-local source instance id.
   manifest = "assets/rag_data/pytorch_reference/sources.toml" # Existing manifest ref.
   ingest_adapter = { id = "generic.http_html", version = "1" } # New adapter contract link.
   ```

   `type` can remain useful for payload metadata, filtering, and transitional
   compatibility, but `ingest_adapter.id` should become the extension mechanism
   that chooses validation/list/fetch/extract behavior.

2. Add a source manifest or dataset pipeline config.

   Add a manifest/config file under the RAG data area or another agreed config
   location. It should describe the source rows for the selected adapter:

   - for generic HTTP HTML/PDF/local-file adapters, this may be a curated source
     list;
   - for dataset-backed collections, this may be a dataset config with dataset
     name, revision, split, subset, filters, and output refs.

3. Choose or add an adapter.

   - If an existing generic adapter can validate/list/fetch/extract the source,
     only catalog and manifest/config edits are needed.
   - If the source is a new dataset or corpus shape, add a production adapter or
     prepare module under `src/rag_data_pipelines/<dataset>/`.
   - The adapter should emit platform contracts such as `SourceDocument` and,
     when labels exist, normalized eval rows/qrels. It should not require edits
     to generic indexing or runtime retrieval.

4. Run the generic lifecycle.

   Once catalog, manifest/config, and adapter are in place, the existing generic
   commands should handle:

   ```text
   validate/list sources
     -> fetch/resolve raw artifacts
     -> extract text
     -> chunk
     -> collect bundles
     -> materialize collection
     -> write manifest/attestation
     -> promote alias when approved
   ```

Changes that should not be needed for an ordinary new collection:

- editing a closed `SourceType` enum;
- editing generic manifest unions for every dataset;
- adding dataset branches to generic fetch/extract/chunk/materialize code;
- changing runtime retrieval or gateway request handling;
- changing Qdrant alias promotion logic.

Generic platform changes are only expected when the new collection needs a
genuinely new capability, such as a new raw artifact class, a new extraction
family, a new chunking strategy, new vector-store behavior, or a new evaluation
contract field.

## One-Button Build Contract

The operator experience should be one action to start a build and wait for a
clear result, but the implementation should stay transparent and staged. CLI
and Airflow should share the same lifecycle request models and command/stage
functions; they should differ only in how parameters are supplied and displayed:

- CLI parses command parameters and can expose `--wait`, `status`, and
  `promote` commands.
- Airflow exposes the same request fields as DAG Params and shows the same
  lifecycle as retryable tasks.

The shared request object should include:

```python
BuildRequest(
    kb_id="pytorch_reference",            # Logical KB to build.
    source_ids=["docs"],                  # Source subset, or all catalog sources for the KB.
    alias_config="challenger",            # Retrieval profile / target alias role to validate against.
    catalog_path="catalog.toml",          # Catalog snapshot used for setup resolution.
    rag_data_root="assets/rag_data",      # Root for versioned and restorable RAG artifacts.
    force=False,                          # Whether to bypass reusable artifacts or previous runs.
    dry_run=False,                        # Whether to only validate and plan.
)
```

The build should create a persisted `BuildRun`:

```python
BuildRun(
    run_id="rag_build_20260616_120000",   # Stable id joining logs, artifacts, metrics, and reports.
    kb_id="pytorch_reference",            # KB being built.
    source_ids=["docs"],                  # Source instances included in this run.
    status="running",                     # planned/running/failed/succeeded/promoted/rolled_back.
    build_profile_digest="sha256:...",    # Digest of materialization and retrieval build settings.
    catalog_digest="sha256:...",          # Exact catalog content used by this run.
    manifest_digests={"docs": "sha256:..."}, # Exact source manifests/configs used.
    adapter_versions={"docs": "generic.http_html@1"}, # Adapter contracts used.
    collection_name=None,                 # Filled after materialization succeeds.
    report_ref=None,                      # Filled when validation/eval report is written.
)
```

Recommended lifecycle:

```text
preflight catalog/manifests/adapters
  -> create BuildRun
  -> resolve source snapshot
  -> fetch or restore raw artifacts
  -> extract
  -> chunk
  -> collect bundles
  -> materialize physical collection
  -> write manifest and attestation
  -> run smoke/eval gates
  -> write build report
  -> promote only when explicitly approved or policy-gated
```

Promotion should remain separate from build by default. The one-button flow can
build a candidate and wait for the report; automatic promotion should be a later
policy once quality gates, rollback, and monitoring are trusted.

## Versioning, Restoration, And Corruption Handling

The system should be clear about which state is declarative, which state is
versioned, which state is external, and which state can be rebuilt.

### State Classes

- Declarative setup: `catalog.toml`, source manifests/configs, build profiles,
  and adapter code. This is Git-versioned and restored by checking out the
  desired commit.
- Large source/build artifacts: downloaded raw files, extracted artifacts, chunk
  artifacts, normalized benchmark rows, build reports, and collection manifests.
  These are DVC-versioned, or stored in an equivalent artifact store, when they
  are too large or too generated for Git. Restore them with `dvc pull` for the
  matching Git revision.
- Rebuildable external state: Qdrant physical collections, Qdrant aliases, and
  dense/sparse indexes. These are runtime state, not the source of truth.
  Recreate them from Git/DVC artifacts and collection manifests.
- Ephemeral runtime state: CLI process state, Airflow task logs, in-memory
  adapter objects, and retriever caches. These are not versioned and should be
  recreated by rerun or service restart.
- Observability state: inference events, eval rows, metrics, `BuildRun` status,
  and failure summaries. These live in DB/analytics storage with operational
  retention; important reports can also be DVC-versioned. Treat them as audit
  and diagnosis state, not canonical build input.

### Creating A New Collection

1. Commit the declarative setup to Git:
   - `catalog.toml` source entry;
   - source manifest or dataset pipeline config;
   - adapter registration/code when a new adapter is needed;
   - build profile changes when retrieval/materialization settings change.
2. Run a dry plan.
   - Validate catalog schema;
   - resolve source instances and ingest adapters;
   - validate manifests/configs;
   - estimate document count, chunk count, embedding cost, and target collection
     name.
3. Start a build run through CLI or Airflow.
   - CLI and Airflow call the same lifecycle commands;
   - Airflow exposes each lifecycle stage as a task;
   - the build writes artifacts, manifest, attestation, and report.
4. Version large outputs.
   - Push raw/extracted/chunk/build-report artifacts to DVC when they should be
     reproducible without refetching from original sources;
   - keep the lightweight pointers and setup in Git.
5. Promote only after review or gates.
   - Alias promotion should point runtime traffic at the approved physical
     collection;
   - previous aliases/collections should remain available for rollback.

### Restoring A Version

This is a single-repo project: application code, infrastructure, manifests, DVC
pointers, and docs all share one Git history. Therefore, restoring an older RAG
artifact version must not casually move the whole working tree to an old commit.
That could also downgrade runtime code, deployment config, migrations, or
operator docs.

Prefer path-scoped or isolated DVC restore workflows.

To inspect or rebuild from an old artifact version without touching the current
branch:

```text
git worktree add ../agent-042-rag-restore <old-commit-or-tag>
  -> cd ../agent-042-rag-restore
  -> dvc pull assets/rag_data/<dataset-or-kb>.dvc
  -> validate collection manifest and artifact checksums
  -> recreate Qdrant physical collection from restored chunk artifacts
  -> write/verify attestation
  -> repoint alias from the current runtime environment if this version should serve traffic
```

The temporary worktree isolates old Git-tracked files and DVC pointers from the
active development branch. It is the safest default for incident response and
for comparing old RAG artifacts against the current runtime.

To fetch an old artifact payload into a scratch directory while keeping the
current worktree untouched:

```text
dvc get . assets/rag_data/<dataset-or-kb> --rev <old-commit-or-tag> -o /tmp/rag_restore/<dataset-or-kb>
  -> validate manifest/checksums from the scratch directory
  -> rebuild or inspect without changing tracked files
```

This is useful for read-only inspection or manual recovery. It should not be
treated as a catalog/config change, because the current Git branch still points
at its existing DVC metadata.

To intentionally restore an old DVC pointer onto the current branch:

```text
git switch develop
  -> git switch -c data/rag-restore/<kb-or-dataset>/<reason>
  -> git restore --source <old-commit-or-tag> -- assets/rag_data/<dataset-or-kb>.dvc
  -> dvc pull assets/rag_data/<dataset-or-kb>.dvc
  -> validate restored artifacts with the current runtime/build code
  -> commit only the path-scoped DVC pointer and any required manifest/catalog changes
  -> open a PR back to develop
```

Use this only when the older data pointer should become the new canonical
version for the active branch. Do not use a whole-repo `git checkout
<old-commit>` or revert commit unless the intent is to roll back code,
infrastructure, and data together.

To restore from the artifact store as part of a normal rebuild:

```text
git checkout <current-branch-or-release-commit>
  -> dvc pull
  -> validate collection manifest and artifact checksums
  -> recreate Qdrant physical collection from chunk artifacts
  -> write/verify attestation
  -> repoint alias if this version should serve traffic
```

To restore from original sources:

```text
git checkout <current-branch-or-release-commit>
  -> run build with the same adapter versions and build profile
  -> refetch/resolve raw sources
  -> regenerate extraction and chunk artifacts
  -> materialize a new physical collection
  -> compare manifest/report against the previous known-good version
```

The artifact-store path is preferred when exact reproducibility matters. The
original-source path is useful when artifacts were intentionally not retained or
when the source should be refreshed.

### Branch And DVC Policy

Current branch policy already points in a useful direction:

- feature and planning work happens on branches targeting `develop`;
- CI runs for PRs into `develop` and `main`;
- image builds are tied to `main`;
- deployment validates that the requested SHA belongs to the selected branch;
- `rag_lifecycle` already has `dvc_base_branch` and `dvc_bot_branch` controls
  for DVC sync PRs.

Keep that shape, but make RAG data changes more explicit:

- DVC sync from Airflow should create a data-only bot branch such as
  `data/rag/<kb>/<run_id>` from the configured base branch.
- The bot branch should commit only DVC pointer changes, generated artifact
  metadata that belongs in Git, and the build report pointer when needed.
- Runtime code, infra, catalog routing, and adapter code changes should happen
  in normal feature branches, not in the bot DVC sync branch.
- If a RAG build requires both code changes and new DVC artifacts, merge the
  code/config PR first, then run the build from that commit, then open the
  data-only DVC PR produced by the build.
- Build reports should record both the code/config Git SHA that produced the
  artifacts and the DVC pointer commit/PR that made the artifacts discoverable.
- Production alias promotion should reference a `BuildRun` and collection
  manifest, not just "whatever DVC files are currently checked out".

This avoids treating Git history as a single rollback lever. Most incidents
should restore or repoint only the RAG artifacts/aliases involved; whole-repo
rollback should be reserved for cases where the runtime or infrastructure change
is part of the failure.

### Corruption Response

If a generated artifact is corrupted:

- verify checksums from the collection manifest and chunk/extraction artifacts;
- delete only the corrupted generated file or failed stage output;
- restore it from DVC when available;
- otherwise rerun the earliest affected stage from its versioned inputs.

If a Qdrant physical collection is corrupted:

- treat Qdrant as rebuildable, not canonical;
- stop or avoid promotion to the corrupted collection;
- recreate the physical collection from DVC-restored chunk artifacts and the
  Git-versioned build profile;
- verify attestation before moving any alias.

If an alias points to the wrong collection:

- use the previous known-good collection manifest and attestation to identify
  the correct target;
- repoint the alias atomically;
- record the rollback in the `BuildRun` or operational audit log.

If the catalog or manifest is wrong:

- fix it in Git;
- create a new build run rather than mutating old build artifacts in place;
- keep the broken run/report for auditability when it has already been used for
  evaluation or promotion.

## Data Flow

The target data flow should make each boundary explicit:

```text
Catalog source config / dataset config
  -> source selector or dataset prepare entrypoint
  -> selected source instance(s) and source manifest refs
  -> SourceAdapter.list_documents()
  -> SourceDocument[]
  -> Fetcher or ArtifactResolver
  -> RawArtifactRef and raw cache files
  -> Extractor
  -> ExtractedDocument
  -> extraction artifact writer
  -> ExtractedDocumentArtifact[]
  -> Chunker
  -> Chunk[]
  -> chunk artifact writer
  -> ChunkArtifact[]
  -> bundle collector
  -> SourceChunkBundle[]
  -> materializer
  -> Qdrant physical collection + IndexManifest
  -> alias promoter
  -> Qdrant alias
  -> runtime retriever
  -> RetrievalHit[]
  -> prompt builder
  -> model answer, citations, and observability
```

The benchmark/evaluation flow runs beside the build flow:

```text
Dataset labels / benchmark files
  -> DatasetEvalAdapter
  -> NormalizedEvalRow[] and qrels/evidence refs
  -> evaluation runner
  -> runtime retrieval and generation
  -> EvalResult[]
  -> metrics, reports, and promotion decision
```

Core lifecycle code should only depend on the generic data objects after each
adapter boundary. Dataset-specific code can preserve natural metadata, but it
must emit platform contracts before the generic lifecycle takes over.

### Example Notation

The walkthrough uses different formats on purpose. Not every data object is a
JSON document.

- **TOML file**: operator-authored catalog, manifest, or pipeline config.
- **Python model**: in-memory Pydantic/dataclass-style project contract, shown as
  constructor syntax for clarity.
- **JSON artifact**: persisted lifecycle artifact written under the RAG data
  root.
- **External state**: Qdrant collection/alias, raw cache file, or analytics row.
- **Processing block**: function, class, module, or service that transforms the
  input object into the output object.

Each hop follows this shape:

```text
Input object -> Processing block -> Output object
Code/data home
Operator action / automatic now / future automation
Small typed example
```

### Example Build Walkthrough

This example uses one PyTorch docs page because it is small, but the same shape
should work for QASPER papers, BEIR documents, arXiv PDFs, or private corpora.

1. Operator selects a source.

   Input object: `catalog.toml` source entry plus CLI/Airflow params.
   Processing block: `app_config.catalog.load_catalog()` plus RAG source selector.
   Output object: `SourceInstanceSelection` in memory.
   Code/data home: catalog schema belongs in `src/app_config/catalog`; source
   selection helpers belong in `src/rag/ingest`; dataset-specific defaults belong
   in catalog or `src/rag_data_pipelines/*`.
   Operator action: choose `kb`, source subset, catalog path, and data root, or
   use catalog defaults.
   Automatic now: catalog parsing and source lookup.
   Future automation: scheduled builds can select all eligible source instances
   from the catalog.

   Input file: `catalog.toml` (TOML file).

   ```toml
   [[sources]]
   type = "html_docs"                    # Existing source metadata; retained for compatibility.
   kb = "pytorch_reference"              # Logical KB that will own the build.
   id = "docs"                           # KB-local source instance id.
   manifest = "assets/rag_data/pytorch_reference/sources.toml" # Source manifest ref.
   ingest_adapter = { id = "generic.http_html", version = "1" } # Adapter used to validate/list/fetch.
   ```

   Output object: `SourceInstanceSelection` (Python model, in memory).

   ```python
   SourceInstanceSelection(
       kb_id="pytorch_reference",          # Logical KB selected for the build.
       source_instance_id="docs",          # Source instance selected within the KB.
       source_type="html_docs",            # Existing source metadata for payloads/debugging.
       adapter_id="generic.http_html",     # Adapter family selected from catalog.
       adapter_version="1",                # Adapter version selected from catalog.
       manifest_ref="assets/rag_data/pytorch_reference/sources.toml",  # Manifest to load.
   )
   ```

2. Adapter validates the manifest.

   Input object: source manifest envelope.
   Processing block: `SourceAdapter.validate_manifest()`.
   Output object: adapter-owned validated manifest payload.
   Code/data home: generic envelope loading belongs in `src/rag/ingest`; adapter
   validation belongs in generic source adapters or `src/rag_data_pipelines/*`.
   Operator action: fix the manifest or adapter settings if validation fails.
   Automatic now: validation runs before build/prepare work.
   Future automation: CI or Airflow preflight can validate manifests before
   fetch/extract work starts.

   Input file: `sources.toml` (TOML file).

   ```toml
   schema_version = 1                      # Manifest schema version.
   source_type = "html_docs"               # Source metadata expected by this manifest.

   [[documents]]
   id = "tensors"                         # Manifest-local document id.
   url = "https://pytorch.org/docs/stable/tensors.html" # Existing fetch target field.
   title = "Tensors"                      # Human-readable title for citations.
   ```

   Output object: `ValidatedManifest` (Python model, in memory).

   ```python
   ValidatedManifest(
       adapter_id="generic.http_html",     # Adapter that validated the payload.
       adapter_version="1",                # Adapter contract version.
       documents=[                         # Adapter-owned normalized manifest rows.
           {
               "id": "tensors",            # Manifest-local document id.
               "title": "Tensors",         # Human-readable title.
               "uri": "https://pytorch.org/docs/stable/tensors.html", # Normalized fetch target.
           }
       ],
   )
   ```

3. Adapter lists platform documents.

   Input object: `ValidatedManifest`.
   Processing block: `SourceAdapter.list_documents()`.
   Output object: `list[SourceDocument]`.
   Code/data home: `SourceDocument` belongs in `src/rag/contracts`; adapter
   mapping belongs with the adapter; source-specific fields live in `metadata`.
   Operator action: optionally review document counts, ids, and titles before a
   full build.
   Automatic now: lifecycle calls document listing.
   Future automation: source-list diffs can flag added, removed, or renamed
   documents.

   Output object: `SourceDocument` (Python model, in memory).

   ```python
   SourceDocument(
       id="html:tensors",                  # Stable namespaced platform document id.
       source_type="html_docs",            # Metadata for filtering/debugging, not extension.
       uri="https://pytorch.org/docs/stable/tensors.html", # Canonical fetch target.
       title="Tensors",                    # Display/citation title.
       metadata={"page_id": "tensors"},    # Source-specific id preserved for joins/debugging.
   )
   ```

4. Raw content is fetched or resolved.

   Input object: `SourceDocument`.
   Processing block: `SourceFetcher.fetch()` or `ArtifactResolver.resolve()`.
   Output object: `RawArtifactRef` plus raw cache file.
   Code/data home: generic fetch/cache mechanics belong in `src/rag/ingest`;
   source-specific URL/path resolution belongs in adapters; raw bytes live under
   the RAG data root and are not contract models.
   Operator action: choose cache invalidation flags such as `force_fetch` when a
   source must be refreshed.
   Automatic now: fetchers resolve and cache raw artifacts.
   Future automation: incremental policies can refresh only changed documents.

   Output object: `RawArtifactRef` (Python model, in memory).

   ```python
   RawArtifactRef(
       path="assets/rag_data/pytorch_reference/raw/docs/html_tensors/page.html", # Cached raw file.
       checksum="sha256:...",              # Byte-level identity for reproducibility.
       content_type="text/html",           # Content type used to validate extractor choice.
   )
   ```

   External state: raw cache file.

   ```text
   assets/rag_data/pytorch_reference/raw/docs/html_tensors/page.html
   # Immutable cached HTML bytes used as extraction input.
   ```

5. Text is extracted.

   Input object: `RawArtifactRef` plus `SourceDocument`.
   Processing block: `SourceExtractor.extract()`.
   Output object: `ExtractedDocument`.
   Code/data home: `ExtractedDocument` and `DocumentSection` belong in
   `src/rag/contracts`; extractor implementations live in generic adapters or
   `src/rag_data_pipelines/*` when corpus-specific.
   Operator action: choose extractor/parser settings through the build profile
   when defaults are not appropriate.
   Automatic now: selected extractor runs during build.
   Future automation: eval-backed parser selection can choose extractor settings
   per corpus type.

   Output object: `ExtractedDocument` (Python model, in memory).

   ```python
   ExtractedDocument(
       id="html:tensors",                  # Extracted document id.
       source_document_id="html:tensors",  # Link back to SourceDocument.
       text="A torch.Tensor is a multi-dimensional matrix...", # Full-text fallback.
       sections=[                          # Structured sections improve chunking/citations.
           DocumentSection(
               title="Tensors",            # Section heading.
               text="A torch.Tensor is a multi-dimensional matrix...", # Section text.
               level=1,                    # Heading depth or structural level.
               ordinal=0,                  # Stable order inside the document.
               metadata={},                # Extractor-specific section metadata.
           )
       ],
       extraction_method="html_bs4",       # Extractor/method id for provenance.
   )
   ```

6. Extraction is persisted.

   Input object: `SourceDocument` plus `RawArtifactRef` plus `ExtractedDocument`.
   Processing block: `extracted_artifact_from_result()` and
   `write_extracted_artifact()`.
   Output object: `ExtractedDocumentArtifact` persisted as JSON.
   Code/data home: artifact wrappers and path conventions belong in
   `src/rag/ingest`; extracted text contracts stay in `src/rag/contracts`;
   generated artifacts live under the RAG data root.
   Operator action: inspect failed extraction summaries and decide retry/skip/fix.
   Automatic now: successful extraction artifacts are written by the lifecycle.
   Future automation: quality gates can block materialization on high failure or
   empty-text rates.

   Output file: extracted artifact (JSON artifact).

   ```jsonc
   {
     "schema_version": 1,                 // Artifact schema version.
     "kb_id": "pytorch_reference",        // Owning KB for artifact layout.
     "source_instance_id": "docs",        // Source instance that produced it.
     "source_type": "html_docs",          // Source metadata copied for filtering/debugging.
     "raw": {                             // Raw bytes that were extracted.
       "path": "assets/rag_data/pytorch_reference/raw/docs/html_tensors/page.html", // Raw input file.
       "checksum": "sha256:...",          // Raw input checksum.
       "content_type": "text/html"        // Raw input content type.
     },
     "document": {                        // ExtractedDocument payload, abbreviated.
       "id": "html:tensors",              // Extracted document id.
       "source_document_id": "html:tensors" // Link back to SourceDocument.
     }
   }
   ```

7. Extracted text is chunked.

   Input object: `ExtractedDocumentArtifact` plus `ChunkingConfig`.
   Processing block: `chunk_extracted_artifact()` or `chunk_source_instance()`.
   Output object: `list[Chunk]`.
   Code/data home: `Chunk` belongs in `src/rag/contracts`; chunking config and
   generic chunker implementations belong in `src/rag/ingest`; corpus policies
   can be selected by build profiles or data pipeline adapters.
   Operator action: choose chunking method, size, and overlap.
   Automatic now: chunking runs after extraction.
   Future automation: experiments can promote corpus-specific chunking defaults.

   Input object: `ChunkingConfig` (Python model, in memory).

   ```python
   ChunkingConfig(
       method="llamaindex_sentence_splitter", # Chunker implementation id.
       chunk_size=512,                        # Target chunk size.
       chunk_overlap=64,                      # Overlap between neighboring chunks.
   )
   ```

   Output object: `Chunk` (Python model, in memory).

   ```python
   Chunk(
       id="html:tensors:chunk:0000",       # Stable chunk id for Qdrant/citations/eval.
       document_id="html:tensors",         # Parent extracted document id.
       source_document_id="html:tensors",  # Original source document id.
       text="A torch.Tensor is a multi-dimensional matrix...", # Text to embed/retrieve.
       section_title="Tensors",            # Section label for prompt/citations.
       ordinal=0,                          # Chunk order inside the document.
       metadata={                          # Retrieval/citation/eval metadata.
           "source_uri": "https://pytorch.org/docs/stable/tensors.html", # Source URL.
           "title": "Tensors",            # Display title.
       },
   )
   ```

8. Chunks are persisted.

   Input object: `list[Chunk]` plus extraction provenance.
   Processing block: `write_chunk_artifact()`.
   Output object: `ChunkArtifact` persisted as JSON.
   Code/data home: chunk artifacts and checksum/path conventions belong in
   `src/rag/ingest`; chunk payload metadata remains generic plus adapter-provided
   `metadata`.
   Operator action: inspect chunk counts and failures for suspicious builds.
   Automatic now: chunk artifacts are written by the lifecycle.
   Future automation: quality gates can detect empty, duplicate, tiny, or
   oversized chunks.

   Output file: chunk artifact (JSON artifact).

   ```jsonc
   {
     "schema_version": 1,                 // Artifact schema version.
     "source_document_id": "html:tensors", // Source document that was chunked.
     "extracted_checksum": "sha256:...",  // Extraction artifact checksum.
     "chunking": {                        // Exact chunking settings.
       "method": "llamaindex_sentence_splitter", // Chunker implementation id.
       "chunk_size": 512,                 // Target chunk size.
       "chunk_overlap": 64                // Chunk overlap.
     },
     "chunks": [                          // Persisted chunks, abbreviated.
       {
         "id": "html:tensors:chunk:0000", // Stable chunk id.
         "ordinal": 0                     // Chunk order.
       }
     ]
   }
   ```

9. Source chunks are collected.

   Input object: chunk artifact files.
   Processing block: `collect_source_chunks()`.
   Output object: `SourceChunkBundle`.
   Code/data home: strict artifact validation and bundle collection belong in
   `src/rag/ingest`; the bundle is the generic materialization input for one
   source instance.
   Operator action: choose source instances or document ids to include.
   Automatic now: materialization collects requested bundles.
   Future automation: KB-level materialization can collect all catalog-selected
   bundles by default.

   Output object: `SourceChunkBundle` (Python model, in memory).

   ```python
   SourceChunkBundle(
       kb_id="pytorch_reference",          # KB all chunks belong to.
       source_instance_id="docs",          # Source instance represented by this bundle.
       document_count=1,                   # Number of source documents included.
       chunk_count=37,                     # Number of chunks included.
       chunk_artifact_checksums={          # Inputs used for source snapshot hashing.
           "assets/rag_data/pytorch_reference/chunks/docs/html_tensors.json": "sha256:..."
       },
   )
   ```

10. One or more bundles become a collection.

    Input object: `list[SourceChunkBundle]` plus materialization build profile.
    Processing block: `materialize_kb_collection()`.
    Output object: Qdrant physical collection.
    Code/data home: materialization orchestration belongs in `src/rag/indexing`;
    embedding/sparse clients are infrastructure dependencies; Qdrant stores
    vectors and payloads, not dataset-specific Python objects.
    Operator action: run materialization, select retrieval capability/profile,
    and choose whether to force-recreate.
    Automatic now: embeddings, sparse vectors when enabled, collection creation,
    and upserts are handled by the materializer.
    Future automation: successful builds can trigger materialization for approved
    build profiles.

    Input object: `MaterializationBuildProfile` (Python model, in memory).

    ```python
    MaterializationBuildProfile(
        retrieval_capability="dense",      # Physical retrieval capability to build.
        embedding_model="BAAI/bge-small-en-v1.5", # Dense embedding model.
        sparse_encoder_model=None,         # No sparse encoder for dense-only build.
        qdrant_distance="cosine",          # Vector distance/index setting.
    )
    ```

    External state: Qdrant collection.

    ```text
    rag__pytorch_reference__20260616_120000
    # Physical collection containing vectors, chunk text, and payload metadata.
    ```

11. Build provenance is written.

    Input object: materialization inputs plus collection metadata.
    Processing block: `IndexManifest(...)`, `write_index_manifest()`, and
    `attestation_payload()`.
    Output object: manifest JSON plus Qdrant attestation.
    Code/data home: `IndexManifest` and attestation contracts belong in
    `src/rag/contracts`; collection manifest writing and Qdrant attestation
    updates belong in `src/rag/indexing`.
    Operator action: inspect or diff manifests when reviewing candidates.
    Automatic now: manifests and attestations are written during materialization.
    Future automation: completeness checks can block alias promotion.

    Output file: collection manifest (JSON artifact).

    ```jsonc
    {
      "schema_version": 1,                // Manifest schema version.
      "kb_id": "pytorch_reference",       // Logical KB represented by the collection.
      "collection_name": "rag__pytorch_reference__20260616_120000", // Physical collection.
      "source_snapshot_id": "sha256:...", // Digest of bundle inputs/checksums.
      "embedding_model": "BAAI/bge-small-en-v1.5", // Dense model used.
      "retrieval_capability": "dense",   // Retrieval modes supported.
      "chunk_count": 37                   // Number of chunks materialized.
    }
    ```

12. Collection is promoted.

    Input object: attested physical collection plus alias role.
    Processing block: `promote_materialized_alias()`.
    Output object: Qdrant alias update.
    Code/data home: alias naming and promotion checks belong in
    `src/rag/indexing`; the alias is an operational pointer, not immutable
    collection payload.
    Operator action: explicitly promote to `challenger`, `champion`, or another
    alias after review.
    Automatic now: promotion validates attestation and alias/capability
    compatibility.
    Future automation: gated promotion can happen after offline evals and
    operating guardrails pass.

    External state: Qdrant alias.

    ```text
    rag__pytorch_reference__challenger -> rag__pytorch_reference__20260616_120000
    # Stable runtime alias pointing at a reviewed physical collection.
    ```

13. Runtime resolves the alias.

    Input object: API `RagRuntimeSource` plus catalog KB config.
    Processing block: `RagRuntime._resolve_alias_state()`.
    Output object: attested runtime alias state.
    Code/data home: runtime contracts and alias-state validation belong in
    `src/rag/runtime`; catalog lookup belongs in `src/app_config/catalog`;
    attestation parsing uses `src/rag/contracts`.
    Operator action: no per-request action; maintain aliases and collection health.
    Automatic now: runtime resolves aliases, validates attestations, and caches
    retrievers.
    Future automation: health checks can warm caches and alert on broken aliases.

    Input object: `RagRuntimeSource` (Python model, in memory).

    ```python
    RagRuntimeSource(
        knowledge_base="pytorch_reference", # Logical KB requested by API/user.
        alias="challenger",                # Alias requested, or default if omitted.
    )
    ```

    Output object: `_RuntimeAliasState` (Python model/dataclass, in memory).

    ```python
    RuntimeAliasState(
        kb_id="pytorch_reference",         # Validated KB id.
        alias="challenger",                # Runtime alias role.
        qdrant_alias="rag__pytorch_reference__challenger", # Qdrant alias name.
        collection_name="rag__pytorch_reference__20260616_120000", # Resolved target.
        manifest_id="sha256:...",          # Attested build manifest id.
    )
    ```

14. Query retrieves chunks.

    Input object: user query plus runtime alias state.
    Processing block: `Retriever.retrieve()`.
    Output object: `list[RetrievalHit]`.
    Code/data home: retrieval mechanics belong in `src/rag`; runtime hit
    contracts belong in `src/rag/contracts`; payload metadata must be
    citation/eval-ready.
    Operator action: no per-request action; tune retrieval through alias/build
    profiles.
    Automatic now: runtime embeds query, searches Qdrant, optionally reranks, and
    returns hits.
    Future automation: analytics can recommend challenger retrieval settings.

    Output object: `RetrievalHit` (Python model, in memory).

    ```python
    RetrievalHit(
        chunk_id="html:tensors:chunk:0000", # Retrieved chunk id.
        document_id="html:tensors",        # Parent document id.
        text="A torch.Tensor is...",       # Retrieved context text.
        score=0.82,                        # Retrieval/rerank score.
        source_type="html_docs",           # Source metadata from payload.
        title="Tensors",                   # Citation/display title.
        uri="https://pytorch.org/docs/stable/tensors.html", # Citation URI.
    )
    ```

15. Hits become prompt context.

    Input object: `list[RetrievalHit]`.
    Processing block: prompt/context builders such as `trim_rag_chunks()` and
    `render_rag_sections()`.
    Output object: prompt context items.
    Code/data home: prompt assembly belongs in the gateway layer; it consumes RAG
    contracts but should not know dataset adapter internals.
    Operator action: no per-request action; review prompt/context policies when
    experiments show weak grounding or overflow.
    Automatic now: prompt construction trims and formats retrieved context.
    Future automation: prompt-budget policies can adapt context selection.

    Output object: prompt context item (Python dict/model, in memory).

    ```python
    {
        "citation_label": "[1]",          # Stable label shown to the model/user.
        "chunk_id": "html:tensors:chunk:0000", # Machine-readable source pointer.
        "title": "Tensors",               # Human-readable source title.
        "text": "A torch.Tensor is...",   # Context text included in the prompt.
        "metadata": {"knowledge_base": "pytorch_reference", "alias": "challenger"}, # Provenance.
    }
    ```

16. Model answer is produced.

    Input object: prompt context items plus model request.
    Processing block: gateway generation path plus citation extraction/attachment.
    Output object: answer text plus citation records.
    Code/data home: generation orchestration and API response shaping belong in
    the gateway; citation records point back to `RetrievalHit` metadata.
    Operator action: no per-request action; evaluate citation behavior before
    promoting prompt/model changes.
    Automatic now: generation runs with selected prompt/context.
    Future automation: citation validators can flag unsupported citations or
    claims for feedback and regression sets.

    Output object: API response payload (JSON response).

    ```jsonc
    {
      "answer": "A tensor is PyTorch's multi-dimensional array object [1].", // User-visible answer.
      "citations": [                    // Structured citations returned beside text.
        {
          "label": "[1]",                // Visible citation label.
          "chunk_id": "html:tensors:chunk:0000", // Supporting retrieved chunk.
          "uri": "https://pytorch.org/docs/stable/tensors.html", // Inspectable source.
          "manifest_id": "sha256:..."    // Build provenance.
        }
      ]
    }
    ```

17. Observability is recorded.

    Input object: runtime provenance plus retrieval/generation timings.
    Processing block: inference-event producer / analytics writers.
    Output object: runtime observability row.
    Code/data home: event contracts belong in `src/shared/events` or analytics
    schema modules; RAG runtime supplies provenance but does not own dashboards.
    Operator action: inspect dashboards/logs/reports when debugging or deciding
    promotions.
    Automatic now: provenance, scores, timings, and no-hit/error flags are
    captured.
    Future automation: monitoring can create failure-regression rows or alerts.

    Output object: analytics row (JSON event/ClickHouse row).

    ```jsonc
    {
      "knowledge_base": "pytorch_reference", // Logical KB requested.
      "alias": "challenger",              // Runtime alias used.
      "collection_name": "rag__pytorch_reference__20260616_120000", // Resolved target.
      "manifest_id": "sha256:...",        // Build id for joins/audits.
      "hit_count": 5,                     // Number of retrieved chunks.
      "no_hit": false                     // Whether retrieval returned usable context.
    }
    ```

For a multi-source KB, steps 1-9 run per selected source instance, then step 10
materializes all selected `SourceChunkBundle`s into one physical collection.

### Example Evaluation Walkthrough

This example uses a single QASPER-style benchmark question. It does not replace
the source build flow; it creates rows that the evaluation harness can run
against a built KB or benchmark collection.

1. Dataset split is selected.

   Input object: dataset pipeline config.
   Processing block: `rag_data_pipelines.qasper.prepare`.
   Output object: pinned dataset selection.
   Code/data home: dataset-specific prepare code belongs in
   `src/rag_data_pipelines/qasper`; the generic harness sees only normalized
   contracts and pinned metadata.
   Operator action: choose dataset, split, revision, subset, and output paths.
   Automatic now: prepare entrypoint loads selected dataset config.
   Future automation: benchmark suites can resolve pinned configs from named
   experiment profiles.

   Input file: QASPER pipeline config (TOML file).

   ```toml
   dataset_name = "qasper"                # Benchmark dataset family.
   dataset_version = "allenai/qasper@<revision>" # Exact dataset revision.
   split = "validation"                   # Dataset split selected.
   benchmark_scope = "kb"                 # Result scope for promotion decisions.
   ```

2. Corpus documents are normalized.

   Input object: QASPER corpus rows.
   Processing block: QASPER source adapter.
   Output object: `list[SourceDocument]`.
   Code/data home: QASPER normalization belongs in
   `src/rag_data_pipelines/qasper`; emitted `SourceDocument`s use
   `src/rag/contracts`; QASPER-only fields stay in `metadata`.
   Operator action: review normalization rules when importing a new dataset.
   Automatic now: adapter maps corpus rows into `SourceDocument`s.
   Future automation: import checks can compare normalized counts and ids.

   Output object: `SourceDocument` (Python model, in memory).

   ```python
   SourceDocument(
       id="qasper:paper:123",              # Stable id for corpus/qrel/chunk joins.
       source_type="paper_pdf_or_json",    # Source metadata describing corpus form.
       uri="hf://allenai/qasper/123",      # Dataset-local source reference.
       title="Example NLP Paper",          # Human-readable paper title.
       metadata={"dataset": "qasper", "split": "validation"}, # Dataset provenance.
   )
   ```

3. Labels become eval rows.

   Input object: QASPER QA labels.
   Processing block: `DatasetEvalAdapter.emit_rows()`.
   Output object: `list[NormalizedEvalRow]`.
   Code/data home: adapter logic belongs in `src/rag_data_pipelines/qasper`; the
   normalized row model belongs in `src/rag/evaluation`; rubrics and notes remain
   structured fields, not notebook state.
   Operator action: review row mapping, answer/rubric policy, and benchmark
   scope before trusting the dataset for promotion.
   Automatic now: adapter emits normalized eval rows.
   Future automation: schema validation and sample audits can run in CI/Airflow.

   Output object: `NormalizedEvalRow` (Python model, in memory).

   ```python
   NormalizedEvalRow(
       dataset_name="qasper",              # Dataset family.
       dataset_version="allenai/qasper@<revision>", # Exact dataset revision.
       benchmark_scope="kb",              # Evaluation scope.
       query_id="qasper:123:q1",           # Stable query id.
       query="What dataset is introduced?", # User-style query.
       expected_answer="...",             # Expected answer or rubric.
       relevant_doc_ids=["qasper:paper:123"], # Supporting source documents.
   )
   ```

4. Evidence becomes retrieval truth.

   Input object: normalized evidence spans / relevant ids.
   Processing block: qrel/evidence builder.
   Output object: qrels and evidence refs.
   Code/data home: generic qrel/evidence contracts belong in
   `src/rag/evaluation`; dataset-specific evidence mapping belongs in the data
   pipeline adapter.
   Operator action: choose evidence-span or document-level relevance policy when
   the source dataset is ambiguous.
   Automatic now: qrels are generated from normalized evidence fields.
   Future automation: chunk-level qrels can be derived after chunking.

   Output object: `Qrel` (Python model, in memory).

   ```python
   Qrel(
       query_id="qasper:123:q1",           # Query this label belongs to.
       document_id="qasper:paper:123",     # Relevant document id.
       relevance_grade=1,                  # Binary or graded relevance value.
       evidence_ref={"section": "Experiments", "span": [120, 180]}, # Evidence location.
   )
   ```

5. Evaluation runs retrieval.

   Input object: `NormalizedEvalRow` plus candidate KB/alias.
   Processing block: evaluation runner plus `RagRuntime.retrieve()`.
   Output object: retrieval outputs with collection provenance.
   Code/data home: orchestration belongs in `src/rag/evaluation`; runtime
   retrieval remains in `src/rag/runtime`; dataset adapters should not call
   Qdrant directly.
   Operator action: choose run id, candidate KB/alias, subset, and eval profile.
   Automatic now: runner resolves alias and records collection/manifest metadata.
   Future automation: benchmark suites can run after candidate materialization.

   Output object: retrieval eval observation (Python model/dict, in memory).

   ```python
   RetrievalEvalObservation(
       query_id="qasper:123:q1",           # Eval row executed.
       knowledge_base="research_papers",  # KB under evaluation.
       alias="challenger",                # Candidate alias/profile.
       resolved_collection="rag__research_papers__20260616_120000", # Collection queried.
       manifest_id="sha256:...",          # Build manifest id.
   )
   ```

6. Evaluation runs generation.

   Input object: retrieval outputs plus prompt/generation profile.
   Processing block: evaluation runner through gateway/runtime generation path.
   Output object: answer outputs.
   Code/data home: evaluation runner owns experiment execution; gateway owns
   prompt/generation behavior; eval rows should not encode prompt internals.
   Operator action: choose generation profile and retrieval-only vs full-answer
   evaluation.
   Automatic now: runner calls retrieval, prompt construction, and generation.
   Future automation: tiered eval can start with retrieval-only checks.

   Output object: answer eval observation (Python model/dict, in memory).

   ```python
   AnswerEvalObservation(
       query_id="qasper:123:q1",           # Eval row answered.
       answer="...",                      # Generated answer.
       retrieved_chunk_ids=["qasper:paper:123:chunk:0004"], # Retrieved context.
       cited_chunk_ids=["qasper:paper:123:chunk:0004"], # Cited context.
       prompt_tokens=1840,                 # Prompt budget usage.
   )
   ```

7. Metrics are computed.

   Input object: eval row plus qrels plus retrieval/answer outputs.
   Processing block: metric calculators.
   Output object: metric values.
   Code/data home: reusable metrics belong in `src/rag/evaluation`; exploratory
   metrics can start in `experiments/` but must be promoted before gating.
   Operator action: choose metric set and thresholds.
   Automatic now: configured metrics are computed.
   Future automation: thresholds can be enforced as promotion gates.

   Output object: metrics dict (Python dict, in memory).

   ```python
   {
       "recall_at_5": 1.0,                # Relevant evidence appeared in top 5.
       "mrr": 1.0,                        # Rank-sensitive retrieval score.
       "answer_correctness": 0.8,         # Answer quality score/judge output.
       "citation_precision": 1.0,         # Citations point to supporting context.
       "latency_ms": 920.0,               # End-to-end latency for this row.
   }
   ```

8. Result is persisted.

   Input object: row outputs plus metrics plus manifest/runtime provenance.
   Processing block: eval result writer.
   Output object: `EvalResult` persisted to configured storage.
   Code/data home: `EvalResult` belongs in `src/rag/evaluation`; storage adapters
   can write ClickHouse, files, or experiment artifacts without changing dataset
   adapters.
   Operator action: choose persistence target and inspect failed rows/reports.
   Automatic now: metrics, errors, and manifest ids are persisted.
   Future automation: result tables can feed dashboards and comparison notebooks.

   Output artifact/row: `EvalResult` (JSON artifact or DB row).

   ```jsonc
   {
     "type": "EvalResult",               // Persisted result contract.
     "run_id": "rag_baseline_v1_20260616", // Evaluation run id.
     "query_id": "qasper:123:q1",         // Eval row id.
     "rag_manifest_id": "sha256:...",     // Candidate build evaluated.
     "metrics": {                         // Selected persisted metrics.
       "recall_at_5": 1.0,                // Retrieval quality.
       "answer_correctness": 0.8          // Answer quality.
     },
     "errors": []                         // Row-level failures.
   }
   ```

9. Promotion decision is made.

   Input object: aggregate `EvalResult`s plus guardrail thresholds.
   Processing block: promotion rule/report.
   Output object: approve/reject/defer decision.
   Code/data home: promotion-rule inputs belong with the evaluation harness; the
   actual alias move stays in `src/rag/indexing` and remains explicit until
   automation is proven safe.
   Operator action: approve, reject, or defer promotion.
   Automatic now: reports summarize whether candidate gates passed.
   Future automation: low-risk challenger promotion can be automated after all
   offline and online gates pass.

   Output object: promotion decision (Python model/dict, in memory).

   ```python
   PromotionDecision(
       candidate="rag__research_papers__20260616_120000", # Collection considered.
       promote=True,                       # Whether promotion gates passed.
       passed_gates=["kb_scoped_quality", "latency", "citation_quality"], # Passed gates.
   )
   ```

## Why This Comes First

The upcoming RAG experiments need QASPER, Open RAG Benchmark, BEIR, MS MARCO,
and failure-regression data. If each dataset forces edits inside the generic
ingest/indexing lifecycle, experiments will blur platform code with
data-specific decisions.

This refactor makes the unit of experimentation clearer:

- platform code defines what a valid source, chunk, eval row, collection, and
  alias are;
- pipeline code defines how one corpus becomes those contracts;
- experiment code measures which build configuration should be promoted.

## Refactor Steps

1. Define stable source and evaluation adapter contracts.

   The platform should expose interfaces for:

   - listing source documents;
   - fetching or resolving raw artifacts;
   - extracting text and structured sections;
   - chunking extracted text through a generic chunking contract;
   - emitting eval rows when the dataset has labels.

   Treat adapter identity as `adapter id + adapter version + settings`, not as a
   closed enum of source types. The current `source_type` can remain as payload
   metadata, but it should not be the extension mechanism.

2. Make the source manifest contract extensible.

   Current source manifests are tied to a small fixed set of source types. Move
   toward a generic manifest envelope with adapter-specific settings and
   validation delegated to the selected adapter.

3. Add source-level ingest adapter mapping to the catalog.

   Keep the current `[[sources]]` fields `type`, `kb`, `id`, and `manifest`.
   Add an explicit source-level ingest adapter link, preferably
   `ingest_adapter = { id = "...", version = "...", settings = {...} }`.
   This should not reuse the existing task-level `adapter` field, because that
   field currently describes LoRA/task routing rather than source ingestion.

4. Add production-owned dataset pipeline modules.

   Create `src/rag_data_pipelines/` and start with import/prepare entrypoints
   for the planned benchmark corpora:

   - QASPER;
   - Open RAG Benchmark;
   - BEIR;
   - MS MARCO subset.

   Root-level `pipelines/rag/` should be skipped unless there is a clear need
   for operator-facing wrappers or config bundles. Production code should have
   one importable, tested home.

5. Make KB builds support multiple source instances.

   A knowledge base should be able to materialize one collection from all
   relevant source instances, or from an explicitly selected subset. The CLI and
   Airflow DAG should not assume one source instance per KB.

6. Add a shared lifecycle command layer for CLI and Airflow.

   CLI commands and Airflow DAG tasks should construct the same request models
   and call the same lifecycle functions. CLI supplies command arguments and
   terminal output; Airflow supplies DAG Params, task logs, retries, and
   scheduling. Neither surface should become a separate implementation of the
   RAG lifecycle.

7. Move corpus-specific defaults out of operator surfaces.

   Airflow defaults, shell examples, and notebooks should present generic
   placeholders or catalog-derived choices rather than treating
   `pytorch_reference/docs` as the implicit default.

8. Split application config from cross-cutting shared infrastructure.

   Move catalog and runtime config modules toward `src/app_config/`. Keep
   `src/shared/` for cross-cutting infrastructure such as events, database
   helpers, logging, and telemetry. The catalog is application configuration,
   not a generic shared utility and not purely RAG-owned.

9. Record full build configuration in manifests.

   Collection manifests should preserve enough detail to reproduce a RAG
   experiment:

   - source adapter ids and versions;
   - source manifest refs and dataset versions;
   - raw manifest/config digests;
   - parser/extractor settings;
   - chunk size, overlap, and chunking method;
   - embedding and sparse encoder settings;
   - vector-store distance, index, and payload-index settings;
   - retrieval capability;
   - build config reference or digest;
   - applicable benchmark scope when known.

10. Keep experiments as consumers, not dependencies.

   Notebooks and analysis scripts should call production CLI/pipeline
   entrypoints. They should not be required to build production KB aliases.

## Acceptance Criteria

- Adding a new dataset-backed KB does not require editing the core RAG
  lifecycle, except when a genuinely new generic capability is needed.
- Dataset-specific production code lives in `src/rag_data_pipelines/`, not in
  notebooks.
- Catalog/runtime config code lives in `src/app_config/`; `src/shared/` remains
  limited to cross-cutting infrastructure.
- Catalog source entries keep the existing `type`, `kb`, `id`, and `manifest`
  shape while adding an explicit ingest adapter id/version/settings link.
- If `pipelines/rag/` exists, it contains only thin wrappers or configs and does
  not become a second implementation home.
- Adding a dataset requires registering a new adapter or pipeline module, not
  editing a closed `SourceType` enum, `SourceManifest` union, default fetcher
  maps, or generic lifecycle orchestration.
- Generic lifecycle commands can build and materialize all source instances for
  a KB or a selected subset.
- CLI commands and Airflow DAG tasks use the same lifecycle request models,
  stage functions, build-run status model, and promotion rules.
- A persisted `BuildRun` records run id, catalog digest, manifest/config
  digests, adapter versions, build profile digest, status, artifacts, reports,
  and collection names.
- The plan clearly separates Git-versioned setup, DVC/artifact-versioned
  generated inputs, rebuildable Qdrant state, ephemeral runtime state, and
  observability/audit state.
- RAG restore procedures are path-scoped or isolated in a temporary worktree, so
  restoring old DVC artifacts does not implicitly roll back runtime,
  infrastructure, or unrelated project files.
- Airflow DVC sync uses data-only bot branches/PRs, while code/config changes
  use normal feature branches; build reports link both the producing Git SHA and
  the DVC pointer commit.
- Collection manifests preserve the data-specific and build-specific settings
  needed to reproduce a collection.
- Normalized eval row and evidence contracts are versioned, tested, and shared
  by dataset adapters and the evaluation harness.
- Operator docs distinguish platform lifecycle commands from dataset
  preparation commands.
- Existing PyTorch and arXiv pipelines still work after the split.

## Current Implementation Status

As of 2026-06-17, the core isolation refactor is mostly implemented, with a few
remaining phases tracked below.

Completed:

- Catalog schema/loading moved from `shared.catalog` to `app_config.catalog`.
- RAG contracts moved to `rag.contracts`; stale `rag.domain` compatibility
  modules were removed.
- Source manifests are generic; hardcoded source-specific manifest entry
  classes and closed source-type unions were removed.
- Source-level `ingest_adapter` is required in catalog source entries and is
  used to resolve validation, document listing, fetch, and extract behavior.
- Stale source connector/default fetcher maps were removed from the generic
  lifecycle; source behavior now flows through `SourceAdapter`.
- KB builds support all catalog sources or explicit source subsets.
- CLI and Airflow share `BuildRequest`, `BuildRun`, and lifecycle stage
  functions.
- Materialization and alias promotion live under `rag.indexing`.
- Airflow can persist one build-run audit across build, materialize, DVC sync,
  and optional promotion.
- `dry_run` is wired through CLI/Airflow lifecycle stages.
- `src/rag_data_pipelines/pytorch_docs` exists as the first production-owned
  dataset pipeline module.
- Runtime settings schema/loading moved from `shared.config` into
  `app_config.runtime` (`models.py` + `loaders.py`); `shared/config.py` deleted.
  `src/shared/` is limited to cross-cutting infrastructure (db, events, logging,
  telemetry).
- Architecture and operations docs updated to reflect the completed package split.
- `IndexManifest` extended with `source_adapter_versions`, `source_manifest_digests`,
  `vector_dimension`, `build_config_digest`, and `benchmark_scope`; `_chunking_config`
  reads chunking settings from the first reachable chunk artifact; all new fields are
  wired through `materialize_kb_collection` and threaded from the CLI via the
  persisted `BuildRun`.
- `plan_build`, `list_build_runs`, and `read_build_run` added to the lifecycle layer;
  `plan`, `status`, and `show-build-run` subcommands wired in the CLI; `PlanResult`
  and `SourcePlanEntry` added to `rag.lifecycle` contracts.

Remaining phases:

1. **Dataset pipeline first slices.**
   Add real production modules under `src/rag_data_pipelines/` for at least one
   benchmark corpus, then continue with QASPER, Open RAG Benchmark, BEIR, and
   MS MARCO as needed.

2. **RAG evaluation harness.**
   Continue with normalized eval rows, qrels/evidence, retrieval observations,
   metrics, result persistence, and promotion gates.

## Follow-Up

After this refactor, continue with:

1. [Inference Baseline](inference-baseline.md)
2. [RAG Evaluation Harness](rag-evaluation-harness.md)
3. [RAG Experiment Series](rag-experiment-series.md)
