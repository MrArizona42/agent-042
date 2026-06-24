# Declarative RAG Alias Workflow Implementation Plan

This document defines the control-plane refactor that follows the completed
RAG catalog isolation and LlamaIndex transition. The data-plane mechanics are
already in place: catalog-declared adapters produce native LlamaIndex
documents and nodes, Qdrant stores materialized indexes, runtime retrieval uses
LlamaIndex, and benchmark results are written to Postgres.

The remaining problem is operational. The current CLI exposes internal stages
(`plan`, `build-source`, `materialize`, `promote-alias`, and build-run artifact
commands) instead of the desired-state operation an operator actually means:

```text
make this KB alias match its catalog declaration
```

This plan replaces the stage-oriented workflow with an alias-centered,
declarative control plane.

## Status And Scope

Status: proposed implementation plan.

This plan supersedes the operator-workflow and promotion portions of
`docs/planning/rag-llamaindex-transition-plan.md`. It does not replace the
LlamaIndex document, node, indexing, retrieval, query-engine, or evaluator
decisions implemented by that plan.

In scope:

- alias-owned build and retrieval configuration in `catalog.toml`;
- immutable RAG releases;
- desired state from the catalog versus applied state in Postgres;
- deterministic build, retrieval, source, and release fingerprints;
- configuration-aware source node caches;
- alias diff and apply reconciliation;
- applied-state runtime resolution;
- release-aware benchmark execution;
- a unified nested Typer CLI;
- Airflow and cleanup migration;
- removal of the old lifecycle CLI and build-run artifacts.

Out of scope:

- scheduled benchmark execution;
- automatic metric-threshold promotion policy;
- a web UI for RAG operations;
- support for multiple dense or sparse provider endpoints in one process;
- backward-compatible execution of old lifecycle commands;
- migration of old physical collections without rebuilding them;
- changing the normalized benchmark case, label, qrel, or observation shapes.

## Locked Design Decisions

1. `catalog.toml` is desired state.
2. An alias declaration contains both `build` and `retrieve` configuration.
3. Postgres is required and stores applied alias state, releases, and build
   attempts.
4. Qdrant aliases remain operational mirrors and inspection aids, but gateway
   runtime resolution uses the active Postgres deployment record directly.
5. Editing an alias `build` or `retrieve` block alone never changes live
   retrieval behavior. Structural routing fields such as task-to-KB mapping and
   `default_alias` remain normal application configuration and take effect when
   that catalog version is deployed.
6. `rag alias apply <kb> <alias>` is the mutation operation. There is no
   operator-facing promotion operation.
7. `rag alias diff <kb> <alias>` compares desired and applied state. This gives
   the old `plan` concept a precise replacement.
8. Releases are immutable and may be applied to more than one alias.
9. Build attempts and successful releases are different contracts. A failed
   build attempt never becomes a release.
10. Dense and sparse vector producers use symmetric names: `dense_encoder` and
    `sparse_encoder`.
11. Encoder model identity is build-semantic catalog configuration. Provider
    URLs, credentials, timeouts, concurrency, and batching remain runtime
    configuration.
12. A release snapshots the complete resolved build configuration. Later
    catalog edits do not change an existing release.
13. Benchmark execution always names an explicit alias.
14. Benchmark preparation is automatic when artifacts are missing or stale.
15. The default alias is protected: apply may reuse an evaluated matching
    release, but it does not silently create a new release for the default
    alias.
16. The current project has no production collection that requires a legacy
    compatibility path. Old collections and build-run artifacts may be
    discarded and rebuilt.
17. Typer owns parsing, help, completion, output selection, and exit-code
    translation only. Application services remain independent of Typer.

## Current State And Problems

### Catalog

Current `CatalogAliasConfig` contains only query-time fields:

```text
top_k
score_threshold
reranker
retrieval_strategy
reranker_multiplier
```

Index-time behavior is assembled from unrelated locations:

- source membership comes from source instances in `catalog.toml`;
- chunking comes from code defaults or CLI overrides;
- the dense encoder model comes from `runtime.toml`;
- the sparse encoder model comes from `runtime.toml`;
- physical dense/hybrid capability is inferred from the selected alias's
  retrieval strategy;
- Qdrant batch sizes come from `runtime.toml`.

This prevents the catalog from describing the release an alias is supposed to
serve.

### Lifecycle

`src/rag/sources/cli.py` exposes internal stages and asks operators to repeat
catalog paths, artifact roots, KB ids, source ids, collection names, run ids,
and persistence flags. `BuildRun` is used both as an execution record and as
release provenance even though those concepts have different lifecycles.

Materialization can produce a collection without complete provenance when no
explicit persisted build-run id is supplied. Reopening a build run for a later
stage can retain a build-profile digest produced from an earlier stage request.

### Runtime

Runtime currently combines:

- retrieval parameters read immediately from the active catalog;
- the collection currently targeted by a Qdrant alias;
- global dense/sparse encoder identity from runtime settings.

Consequently, editing retrieval settings in the catalog can affect live
behavior before an operator performs a deployment operation. The runtime also
cannot represent the exact applied catalog snapshot separately from current
desired state.

### Benchmarking

Benchmark execution records physical collection and manifest identity, but it
does not have first-class release or alias-deployment identity. One benchmark
execution creates a separate `eval_runs` row id for every metric, so there is
no shared execution id suitable for release evaluation coverage.

Benchmark preparation is a separate mandatory CLI step even though its
metadata already contains enough provenance to determine whether regeneration
is required.

## Target Conceptual Model

### Desired Alias

A catalog alias is the complete desired state for one named KB deployment:

```text
DesiredAlias
  = AliasBuildConfig
  + AliasRetrievalConfig
```

`AliasBuildConfig` describes artifacts that must exist. Changing it can require
new chunks and a new physical collection.

`AliasRetrievalConfig` describes how queries use an already compatible
release. Changing it does not require rebuilding a collection.

### Release

A release is an immutable, reusable build result:

```text
RagRelease
  = KB identity
  + source snapshot
  + resolved build configuration
  + physical Qdrant collection
  + release manifest
  + Qdrant attestation
```

The same release can be deployed behind `challenger` and later behind the
default alias. Retrieval configuration belongs to the deployment, not the
release.

### Applied Alias Deployment

An applied alias deployment is an immutable history row:

```text
AliasDeployment
  = KB and alias
  + release id and collection
  + snapshotted retrieval configuration
  + desired catalog/build/retrieval digests
  + activation timestamps and status
```

At most one deployment is active for a `(kb_id, alias)` pair.

### Build Attempt

A build attempt records execution and failure information. It may produce one
release or fail before a release exists. It is not used as the runtime source
of truth.

## Configuration Ownership

Use this rule:

> If changing a value can change release contents, vector compatibility, or
> query semantics, it belongs in `catalog.toml`. If it only changes service
> connectivity or execution performance, it belongs in `runtime.toml`.

Catalog-owned values:

- source membership and adapter identity;
- chunking strategy, size, and overlap;
- dense encoder model and dimension;
- optional sparse encoder model;
- retrieval strategy, top-k, threshold, and reranker identity;
- reranker candidate multiplier.

Runtime-owned values:

- embedding/sparse service URL;
- reranker service URL;
- Qdrant host and port;
- Postgres URL;
- HTTP timeouts and retries;
- embedding and Qdrant batch sizes;
- worker concurrency and device placement.

The current `rag.embedding_model`, `rag.sparse_encoder_model`, and
`rag.embedding_device` runtime fields are retired after catalog migration.
Runtime provider clients must report their actual model identities so catalog
values can be validated before build or query.

## Target Catalog Contract

Increment `schema_version` from `3` to `4`.

### Pydantic Shape

```python
RetrievalStrategy = Literal["dense", "sparse", "hybrid"]
ChunkingStrategy = Literal["sentence"]


class DenseEncoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    dimension: int = Field(gt=0)


class SparseEncoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str


class AliasChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: ChunkingStrategy
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)


class AliasBuildConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunking: AliasChunkingConfig
    dense_encoder: DenseEncoderConfig
    sparse_encoder: SparseEncoderConfig | None = None


class AliasRetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: RetrievalStrategy
    top_k: int = Field(gt=0)
    score_threshold: float
    reranker: str | None = None
    reranker_multiplier: int = Field(default=1, gt=0)


class CatalogAliasConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    build: AliasBuildConfig
    retrieve: AliasRetrievalConfig
```

Validation rules:

- `dense_encoder` is required in the first implementation;
- `sparse` retrieval requires `sparse_encoder`;
- `hybrid` retrieval requires both encoders;
- `chunk_overlap < chunk_size`;
- omitted `reranker` requires `reranker_multiplier == 1`;
- configured provider model identity and dense dimension must match the live
  provider before build and runtime activation;
- every KB `default_alias` must still reference a declared alias.

Physical capability is derived from encoders rather than stored redundantly:

```text
dense encoder only  -> dense
both encoders       -> hybrid
```

Sparse retrieval uses the sparse leg of a hybrid-capable release. Sparse-only
physical collections are not added by this refactor.

### Exact TOML Example

```toml
schema_version = 4

[[tasks]]
id = "code"
description = "Programming help for ML systems and PyTorch."
knowledge_bases = ["pytorch_reference"]
lora_adapter = { enabled = false }

[[knowledge_bases]]
id = "pytorch_reference"
description = "PyTorch documentation for coding assistance."
update_strategy = "replace" # allowed values: "replace", "incremental"
default_alias = "champion"

[knowledge_bases.aliases.champion.build.chunking]
strategy = "sentence" # allowed values currently: "sentence"
chunk_size = 512
chunk_overlap = 64

[knowledge_bases.aliases.champion.build.dense_encoder]
model = "sentence-transformers/all-MiniLM-L6-v2"
dimension = 384

# Omit the whole sparse_encoder table for a dense-only release.
[knowledge_bases.aliases.champion.retrieve]
strategy = "dense" # allowed values: "dense", "sparse", "hybrid"
top_k = 5
score_threshold = 0.35
# reranker is optional; when omitted, reranker_multiplier must be 1.
reranker_multiplier = 1

[knowledge_bases.aliases.challenger.build.chunking]
strategy = "sentence" # allowed values currently: "sentence"
chunk_size = 512
chunk_overlap = 64

[knowledge_bases.aliases.challenger.build.dense_encoder]
model = "sentence-transformers/all-MiniLM-L6-v2"
dimension = 384

[knowledge_bases.aliases.challenger.build.sparse_encoder]
model = "Qdrant/bm25"

[knowledge_bases.aliases.challenger.retrieve]
strategy = "hybrid" # allowed values: "dense", "sparse", "hybrid"
top_k = 5
score_threshold = 0.01
reranker = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker_multiplier = 4

[[source_adapters]]
id = "generic.http_html"
version = "1"
description = "Fetches HTTP HTML pages and extracts readable text."
factory = "rag.adapters.sources:make_http_html_adapter"

[[source_instances]]
id = "pytorch_reference.docs"
description = "Official PyTorch documentation pages."
role = "corpus" # allowed values: "corpus", "benchmark"
knowledge_base = "pytorch_reference"
adapter = { id = "generic.http_html", version = "1" }
```

Duplicating a build block between aliases is intentional: each alias owns its
complete desired state. The canonical digest removes formatting differences,
and apply reuses an existing release when the resolved build and source
fingerprints match.

## Fingerprints And Identity

All fingerprints use canonical Pydantic JSON with sorted keys, compact
separators, UTF-8 encoding, and SHA-256 prefixed by `sha256:`.

### Build Configuration Digest

```text
build_config_digest = sha256(canonical(alias.build))
```

It excludes retrieval configuration and runtime-only batching/connectivity.

### Retrieval Configuration Digest

```text
retrieval_config_digest = sha256(canonical(alias.retrieve))
```

### Source Declaration Digest

```text
source_declaration_digest = sha256(
  ordered corpus source instance ids
  + source manifest digests
  + adapter ids and versions
)
```

This can be computed without fetching remote content.

### Transformation Digest

```text
transformation_digest = sha256(
  alias.build.chunking
  + adapter extraction/node metadata contract version
)
```

It scopes node/chunk artifacts. Dense and sparse encoder identities are not
included because vectors are materialized in Qdrant, not cached as node
artifacts.

### Source Snapshot Id

```text
source_snapshot_id = sha256(
  ordered source instance ids
  + node artifact checksums
)
```

This identifies the exact content consumed by materialization.

### Release Fingerprint And Id

```text
release_fingerprint = sha256(
  kb_id
  + build_config_digest
  + source_declaration_digest
  + source_snapshot_id
)

release_id = "ragrel_<sanitized-kb-id>_<first-16-hex>"
collection_name = "rag__<sanitized-kb-id>__<first-16-hex>"
```

The full fingerprint remains stored and validated. A ready release with the
same full fingerprint is reused rather than rebuilt.

`manifest_id` is the SHA-256 digest of the canonical release manifest payload
excluding `manifest_id` itself. Digest directory names use the hexadecimal
portion without the `sha256:` prefix.

### Build Attempt Id

Build attempts use generated UUIDs. Retries create new attempts while still
being able to produce or reuse the same content-identified release.

## Target Contracts

Add these Pydantic contracts under `src/rag/control_plane/models.py`.

```python
BuildStatus = Literal["running", "failed", "completed"]
DeploymentStatus = Literal["pending", "active", "superseded", "failed"]


class ReleaseBuildAttempt(BaseModel):
    id: UUID
    kb_id: str
    requested_alias: str
    status: BuildStatus
    catalog_digest: str
    build_config_digest: str
    retrieval_config_digest: str
    source_declaration_digest: str
    source_snapshot_id: str | None = None
    release_id: str | None = None
    collection_name: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    error: str | None = None


class RagRelease(BaseModel):
    schema_version: int = 1
    id: str
    kb_id: str
    collection_name: str
    manifest_id: str
    release_fingerprint: str
    catalog_digest: str
    build_config_digest: str
    source_declaration_digest: str
    source_snapshot_id: str
    build_config: AliasBuildConfig
    source_manifest_digests: dict[str, str]
    source_adapter_versions: dict[str, str]
    document_count: int
    chunk_count: int
    created_at: datetime


class AliasDeployment(BaseModel):
    id: UUID
    kb_id: str
    alias: str
    release_id: str
    collection_name: str
    catalog_digest: str
    build_config_digest: str
    retrieval_config_digest: str
    retrieval_config: AliasRetrievalConfig
    status: DeploymentStatus
    applied_at: datetime | None = None
    superseded_at: datetime | None = None
    error: str | None = None


class AliasDiff(BaseModel):
    kb_id: str
    alias: str
    desired_catalog_digest: str
    desired_build_config_digest: str
    desired_retrieval_config_digest: str
    applied_deployment_id: UUID | None
    applied_release_id: str | None
    build_drift: bool
    retrieval_drift: bool
    source_declaration_drift: bool
    provider_mismatches: list[str]
    reusable_release_ids: list[str]
```

`RagRelease` contains no alias field. Applying it to an alias is represented by
`AliasDeployment`.

Replace `IndexManifest` with a release manifest using the `RagRelease` shape.
Replace `CollectionAttestation` schema version 1 with schema version 2:

```python
class CollectionAttestation(BaseModel):
    schema_version: Literal[2] = 2
    release_id: str
    manifest_id: str
    kb_id: str
    collection_name: str
    release_fingerprint: str
    build_config_digest: str
    source_snapshot_id: str
    dense_encoder_model: str
    dense_vector_dimension: int
    sparse_encoder_model: str | None
    retrieval_capability: RetrievalCapability
    chunk_count: int
    created_at: datetime
```

Remove obsolete `IndexManifest` fields:

- `alias`;
- `build_profile_digest`;
- `benchmark_scope`;
- `eval_summary`.

Remove `PromotionDecision`; this control plane has no promotion operation.

## Database Contract

Postgres becomes required for RAG runtime and operations. Add ORM models to
`src/clients/db/models.py` and matching idempotent SQL migration files under
`migrations/postgres/`.

`Base.metadata.create_all()` creates new tables for fresh databases, but it
does not alter existing tables. The deployment procedure must execute the
explicit `ALTER TABLE` migration for new `eval_runs` columns.

Add an idempotent `scripts/apply_agent042_db_migrations.sh` entrypoint that
applies the control-plane table SQL and `eval_runs` alteration before gateway,
RAG operations, or Airflow uses the new schema. Do not make gateway startup a
private migration runner.

The control plane uses repository protocols. The initial implementation may
use a synchronous SQLAlchemy engine because source processing, Qdrant
materialization, and `RagRuntime.retrieve()` are synchronous today. Database
URL normalization belongs in one shared helper rather than being copied from
`eval_writer.py`. The gateway's existing async ORM engine remains available to
unrelated API code.

### `rag_release_builds`

```sql
id                        UUID PRIMARY KEY
kb_id                     TEXT NOT NULL
requested_alias           TEXT NOT NULL
status                    TEXT NOT NULL
catalog_digest            TEXT NOT NULL
build_config_digest       TEXT NOT NULL
retrieval_config_digest   TEXT NOT NULL
source_declaration_digest TEXT NOT NULL
source_snapshot_id        TEXT
release_id                TEXT
collection_name           TEXT
started_at                TIMESTAMPTZ NOT NULL
finished_at               TIMESTAMPTZ
error                     TEXT
details                   JSONB NOT NULL DEFAULT '{}'
```

Indexes: `(kb_id, started_at DESC)`, `status`, `release_id`.

### `rag_releases`

```sql
id                        TEXT PRIMARY KEY
kb_id                     TEXT NOT NULL
collection_name           TEXT NOT NULL UNIQUE
manifest_id               TEXT NOT NULL UNIQUE
manifest_path             TEXT NOT NULL
release_fingerprint       TEXT NOT NULL UNIQUE
catalog_digest            TEXT NOT NULL
build_config_digest       TEXT NOT NULL
source_declaration_digest TEXT NOT NULL
source_snapshot_id        TEXT NOT NULL
build_config              JSONB NOT NULL
source_manifest_digests   JSONB NOT NULL
source_adapter_versions   JSONB NOT NULL
document_count            INTEGER NOT NULL
chunk_count               INTEGER NOT NULL
created_at                TIMESTAMPTZ NOT NULL
retired_at                TIMESTAMPTZ
```

Indexes: `(kb_id, created_at DESC)`, `build_config_digest`,
`source_declaration_digest`, `source_snapshot_id`.

### `rag_alias_deployments`

```sql
id                       UUID PRIMARY KEY
kb_id                    TEXT NOT NULL
alias                    TEXT NOT NULL
release_id               TEXT NOT NULL REFERENCES rag_releases(id)
collection_name          TEXT NOT NULL
catalog_digest           TEXT NOT NULL
build_config_digest      TEXT NOT NULL
retrieval_config_digest  TEXT NOT NULL
retrieval_config         JSONB NOT NULL
status                   TEXT NOT NULL
created_at               TIMESTAMPTZ NOT NULL
applied_at               TIMESTAMPTZ
superseded_at            TIMESTAMPTZ
error                    TEXT
```

Indexes and constraints:

```sql
CREATE UNIQUE INDEX ... ON rag_alias_deployments (kb_id, alias)
WHERE status = 'active';

CREATE INDEX ... ON rag_alias_deployments (release_id);
CREATE INDEX ... ON rag_alias_deployments (kb_id, alias, created_at DESC);
```

### `eval_runs` Extensions

```sql
benchmark_execution_id   UUID
rag_release_id           TEXT REFERENCES rag_releases(id)
alias_deployment_id      UUID REFERENCES rag_alias_deployments(id)
build_config_digest      TEXT
retrieval_config_digest  TEXT
```

Every metric row produced by one benchmark invocation shares one
`benchmark_execution_id`. These columns are first-class rather than hidden in
`extra`; detailed benchmark-specific metadata remains in `extra`.

## Artifact Contract

Target layout:

```text
assets/rag_data/
  source_instances/
    <source_instance_id>/
      manifest.toml
      raw/
      extracted/
      chunks/
        <transformation_digest>/
          <document-artifact>.json
      benchmark/
        corpus.jsonl
        cases.jsonl
        labels.jsonl
        metadata.json

  knowledge_bases/
    <kb_id>/
      releases/
        <release_id>.json
```

Retire:

```text
knowledge_bases/<kb_id>/manifests/<collection>.json
knowledge_bases/<kb_id>/metadata/build_runs/<run_id>.json
```

Release manifests are immutable. Writing a different payload to an existing
release id is an error.

Raw and extracted caches remain reusable across alias build configurations.
Node artifacts are reusable only when the transformation digest matches.

## Provider Identity Contract

The current embedding client accepts a model name but ignores it, while the
service exposes one configured model. The current sparse client does not expose
model identity. The reranker factory also accepts a model name that the service
does not select.

For the first implementation, providers may remain single-model services, but
they must advertise identity:

```text
dense encoder: model id and vector dimension
sparse encoder: model id
reranker: model id
```

Add a provider capabilities endpoint or equivalent client methods and validate
catalog identity before build and alias activation. A mismatch is an external
configuration error, not a reason to substitute a runtime default.

Multi-model provider dispatch is a future extension. The catalog contract does
not need to change when it is added.

## Alias Diff

Command:

```bash
rag alias diff <kb-id> <alias>
```

Data flow:

```text
load catalog desired alias
  -> validate build/retrieve compatibility
  -> query provider identities
  -> compute desired digests
  -> load active AliasDeployment from Postgres
  -> load deployed RagRelease
  -> compare desired and applied digests
  -> find reusable releases
  -> emit AliasDiff
```

`alias diff` does not fetch remote source content by default. It detects source
manifest or adapter changes through `source_declaration_digest`. A future or
advanced `--refresh-sources` option may fetch content and compute a new source
snapshot, but this expensive behavior must remain explicit.

Exit status:

- `0`: no drift;
- `1`: valid drift exists;
- `2`: CLI usage or catalog validation error;
- `3`: ambiguous or protected-state conflict;
- `4`: provider, Postgres, or Qdrant failure.

## Alias Apply

Command:

```bash
rag alias apply <kb-id> <alias>
```

`apply` is the public verb because it may validate, reuse or build a release,
and activate desired state. `reconcile` is the internal service behavior.
`sync` is not used because it suggests unconditional copying and hides the
build and safety decisions.

Application service:

```python
class AliasService:
    def diff(self, request: AliasDiffRequest) -> AliasDiff: ...
    def apply(self, request: AliasApplyRequest) -> AliasApplyResult: ...
```

Reconciliation cases:

| Desired versus applied state | Action |
| --- | --- |
| No drift | Return the active deployment without mutation. |
| Retrieval-only drift | Create and activate a deployment using the same release. |
| Build/source drift with one reusable release | Reuse it and activate a deployment. |
| Build/source drift with no reusable release on a non-default alias | Build a release, then activate it. |
| Build/source drift with no reusable release on the default alias | Refuse; do not silently build production. |
| Multiple matching releases | Refuse unless `--release <id>` disambiguates. |
| Explicit release does not match desired build/source state | Refuse. |
| Retrieval strategy incompatible with release encoders | Refuse. |

The default-alias restriction is based on `knowledge_base.default_alias`; no
hardcoded `champion` or `challenger` policy exists.

For the default alias, a reusable release is eligible only when evaluation
coverage exists for the same release id and desired retrieval configuration
digest. An explicit `--release` resolves ambiguity but does not bypass
compatibility or evaluation checks.

If the KB has no attached benchmarks, default-alias apply refuses unless the
operator supplies `--allow-unevaluated`. The deployment row records that
override. This is an exceptional bootstrap/emergency action, not the normal
workflow.

Initial bootstrap uses a non-default alias first. If a KB intentionally has
only its default alias, an explicit `--allow-build-default` escape hatch is
required and recorded in the build attempt details. It is not part of the
normal runbook.

### Release Build Data Flow

```text
desired alias build config
  -> validate providers
  -> create ReleaseBuildAttempt(status=running)
  -> resolve all corpus source instances for KB
  -> fetch/extract using existing caches
  -> parse nodes under transformation_digest cache
  -> compute source snapshot id
  -> compute release fingerprint/id
  -> take a Postgres advisory lock derived from the full release fingerprint
  -> reuse ready release if it appeared concurrently
  -> materialize LlamaIndex Qdrant collection
  -> write immutable release manifest
  -> write Qdrant attestation v2
  -> insert RagRelease
  -> complete ReleaseBuildAttempt
```

The advisory lock is held only around release lookup/materialization and is
released when the transaction/session ends. Database uniqueness on the full
release fingerprint remains the final collision guard.

Build errors mark the attempt failed and delete an incomplete collection and
release manifest. Existing source caches remain available for retry.

### Applied-State Activation

Postgres is the serving source of truth; Qdrant alias movement is a mirror.
This avoids pretending a Postgres transaction and a Qdrant alias update are one
distributed transaction.

Apply sequence:

```text
create pending AliasDeployment
  -> validate release attestation
  -> update Qdrant alias mirror
  -> in one Postgres transaction:
       supersede previous active deployment
       activate pending deployment
```

If Qdrant movement fails, the pending deployment becomes failed and runtime
continues using the old active deployment.

If Qdrant succeeds but Postgres activation fails, runtime still uses the old
deployment's physical collection directly. Re-running apply repairs the mirror
and activation. No request depends on the Qdrant alias during this interval.

## Runtime Resolution

Current runtime resolution must change from:

```text
current catalog retrieval config + Qdrant alias target
```

to:

```text
catalog task/KB/default-alias routing
  -> active AliasDeployment from Postgres
  -> applied retrieval config from deployment
  -> RagRelease and collection attestation
  -> physical collection opened directly
```

Runtime continues to use the deployed catalog structure for valid task, KB,
and alias names and for `default_alias`. It does not use current desired
`alias.build` or `alias.retrieve` values for serving.

Runtime validation:

- active deployment exists;
- release exists and is not retired;
- deployment collection matches release collection;
- Qdrant collection exists;
- attestation release id, manifest id, KB, collection, and fingerprints match;
- live provider model identities match release encoder identities;
- applied retrieval strategy is compatible with release capability.

Retriever caches are keyed by `AliasDeployment.id`, not Qdrant alias name.
Runtime looks up the active deployment before reusing a cached retriever, so a
new deployment becomes visible without process restart.

## Benchmark Workflow

Commands:

```bash
rag benchmark run --kb <kb-id> --alias <alias>
rag benchmark run <benchmark-source-instance-id> --alias <alias>
```

Exactly one benchmark target is selected:

- `--kb` runs every benchmark source instance attached to the KB;
- a positional benchmark source instance runs that benchmark only.

The alias is always explicit.

Preparation flow:

```text
benchmark source manifest + adapter id/version
  -> compute desired preparation digest
  -> compare benchmark/metadata.json
  -> reuse valid prepared artifacts or regenerate them
```

Execution flow:

```text
load active AliasDeployment
  -> load exact RagRelease and applied retrieval config
  -> prepare benchmark artifacts if stale
  -> for live-KB benchmarks: query release collection directly
  -> for benchmark-owned corpora:
       build disposable collection using release.build_config
       apply deployment.retrieval_config
       delete disposable collection in finally
  -> write eval_runs/eval_samples
```

Every result records:

- benchmark execution id;
- release id and release fingerprint;
- alias deployment id;
- build and retrieval config digests;
- physical collection and manifest id;
- benchmark adapter and artifact digests;
- prompt and judge identity when applicable.

The default-alias apply policy considers a release/retrieval combination
evaluated when every attached benchmark has a completed execution for the same
release id and retrieval config digest. It refuses if any required execution
has `eval_verdict = 'fail'`. `unscored` metrics remain a manual operator review;
editing the default alias desired state and running apply is the explicit
approval action.

Automatic metric thresholds and fully automatic quality promotion remain out
of scope.

## Typer CLI

Add an explicit Typer dependency to every environment that executes the RAG
CLI. Do not rely on the current transitive dependency through DVC/GTO. At the
time of this plan, the root lock contains Typer 0.21.1 while generated Docker
locks contain other versions; declaring it directly and regenerating locks is
required.

Add:

```toml
[project.scripts]
rag = "rag.cli.app:main"
```

Target command tree:

```text
rag
  catalog
    validate
  alias
    diff
    apply
    status
  release
    list
    show
  benchmark
    run
    list
    show
  source
    inspect       # expert diagnostics only
    rebuild       # explicit cache invalidation only
```

Primary commands:

```bash
rag catalog validate
rag alias diff pytorch_reference challenger
rag alias apply pytorch_reference challenger
rag benchmark run --kb pytorch_reference --alias challenger
rag alias apply pytorch_reference champion
rag alias status pytorch_reference
rag release list --kb pytorch_reference
rag release show <release-id>
```

Global defaults:

- catalog path from `CONFIG__CATALOG_PATH`;
- artifact root from a new runtime setting, default `assets/rag_data`;
- Postgres, Qdrant, and provider locations from runtime settings;
- generated build attempt and deployment ids;
- JSON output when stdout is not a TTY;
- human-readable Rich tables when stdout is a TTY.

Global overrides are available before the command group:

```bash
rag --catalog <path> --data-root <path> --output json alias diff <kb> <alias>
```

These are diagnostic/development overrides and are absent from the normal
operator runbook.

Typer command functions must:

- contain no build or reconciliation logic;
- construct Pydantic request objects;
- resolve an application-service factory lazily;
- render the returned Pydantic result;
- emit logs to stderr and result data to stdout;
- avoid interactive prompts;
- map domain errors to stable exit statuses.

Non-interactive shell callers always request JSON output. Airflow calls the
application service directly. Tests use `typer.testing.CliRunner` and inject
application services rather than real providers.

## Target Source Structure

```text
src/rag/
  cli/
    app.py                 # root Typer app and global options
    alias.py               # alias command group
    benchmark.py           # benchmark command group
    catalog.py             # catalog command group
    release.py             # release command group
    source.py              # expert source diagnostics
    output.py              # JSON/table rendering and exit mapping

  control_plane/
    models.py              # release/build/deployment/diff contracts
    fingerprints.py        # canonical digest and identity functions
    repositories.py        # repository protocols
    postgres.py            # SQLAlchemy repository implementations
    release_builder.py     # complete immutable release build
    alias_service.py       # diff/apply reconciliation
    provider_validation.py # encoder/reranker identity checks

  sources/
    ...                    # existing adapters and processing mechanics

  indexing/
    materialize.py         # materialize a ready release, no promotion policy
    llamaindex_qdrant.py   # collection and Qdrant alias primitives

  runtime/
    resolver.py            # active deployment + release resolution
    service.py

  evaluation/
    models.py
    runner.py
    target.py
```

`src/rag/control_plane/` owns desired/applied reconciliation. It contains no
dataset-specific adapters and no LlamaIndex document/node replacements.

Retire after migration:

```text
src/rag/sources/cli.py
src/rag/evaluation/cli.py
src/rag/lifecycle/models.py
src/rag/lifecycle/commands.py
```

Low-level Qdrant alias update primitives remain in indexing code but are not
operator-facing promotion APIs.

## Airflow And Cleanup

Replace the current stage-oriented `rag_lifecycle` DAG with an alias apply DAG.
Its normal parameters are:

```text
kb_id
alias
refresh_sources = false
sync_dvc = false
```

The DAG calls `AliasService.apply()` directly. It does not shell through the
Typer CLI. CLI and Airflow share the same Pydantic request and result contracts.

Keep benchmark execution manually triggered for now. A future DAG can call the
same benchmark application service.

Update `rag_collection_cleanup.py`:

- protect every collection referenced by an active deployment;
- protect collections used by running build attempts;
- optionally retain the newest N superseded deployments per alias;
- never rely only on Qdrant aliases to determine liveness;
- mark a release retired before deleting its physical collection;
- retain immutable release manifests after retirement.

Update `scripts/rag_ops.sh` so operators run:

```bash
bash current/scripts/rag_ops.sh alias diff pytorch_reference challenger
bash current/scripts/rag_ops.sh alias apply pytorch_reference challenger
```

The wrapper invokes the installed `rag` entrypoint inside `rag-ops`; operators
do not pass `python -m ...`.

## Migration Policy

There is no dual-stack compatibility period.

Migration steps:

1. Deploy database tables and `eval_runs` column migration.
2. Deploy catalog schema version 4 and provider identity endpoints together.
3. Remove old source/evaluation CLI entrypoints and lifecycle DAG in the same
   release that adds the Typer CLI and alias apply DAG.
4. Delete old build-run JSON artifacts.
5. Rebuild old attestation-v1 collections through non-default alias apply.
6. Apply the evaluated release to the default alias.
7. Remove any remaining old collection manifests after confirming release
   manifests and database rows.

Runtime rejects:

- attestation schema version 1;
- an alias with no active deployment row;
- deployments whose release or collection metadata does not match;
- catalog schema version 3.

## Implementation Phases

### Phase 0: Contract And Dependency Baseline

Files:

- `pyproject.toml`;
- `uv.lock` and Docker lock files;
- `src/rag/control_plane/models.py`;
- `src/rag/control_plane/fingerprints.py`;
- contract tests under `tests/rag/`.

Work:

1. Add direct Typer dependencies to `rag`, `airflow-worker`, and
   `airflow-worker-gpu` execution surfaces.
2. Add the release, build-attempt, deployment, and diff Pydantic contracts.
3. Add canonical digest helpers and deterministic release naming.
4. Add tests proving formatting-independent canonical digests and identity
   changes for every semantic input.
5. Add no CLI yet.

Acceptance:

- all new contracts forbid unknown fields;
- digest tests cover build, retrieval, source declaration, transformation,
  source snapshot, and release fingerprints;
- no existing runtime behavior changes;
- all lock files resolve one supported Typer constraint.

### Phase 1: Catalog Version 4 And Provider Identity

Files:

- `catalog.toml`;
- `src/app_config/catalog/schema.py`;
- `src/app_config/catalog/models.py`;
- `src/app_config/catalog/loader.py`;
- `src/app_config/catalog/validation.py`;
- `src/app_config/runtime/models.py`;
- embedding and reranker service/client modules;
- catalog and provider contract tests.

Work:

1. Implement `AliasBuildConfig`, symmetric encoder configs, chunking config,
   and `AliasRetrievalConfig`.
2. Migrate catalog aliases to nested `build` and `retrieve` blocks.
3. Increment catalog schema version to 4 and reject version 3.
4. Remove model identity and device fields from RAG runtime settings after
   moving semantic identity to the catalog.
5. Add provider identity reporting for dense, sparse, and reranker services.
6. Validate catalog encoder and reranker identity against providers without
   silently substituting defaults.
7. Update runtime `AliasConfig` consumers to compile against the nested schema,
   while still using desired catalog values temporarily until Phase 4.

Acceptance:

- catalog validation enforces every compatibility rule;
- existing catalog samples and tests use schema version 4 only;
- provider mismatch tests fail before indexing;
- dense, sparse, and hybrid alias configurations are covered;
- no model-selection argument remains accepted and ignored by clients.

### Phase 2: Configuration-Aware Artifacts And Immutable Releases

Files:

- `src/rag/sources/chunks.py`;
- `src/rag/sources/bundles.py`;
- `src/rag/sources/build.py`;
- `src/rag/indexing/materialize.py`;
- `src/rag/contracts/models.py`;
- `src/rag/contracts/manifests.py`;
- `src/rag/control_plane/release_builder.py`;
- artifact/materialization/release tests.

Work:

1. Scope node artifacts by transformation digest.
2. Pass resolved alias build configuration through source parsing and
   materialization; remove CLI chunking overrides from the normal path.
3. Replace `IndexManifest` with immutable release manifests.
4. Upgrade Qdrant attestation to schema version 2.
5. Implement release build cleanup and idempotent concurrent release reuse.
6. Remove alias and benchmark fields from materialization results.
7. Keep raw/extracted caches reusable across build configurations.

Acceptance:

- two aliases with different chunking produce isolated node artifacts;
- identical build and source snapshots reuse one release;
- differing semantic inputs produce different release ids and collections;
- incomplete collections and manifests are cleaned after failure;
- release manifests cannot be overwritten with changed content;
- attestation v1 is rejected.

### Phase 3: Postgres Control-Plane Registry

Files:

- `src/clients/db/models.py`;
- new SQL files under `migrations/postgres/`;
- new `src/clients/db/urls.py`;
- new `scripts/apply_agent042_db_migrations.sh`;
- `src/rag/control_plane/repositories.py`;
- `src/rag/control_plane/postgres.py`;
- repository integration tests.

Work:

1. Add build, release, and deployment ORM models and SQL.
2. Add `eval_runs` release/deployment/execution columns.
3. Implement repository protocols and SQLAlchemy implementations.
4. Implement active-deployment uniqueness and append-only history.
5. Add transaction tests for pending, active, superseded, and failed states.
6. Add the explicit idempotent deployment migration command because
   `create_all()` does not alter existing tables.
7. Centralize sync/async Postgres URL normalization used by the control plane
   and evaluation writer.

Acceptance:

- only one active deployment can exist per KB alias;
- release rows are immutable;
- failed attempts do not create release rows;
- repository tests run against Postgres-compatible behavior;
- existing eval rows remain readable after migration.

### Phase 4: Alias Diff And Apply Service

Files:

- `src/rag/control_plane/alias_service.py`;
- `src/rag/control_plane/provider_validation.py`;
- `src/rag/indexing/llamaindex_qdrant.py`;
- service tests with fake repositories/Qdrant/providers;
- real-service smoke test profile.

Work:

1. Implement desired-state resolution and alias diff.
2. Implement release matching and ambiguity handling.
3. Implement default-alias protection and explicit bootstrap overrides. Until
   Phase 6 provides evaluation coverage queries, non-bootstrap application of
   new default-alias desired state remains disabled.
4. Implement release build invocation for non-default aliases.
5. Implement pending deployment, Qdrant mirror update, and Postgres activation.
6. Make retries repair partial Qdrant/DB state safely.
7. Remove `promote_materialized_alias` as a control-plane operation; retain a
   low-level alias update primitive.

Acceptance:

- no-drift apply is idempotent;
- retrieval-only changes do not build a collection;
- matching releases are reused;
- default alias never silently builds and cannot activate new desired state
  before evaluation integration exists;
- ambiguous release selection fails clearly;
- Qdrant and Postgres failure injection preserves the previous active runtime
  deployment;
- a repeated apply repairs partial state.

### Phase 5: Applied-State Runtime

Files:

- `src/rag/runtime/resolver.py`;
- `src/rag/runtime/service.py`;
- gateway RAG construction and startup validation;
- runtime and gateway tests.

Work:

1. Resolve active deployments and releases from Postgres.
2. Open physical collections directly instead of resolving the Qdrant alias as
   the serving target.
3. Use snapshotted applied retrieval configuration, never current desired
   alias retrieval values.
4. Cache retrievers by deployment id.
5. Validate release attestation and provider identity.
6. Make database absence a startup error when RAG is enabled.
7. Update strict startup checks to validate active deployments.

Acceptance:

- editing the catalog without apply has no serving effect;
- applying retrieval-only drift becomes visible without gateway restart;
- switching releases invalidates only the affected retriever cache;
- runtime continues serving the previous deployment during failed apply;
- Qdrant alias mirror drift does not redirect serving traffic;
- missing or inconsistent applied state follows strict/non-strict runtime
  policy explicitly.

### Phase 6: Release-Aware Benchmark Service

Files:

- `src/rag/evaluation/runner.py`;
- `src/rag/evaluation/target.py`;
- `src/rag/sources/benchmark_prep.py`;
- `src/clients/db/eval_writer.py`;
- evaluation and DB tests.

Work:

1. Add automatic benchmark preparation staleness checks.
2. Resolve active alias deployment and release before execution.
3. Use the release build config for disposable benchmark corpora.
4. Use applied retrieval config for evaluation.
5. Generate one benchmark execution id shared by metric rows.
6. Persist release, deployment, build, and retrieval identities.
7. Implement evaluation-coverage queries used by protected default-alias
   apply.
8. Keep judge construction lazy by suite.
9. Enable normal evaluated default-alias apply after the coverage query is in
   place; retain explicit bootstrap/emergency overrides in deployment history.

Acceptance:

- operators do not run a separate prepare command in the normal workflow;
- benchmark-owned corpora exactly mirror the release build config;
- every metric row can be joined to one execution, release, and deployment;
- benchmark execution still requires an explicit alias;
- temporary collections are deleted on success and failure;
- protected apply can distinguish evaluated, failed, incomplete, and unscored
  coverage.

### Phase 7: Unified Typer CLI

Files:

- new `src/rag/cli/` package;
- `pyproject.toml` console script;
- `scripts/rag_ops.sh`;
- CLI tests using `CliRunner`.

Work:

1. Implement root context and lazy service factories.
2. Implement catalog, alias, release, benchmark, and expert source groups.
3. Implement JSON and Rich table renderers.
4. Implement stable error and exit-code mapping.
5. Change the server wrapper to invoke `rag` directly.
6. Verify `rag --help` works without connecting to external services.
7. Verify JSON stdout contains no logs or progress text.

Acceptance:

- normal commands need only meaningful resource identities;
- `rag alias apply` and `rag benchmark run` expose nested help;
- shell completion can be generated by Typer;
- automation-compatible JSON output is deterministic;
- no command contains application logic;
- all CLI behavior is covered with injected services.

### Phase 8: Airflow, Cleanup, Removal, And Documentation

Files:

- `dags/rag_lifecycle.py`;
- `dags/rag_collection_cleanup.py`;
- related DAG tests;
- old CLI/lifecycle modules and tests;
- `README.md`;
- `docs/architecture/system-design.md`;
- `docs/operations/rag-operations.md`;
- `docs/analytics/evaluation-results.md`;
- `docs/index.md`;
- Compose examples.

Work:

1. Replace the lifecycle DAG with alias apply orchestration.
2. Make cleanup deployment/release-aware.
3. Delete the old source and evaluation CLI entrypoints.
4. Delete `BuildRun`, `BuildRequest`, stage wrappers, status/show commands, and
   build-run artifact paths.
5. Delete promotion terminology and `PromotionDecision` from active code/docs.
6. Remove `collect-bundle` from the operator surface; retain bundle functions as
   internal diagnostics.
7. Rewrite the operator runbook around catalog edit, diff, apply, benchmark,
   and default-alias apply.
8. Update architecture and analytics documentation with desired/applied state
   and DB contracts.
9. Remove old collections and artifacts after final smoke tests.

Acceptance:

- repository search finds no old lifecycle command names outside historical
  planning context;
- the canonical runbook contains no manual collection or run-id handling;
- Airflow and CLI call the same application services;
- cleanup cannot remove active release collections;
- fresh deployment and default-alias bootstrap are documented;
- rollback is documented as editing desired alias state to a previous release
  configuration and applying an unambiguous matching release.

## End-State Operator Workflow

Candidate change:

```text
edit challenger.build and challenger.retrieve in catalog.toml
  -> rag catalog validate
  -> rag alias diff <kb> challenger
  -> rag alias apply <kb> challenger
  -> rag benchmark run --kb <kb> --alias challenger
```

Default-alias change:

```text
copy the accepted challenger build/retrieve state to the default alias
  -> review and deploy the catalog change
  -> rag alias diff <kb> <default-alias>
  -> rag alias apply <kb> <default-alias>
```

The second apply reuses the exact evaluated release. It does not rebuild it.

Retrieval-only tuning follows the same workflow, but apply creates only a new
deployment record and reuses the physical release.

## End-State Acceptance Checklist

- The catalog completely declares each alias's build and retrieval semantics.
- Runtime settings contain provider location and execution settings only.
- Editing alias build/retrieve configuration cannot change live retrieval
  before apply.
- Alias diff reports exact build, retrieval, source, provider, and deployment
  drift.
- Alias apply is idempotent and repairable.
- Releases are immutable, content-identified, and reusable across aliases.
- Default-alias apply cannot silently build an unevaluated release.
- Runtime resolves applied state from Postgres and validates Qdrant
  attestation.
- Benchmarks persist release and deployment identity and prepare stale inputs
  automatically.
- Chunk/node caches are transformation-config aware.
- Operators never provide collection names, build-run ids, persistence flags,
  catalog paths, or data roots in normal commands.
- The primary interface is the nested Typer `rag` command.
- Airflow calls application services directly.
- Old stage-oriented lifecycle and promotion concepts are removed rather than
  maintained as a second workflow.
