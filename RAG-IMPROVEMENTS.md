# RAG Improvements: Alias-Based Lifecycle & Experimentation

## Overview

This document describes the design for alias-based lifecycle management of RAG collections
in Qdrant, enabling safe comparison of retrieval architectures without full production releases,
automated data freshness pipelines, and a clear experimentation workflow.

### Key concepts

| Term | Definition |
|---|---|
| **Knowledge Base (KB)** | A user-facing logical dataset (e.g., `arxiv`, `pytorch_docs`). Shown in the UI. |
| **Collection** | A physical Qdrant collection containing indexed vectors. Named `{kb_name}_{timestamp}` (e.g., `pytorch_docs_20260314_120000`). |
| **Alias** | A Qdrant alias pointing to a collection. Named `{kb_name}_{role}` (e.g., `arxiv_champion`). |
| **Role** | A lifecycle label: `champion` (production), `challenger` (experiment), or `{role}_staging` (build in progress). |
| **Collection metadata** | A sentinel point (`id="_meta"`) inside each collection that stores the build parameters used to create it. This is the single source of truth for how a collection was built. |
| **Update strategy** | How a KB's data is refreshed: `incremental` (upsert new data into existing collections) or `replace` (build a new collection and swap the alias). |

### Design goals

1. Production always serves from `champion` aliases. If a champion alias is missing, the KB is
   unavailable — no silent fallbacks.
2. Experiments happen on `challenger` aliases. They are fully manual and never touched by
   automated DAGs.
3. Data freshness DAGs only rebuild `champion`. They read build parameters from the collection
   itself (no external config files for build params).
4. No config drift: the collection metadata point is the canonical record of how a collection was
   built. There are no YAML/JSON files storing build parameters.
5. Global RAG parameters (`top_k`, `score_threshold`, `reranking`) are system-wide settings,
   not per-alias. Experimenting with these requires deploying a new configuration.

---

## 1. Naming conventions

**Aliases** (globally unique in Qdrant):

| Pattern | Example | Purpose |
|---|---|---|
| `{kb}_{role}` | `arxiv_champion` | Production or experiment alias |
| `{kb}_{role}_staging` | `pytorch_docs_champion_staging` | Temporary alias during index rebuild |

**Collections**:

| Pattern | Example | Purpose |
|---|---|---|
| `{kb}_{timestamp}` | `pytorch_docs_20260314_120000` | Physical collection. Timestamp = build time (`%Y%m%d_%H%M%S`). |

Legacy collections (`chat_documents`, `code_documents`) will be migrated during rollout and
added to a skip-list for the cleanup DAG.

---

## 2. Configuration

The system uses two separate configs with clearly different audiences.

### 2.1 Runtime config — Knowledge Bases registry

**File**: `src/shared/knowledge_bases.json`
**Used by**: gateway, eval runner, production `src/rag/ops` workflows, and notebook wrappers.
**Purpose**: defines which KBs exist, which alias roles are valid, and the update strategy.

```json
[
    {
        "knowledge_base": "arxiv",
        "aliases": ["champion", "challenger"],
        "update_strategy": "incremental"
    },
    {
        "knowledge_base": "pytorch_docs",
        "aliases": ["champion", "challenger"],
        "update_strategy": "replace"
    }
]
```

Loaded into Pydantic `Settings` via the env var `GATEWAY_KNOWLEDGE_BASES_PATH`
(default: bundled `src/shared/knowledge_bases.json`).

**Rules**:
- If a KB has no `champion` in its `aliases` list, it is unavailable in production (UI). This is
  not a startup error — the KB may be in preparation. API calls with an explicit alias
  (e.g., `challenger`) still work if that alias exists in Qdrant.
- The gateway does not know or care about build parameters (chunking, embedding model, etc.).
  It only resolves aliases and queries Qdrant.
- `update_strategy` is a property of the KB itself, not of a build experiment. It determines
  DAG behavior: `incremental` = upsert into existing collections; `replace` = build new
  collection and swap alias.

### 2.2 Build config — Collection metadata (no config files)

There is **no YAML or JSON file** storing build parameters. Instead, each collection stores its
own build configuration as a sentinel point in Qdrant.

When a production create workflow creates a collection, it writes a metadata point:

```python
vector_store.upsert(
    collection_name=collection_name,
    points=[PointStruct(
        id="_meta",
        vector=[0.0] * dimension,       # dummy vector, excluded from search
        payload={
            "type": "collection_meta",
            "build_config": {
                "chunking_strategy": "fixed_token",
                "chunk_size": 512,
                "chunk_overlap": 64,
                "embedding_model": "intfloat/e5-base-v2",
            },
            "kb_name": "pytorch_docs",
            "created_at": "2026-03-14T12:00:00Z",
        },
    )],
)
```

**To exclude the sentinel from search results**, add a filter to `search()`:

```python
must_not=[FieldCondition(key="type", match=MatchValue(value="collection_meta"))]
```

**Who reads the metadata**:
- **DAGs** (data freshness): read `_meta` from the current champion collection to know how to
  rebuild with identical parameters.
- **Eval runner**: reads `_meta` to log build config into `eval_runs` (chunking_strategy,
  embedding_model, etc.).
- **Admin/debugging**: `GET /collections/{name}/points/_meta` for inspection.

**Who does NOT read the metadata**:
- **Gateway**: never reads `_meta`. It only resolves aliases and queries.
- **UI**: only knows about KB names.

---

## 3. API contract

### 3.1 Chat completions request

The current `ChatCompletionRequest` field `knowledge_base: str | None` is replaced with a
structured field supporting multiple KBs with explicit alias selection:

```python
class RAGSource(BaseModel):
    knowledge_base: str
    alias: str = "champion"

class ChatCompletionRequest(BaseModel):
    # ... existing fields ...
    rag_sources: list[RAGSource] | None = Field(
        default=None,
        description="Knowledge bases for RAG retrieval. None = RAG disabled.",
    )
```

**Example request body**:

```json
{
    "messages": [{"role": "user", "content": "Explain attention mechanism"}],
    "rag_sources": [
        {"knowledge_base": "arxiv", "alias": "champion"},
        {"knowledge_base": "pytorch_docs", "alias": "challenger"}
    ]
}
```

**Defaults**:
- `rag_sources: null` → RAG disabled.
- `alias` defaults to `"champion"` if omitted.

**Multi-KB result merging**: each collection returns its own `top_k` results. Results are merged globally and are ALL used by the base LLM. This is a simplification for the MVP. In the future the final reranker will handle this task.

### 3.2 Error handling

| Condition | UI behavior | API response |
|---|---|---|
| KB name not in `KNOWLEDGE_BASES` config | Error message: "Knowledge base unavailable" | `404` with error detail |
| Alias not in KB's allowed aliases list | Error message: "Knowledge base unavailable" | `404` with error detail |
| Alias exists in config but not in Qdrant | Error message: "Knowledge base unavailable" | `404` with error detail |

No silent fallbacks. If any requested KB+alias pair is unavailable, the request fails for that
source and the error is surfaced to the caller.

### 3.3 Admin endpoints

| Endpoint | Purpose |
|---|---|
| `GET /v1/knowledge-bases` | List available KBs, their aliases, and which aliases currently resolve in Qdrant. |

Alias promotion is performed only via notebooks / Python APIs (`experiments.rag.notebook_ops`
and `rag.ops.aliases`) or automatically within DAGs.
There is no promotion endpoint in the API.

---

## 4. Resolution logic

The gateway resolves a `(knowledge_base, alias)` pair to a Qdrant alias name using a simple
concatenation: `{kb_name}_{alias}` → e.g., `arxiv_champion`.

Full resolution flow per request:

```
1. For each entry in rag_sources:
   a. Validate kb_name exists in KNOWLEDGE_BASES config.
   b. Validate alias is in the KB's allowed aliases list.
   c. Construct Qdrant alias name: f"{kb_name}_{alias}"
   d. Create QdrantVectorStore(collection_name=qdrant_alias_name)
   e. Check collection_exists() — if False, return 404.
   f. Execute search(query_embedding, top_k, score_threshold)
2. Merge results from all sources, pass all to prompt builder
```

**No caching**: aliases are resolved on every query. This is simple and ensures promotions take
effect immediately without cache invalidation logic.

---

## 5. Operator entrypoints

Production-safe RAG lifecycle code lives under `src/rag/ops/`.
`experiments/rag/rag_ops.ipynb` does not implement a second runtime; it imports
`experiments.rag.notebook_ops`, which is a convenience layer over the same production entrypoints.

| Workflow | Production entrypoints | Notebook wrappers | Typical caller |
|---|---|---|---|
| Create fresh collection | `create_arxiv_collection`, `create_pytorch_docs_collection` | `create_arxiv`, `create_pytorch_docs` | bootstrap, challenger builds |
| Refresh existing alias from `_meta` | `update_arxiv_collection`, `update_pytorch_docs_collection` | `refresh_arxiv`, `refresh_pytorch_docs` | Airflow DAGs, manual repair |
| Alias management | `assign_alias_to_collection`, `promote_alias`, `detach_alias` | `assign_alias`, `promote`, `detach` | operator notebook |
| Inspection | `inspect_alias`, `inspect_collection`, `list_alias_mappings` | `inspect_kb_alias`, `inspect_existing_collection`, `list_aliases` | operator notebook, debugging |

### 5.1 Automated data freshness (DAG-driven, automated)

**Purpose**: keep production data current (new arXiv papers, updated PyTorch docs).
**Scope**: production DAGs target only `champion`.
**Trigger**: scheduled Airflow DAGs.

#### Incremental strategy (ArXiv)

DAG: `arxiv_rag_update` (daily).

1. Download new data (arXiv papers).
2. Version with DVC.
3. Call `update_arxiv_collection(kb="arxiv", alias="champion")`.
4. Resolve `arxiv_champion` and read `_meta` from the target collection.
5. Reconstruct chunking and embeddings settings from stored build config.
6. Upsert refreshed data into the same collection.

The DAG does not touch challenger collections. If a non-production alias needs a refresh, an
operator runs it manually from the notebook path.

#### Replace strategy (PyTorch docs)

DAG: `pytorch_docs_rag_update` (weekly).

1. Call `update_pytorch_docs_collection(kb="pytorch_docs", alias="champion")`.
2. Resolve `pytorch_docs_champion` and read `_meta` from the current champion collection.
3. Create a successor collection `pytorch_docs_{timestamp}` with fresh `_meta`.
4. Point `pytorch_docs_champion_staging` at the successor collection.
5. Build the new index into the successor collection.
6. Atomically swap `pytorch_docs_champion` to the successor collection.

### 5.2 Manual notebook operations (human-driven, manual)

**Purpose**: bootstrap empty clusters, build challengers, inspect state, or manually repair alias
mappings.
**Scope**: notebook path can create, refresh, assign, promote, detach, and inspect collections.
**Trigger**: manual notebook invocation from `experiments/rag/rag_ops.ipynb`.

Typical notebook flow:

1. **Create** a challenger or first champion collection:
   ```python
   from experiments.rag.notebook_ops import create_pytorch_docs

   create_pytorch_docs(
      chunking_strategy="section_aware",
      chunk_size=1024,
      chunk_overlap=128,
      alias="challenger",
   )
   ```
2. **Evaluate** champion vs challenger via the eval runner.
3. **Promote** or reattach aliases with `promote(...)` / `assign_alias(...)` / `detach(...)`.
4. **Inspect** live alias mappings and `_meta` with `inspect_kb_alias(...)` or `list_aliases()`.

Because the notebook wrappers call `src/rag/ops`, notebook experiments and DAGs stay aligned on
the same production runtime.

### 5.3 Sandbox boundary

- Notebook-only experimental retrieval code lives under `experiments/rag/sandboxes/`.
- Production services, Airflow DAGs, and production evals must never import from that directory.
- If a sandbox experiment is worth promoting, port the implementation into `src/rag` or
  `src/rag/ops` first, then rebuild and promote a collection using the production entrypoints.

---

## 6. Collection management and cleanup

There is no dedicated `manage_rag.py` operator CLI anymore. Alias operations are performed from the
notebook path (`experiments.rag.notebook_ops`) or directly through the Python APIs in `rag.ops`.

### 6.1 Orphan collection cleanup

A daily Airflow DAG (`rag_collection_cleanup`) handles garbage collection:

1. List all Qdrant collections.
2. List all Qdrant aliases → build a set of collections that have at least one alias.
3. For each collection **not** in that set:
   - Parse the timestamp from its name (`{kb_name}_{timestamp}`).
   - If older than 7 days → delete.
4. Legacy collections not following the naming convention are on an explicit skip-list.

---

## 7. Eval pipeline integration

Sections 7-9 are rollout and future-state notes. The current live operator path is defined in
sections 5-6: production entrypoints in `src/rag/ops`, notebook wrappers in
`experiments/rag/notebook_ops.py`, and notebook-only experimental forks in
`experiments/rag/sandboxes/`.

### 7.1 Request routing

| Caller | What is sent | Alias resolution |
|---|---|---|
| **UI** | `knowledge_base` only (alias defaults to `champion`) | Gateway resolves `{kb}_champion` |
| **Eval runner (generation)** | `knowledge_base` + explicit `alias` | Gateway resolves `{kb}_{alias}` |
| **Eval runner (retrieval-only)** | `kb_name` + `rag_alias` | Resolves `{kb}_{alias}` → reads `_meta` → builds temp collection |

**Generation evals** (Chat, Summarization, Code, RAG+Chat, RAG+Code) call the **gateway API**.
The gateway is the single source of truth for alias resolution, RAG retrieval, and inference.

**Retrieval-only evals** use the production `src/rag` library directly (not the gateway) to build temporary
benchmark collections replicating a production collection's build config. See README-EVAL.md
Section 2.5 for details.

### 7.2 Eval runs table

The `eval_runs` table must include a `rag_alias` column:

```sql
ALTER TABLE eval_runs ADD COLUMN rag_alias TEXT;
```

The eval runner also reads `_meta` from the target collection to populate `chunking_strategy`,
`embedding_model`, `chunk_size`, `chunk_overlap` in the eval run record.

### 7.3 Gating metric for auto-promotion

| Metric | Dataset | Used for |
|---|---|---|
| nDCG@10 | BEIR-SciFact | Single gating metric for DAG auto-promotion |

Thresholds:
- **-5% to +20%**: auto-promote (normal variance and improvements).
- **> +20%**: anomaly, log and hold (suspicious improvement, needs manual review).
- **< -5%**: regression, log and hold (do not promote).

---

## 8. Current state → target state

Summary of what changes from the current codebase:

| Component | Current state | Target state |
|---|---|---|
| `shared/config.py` `KNOWLEDGE_BASES` | Hardcoded dict mapping KB → single collection name | Loaded from `src/shared/knowledge_bases.json` with aliases list and update strategy |
| `ChatCompletionRequest` | `knowledge_base: str \| None` | `rag_sources: list[RAGSource] \| None` |
| `RAGService._get_retriever()` | `KNOWLEDGE_BASES[kb]["collection"]` → single vector store | `(kb, alias)` → `{kb}_{alias}` → vector store, resolved per query |
| `QdrantVectorStore.search()` | No metadata exclusion filter | Add `must_not` filter for `type=collection_meta` |
| ArXiv refresh path | Champion refresh via `rag.ops.update.update_arxiv_collection()` | Champion-only `_meta`-driven incremental refresh |
| PyTorch docs refresh path | Champion refresh via `rag.ops.update.update_pytorch_docs_collection()` | Champion-only successor build via staging alias |
| Airflow DAGs | Build + swap in one step | Build → staging → eval → conditional promote |
| Build parameters | Passed ad hoc by mixed scripts | Stored in collection `_meta` point; supplied only on create via `src/rag/ops/create/*` or notebook wrappers |
| Eval runner | No alias awareness | Sends `rag_alias` in API request; logs alias + build config in `eval_runs` |
| Admin | No KB discovery endpoint | `GET /v1/knowledge-bases` |
| Cleanup | Manual | Daily DAG: delete orphan collections older than 7 days |

---

## 9. Implementation order

Suggested sequence for implementation:

1. **Collection metadata point**: add `_meta` write to production create workflows, add `must_not`
   filter to `QdrantVectorStore.search()`.
2. **Runtime config**: create `src/shared/knowledge_bases.json`, add Pydantic model and loader to
   `shared/config.py`, replace `KNOWLEDGE_BASES` dict.
3. **API contract**: add `RAGSource` schema, update `ChatCompletionRequest`, update
   `RAGService` resolution logic.
4. **Error handling**: add 404 responses for missing alias/KB in gateway routes.
5. **Production ops**: add create / update / alias / inspect entrypoints under `src/rag/ops`.
6. **Notebook surface**: expose those entrypoints through `experiments/rag/notebook_ops.py` and
   keep notebook-only experimental forks under `experiments/rag/sandboxes/`.
7. **DAGs**: update `arxiv_rag_update` and `pytorch_docs_rag_update` for new alias workflow.
8. **Eval integration**: add `rag_alias` column, update eval runner to pass alias in API calls.
9. **Admin endpoint**: implement `GET /v1/knowledge-bases`.
10. **Cleanup DAG**: implement `rag_collection_cleanup`.
11. **Migration**: migrate legacy `chat_documents`/`code_documents` to alias-based naming.
