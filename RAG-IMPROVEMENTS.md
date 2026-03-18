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
**Used by**: gateway, eval runner, build scripts.
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

When a build script creates a collection, it writes a metadata point:

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

Alias promotion is performed only via CLI (over SSH) or automatically within DAGs.
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

## 5. Two workflows: data freshness vs. experimentation

The system has two distinct workflows that must not interfere with each other.

### 5.1 Data freshness (DAG-driven, automated)

**Purpose**: keep production data current (new arxiv papers, updated pytorch docs).
**Scope**: only touches `champion` alias.
**Trigger**: scheduled Airflow DAGs.

#### Incremental strategy (e.g., arxiv)

DAG: `arxiv_rag_update` (daily).

1. Download new data (arxiv papers).
2. Version with DVC.
3. For each alias in `KNOWLEDGE_BASES["arxiv"]["aliases"]`: resolve alias → get collection name.
4. Read `_meta` from each resolved collection → extract build config (chunking, embedding model).
5. Upsert new data into each collection using its own build config.
6. If an alias does not resolve (collection doesn't exist), log a warning and skip. Do not fail
   the entire DAG — other aliases may still be valid.

**Why update all aliases, not just champion**: for incremental KBs, champion and challenger share
the same underlying data — they may differ in chunking/embedding but both need fresh papers. If
challenger doesn't exist, it's simply skipped.

#### Replace strategy (e.g., pytorch_docs)

DAG: `pytorch_docs_rag_update` (weekly).

**The DAG only rebuilds champion. It does not touch challenger.**

Step-by-step:

1. Resolve `pytorch_docs_champion` → get current collection name → read `_meta` → extract
   build config.
2. Create new collection: `pytorch_docs_{timestamp}`.
3. Create staging alias: `pytorch_docs_champion_staging` → point to new collection.
4. Write `_meta` point with the same build config to the new collection.
5. Build index into new collection (via staging alias).
6. Run gating eval: **nDCG@10 on BEIR-SciFact** (single metric).
7. Compare with current champion's last eval score:
   - **Within -5% to +20%**: auto-promote.
     - Re-point `pytorch_docs_champion` to the new collection.
     - Staging alias `pytorch_docs_champion_staging` remains on the same collection (will be
       re-pointed at next build).
     - Old collection (previously behind champion) loses its alias and will be cleaned up
       by the cleanup DAG after 7 days.
   - **Outside range** (drop > 5% or improvement > 20%):
     - Log as anomaly (future: send notification).
     - Do **not** re-point champion. Production continues serving from old collection.
     - Staging alias stays on the new collection. At next build, staging will be re-pointed
       to an even newer collection.

### 5.2 Experimentation (human-driven, manual)

**Purpose**: test new retrieval architectures (different chunking, embedding model, etc.).
**Scope**: only touches `challenger` alias. Never modifies `champion`.
**Trigger**: manual CLI invocation.

Step-by-step:

1. **Build** a new collection with experimental parameters:
   ```bash
   python build_vector_index.py \
       --kb pytorch_docs \
       --alias challenger \
       --chunking-strategy section_aware \
       --chunk-size 1024 \
       --chunk-overlap 128 \
       --embedding-model intfloat/e5-large-v2
   ```
   The script:
   - Creates collection `pytorch_docs_{timestamp}`.
   - Writes `_meta` with the provided build config.
   - Builds the index.
   - Points `pytorch_docs_challenger` at the new collection.

2. **Evaluate** against champion:
   ```bash
   python -m experiments.scripts.eval.runner \
       --task retrieval --kb pytorch_docs --dataset beir_scifact \
       --rag-aliases champion,challenger
   ```
   For each alias, the eval runner resolves `pytorch_docs_{alias}` → reads `_meta` → builds a
   temporary benchmark collection with the same config → computes retrieval metrics. Results are
   logged to `eval_runs` with the `rag_alias` and `knowledge_base` columns.

3. **Promote** if results are good:
   ```bash
   python manage_rag.py promote --kb pytorch_docs --from challenger --to champion
   ```
   This re-points `pytorch_docs_champion` to the collection currently behind
   `pytorch_docs_challenger`. The old champion collection loses its alias and will be cleaned up.

4. **After promotion**: the next DAG run for data freshness will read `_meta` from the new
   champion collection — which now has the experimental build config. The DAG automatically
   inherits the new parameters. No config files to update.

---

## 6. Promotion and collection management

### 6.1 Manual promotion CLI

A new script `manage_rag.py` (analogous to `manage_registry.py` for LoRA adapters):

```bash
# Promote challenger → champion
python -m scripts.manage_rag promote --kb pytorch_docs --from challenger --to champion

# List all aliases and their target collections
python -m scripts.manage_rag list

# Inspect collection metadata
python -m scripts.manage_rag inspect --kb pytorch_docs --alias champion
```

### 6.2 Orphan collection cleanup

A daily Airflow DAG (`rag_collection_cleanup`) handles garbage collection:

1. List all Qdrant collections.
2. List all Qdrant aliases → build a set of collections that have at least one alias.
3. For each collection **not** in that set:
   - Parse the timestamp from its name (`{kb_name}_{timestamp}`).
   - If older than 7 days → delete.
4. Legacy collections not following the naming convention are on an explicit skip-list.

---

## 7. Eval pipeline integration

### 7.1 Request routing

| Caller | What is sent | Alias resolution |
|---|---|---|
| **UI** | `knowledge_base` only (alias defaults to `champion`) | Gateway resolves `{kb}_champion` |
| **Eval runner (generation)** | `knowledge_base` + explicit `alias` | Gateway resolves `{kb}_{alias}` |
| **Eval runner (retrieval-only)** | `kb_name` + `rag_alias` | Resolves `{kb}_{alias}` → reads `_meta` → builds temp collection |

**Generation evals** (Chat, Summarization, Code, RAG+Chat, RAG+Code) call the **gateway API**.
The gateway is the single source of truth for alias resolution, RAG retrieval, and inference.

**Retrieval-only evals** use the `RAG/` library directly (not the gateway) to build temporary
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
| `build_chat_index()` | Upserts into `chat_documents` (no alias) | Resolves all aliases for KB → reads `_meta` per collection → upserts with correct config |
| `build_code_index()` | Single-alias swap (`code_documents`) | Champion-only rebuild via staging alias, with gating eval |
| Airflow DAGs | Build + swap in one step | Build → staging → eval → conditional promote |
| Build parameters | Hardcoded in build scripts | Stored in collection `_meta` point; passed via CLI args on first build |
| Eval runner | No alias awareness | Sends `rag_alias` in API request; logs alias + build config in `eval_runs` |
| Admin | No KB discovery endpoint | `GET /v1/knowledge-bases` |
| Cleanup | Manual | Daily DAG: delete orphan collections older than 7 days |

---

## 9. Implementation order

Suggested sequence for implementation:

1. **Collection metadata point**: add `_meta` write to `build_vector_index.py`, add `must_not`
   filter to `QdrantVectorStore.search()`.
2. **Runtime config**: create `src/shared/knowledge_bases.json`, add Pydantic model and loader to
   `shared/config.py`, replace `KNOWLEDGE_BASES` dict.
3. **API contract**: add `RAGSource` schema, update `ChatCompletionRequest`, update
   `RAGService` resolution logic.
4. **Error handling**: add 404 responses for missing alias/KB in gateway routes.
5. **Build scripts**: refactor `build_chat_index` and `build_code_index` to use aliases + `_meta`.
6. **Promotion CLI**: create `manage_rag.py`.
7. **DAGs**: update `arxiv_rag_update` and `pytorch_docs_rag_update` for new alias workflow.
8. **Eval integration**: add `rag_alias` column, update eval runner to pass alias in API calls.
9. **Admin endpoint**: implement `GET /v1/knowledge-bases`.
10. **Cleanup DAG**: implement `rag_collection_cleanup`.
11. **Migration**: migrate legacy `chat_documents`/`code_documents` to alias-based naming.
