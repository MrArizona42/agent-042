# RAG Refactor Plan: Alias-Based Query Config

## The core idea

The system already uses aliases for two things: LoRA adapter selection and
Qdrant collection management. This refactor adds a **third alias namespace** for
query-time RAG behavior, and makes all three orthogonal — they can be swapped
independently without any cross-system sync.

| Alias namespace | What it controls | Where it lives | Set / changed by |
|---|---|---|---|
| **LoRA alias** | Which adapter weights | Model registry | Training / promotion |
| **RAG build alias** | Which physical Qdrant collection | Qdrant | Notebook `rag_ops.ipynb` |
| **RAG query alias** | Post-search retrieval params | `knowledge_bases.json` | JSON edit + reload |

Any combination of these three is valid. They do not need to be kept in sync
because they are fully independent dimensions of the system.

---

## Why this matters for experimentation

Today, query params (`top_k`, `score_threshold`, `context_max_length`) live in
env vars / `RagSettings` — one global value for the entire service. To try
different retrieval settings for a challenger you have to change environment
variables and restart.

After this refactor, each RAG alias carries its own query config in
`knowledge_bases.json`. Testing a new configuration is:

1. Edit the challenger entry in `knowledge_bases.json`
2. POST to the new authenticated `/v1/admin/reload-config` endpoint (no restart required)
3. Send requests with `"alias": "challenger"` and compare

---

## Config contracts

### `knowledge_bases.json` — query-alias config

The only structural change to the JSON: `aliases` goes from a list of strings
to a dict of config objects. `default_alias` is added at the KB level.
Everything else is unchanged.

```json
[
  {
    "task": "chat",
    "label": "General knowledge",
    "knowledge_bases": [
      {
        "name": "arxiv",
        "default_alias": "champion",
        "aliases": {
          "champion":   { "top_k": 5,  "score_threshold": 0.35, "context_max_length": 4000, "reranker": null },
          "challenger": { "top_k": 10, "score_threshold": 0.25, "context_max_length": 4000, "reranker": null }
        },
        "update_strategy": "incremental",
        "label": "ArXiv papers (ML / AI theory)",
        "description": "Deep discussions about ML/AI theory and latest trends"
      }
    ]
  }
]
```

`AliasConfig` is validated by Pydantic with no defaults — every field must be
explicit. What you read in the file is exactly what runs. Adding a new field
forces you to update every alias entry; Pydantic will reject incomplete entries
at startup, making omissions immediately visible.

```python
class AliasConfig(BaseModel):
    top_k: int
    score_threshold: float
    context_max_length: int
    reranker: Optional[str]   # null today; model name when reranker is implemented
```

### `BuildConfig` — collection-structural facts (Qdrant `_meta`)

`BuildConfig` is stored as a sentinel point inside each Qdrant collection at
creation time. It records *how the collection was built* and is never changed
afterwards. The gateway reads it once at startup per active alias.

```python
@dataclass(frozen=True)
class BuildConfig:
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    sparse_encoder: str | None        # None → dense-only collection
    retrieval_strategy: Literal["dense", "hybrid", "sparse"]  # set at build time
```

`retrieval_strategy` belongs here — not in the JSON — because it is a
structural property of the collection. A collection either has a sparse vector
field or it doesn't; that fact cannot be changed without a rebuild. Keeping it
in `BuildConfig` means the collection declares how it wants to be queried, and
there is no cross-system mismatch possible.

### The two-layer split

| Layer | Fields | Lives in | Changed by |
|---|---|---|---|
| **Build config** | `chunking_strategy`, `chunk_size`, `chunk_overlap`, `embedding_model`, `sparse_encoder`, `retrieval_strategy` | Qdrant `_meta` | Collection rebuild only |
| **Query config** | `top_k`, `score_threshold`, `context_max_length`, `reranker` | `knowledge_bases.json` | JSON edit + authenticated `/v1/admin/reload-config` |

---

## Migration rules

- The `knowledge_bases.json` schema change is a hard cut. The legacy
  `aliases: ["champion", "challenger"]` list format is rejected at load time;
  code should not try to support both formats.
- `_KB_INDEX` is a flat lookup keyed by KB name. Duplicate KB names across
  task groups are invalid and should raise during config load.
- Legacy Qdrant collections whose `_meta.build_config` lacks
  `sparse_encoder` or `retrieval_strategy` are invalid after this refactor.
  In non-strict mode the gateway logs and marks the alias unavailable; with
  `rag_strict_startup=true` startup raises. Rebuild is required.
- The Streamlit UI migration is part of this refactor. Remove `_KBProxy` only
  after `src/ui/app.py` is switched to shared config helpers.
- The reload endpoint reuses the existing authenticated-user session flow.
  Do not introduce a separate admin key. If auth middleware is disabled, the
  endpoint should be unavailable rather than public.
- Removing `top_k`, `score_threshold`, `context_max_length`, `default_alias`
  from `RagSettings` makes the corresponding env vars (`GATEWAY_TOP_K`,
  `GATEWAY_SCORE_THRESHOLD`, `GATEWAY_CONTEXT_MAX_LENGTH`,
  `GATEWAY_DEFAULT_ALIAS`) dead config. Deployment `.env` files, Helm
  values, and CI scripts must be cleaned in lockstep with this change.
- Airflow DAGs (`dags/arxiv_rag_update.py`, `dags/pytorch_docs_rag_update.py`)
  call `rag.ops.update` functions which use the duplicated
  `_validate_kb_alias()` being deleted. After ops consolidation the DAG
  callsites are unaffected — the imported ops modules switch to
  `shared.config.validate_kb_alias` internally. No DAG code changes needed.

---

## Phase 1 — Config schema

**`src/shared/config.py`**

Add `AliasConfig` before `KBConfig`. No defaults — pure validator:

```python
class AliasConfig(BaseModel):
    top_k: int
    score_threshold: float
    context_max_length: int
    reranker: Optional[str]
```

Change `KBConfig`:
- `aliases: list[str]` → `aliases: dict[str, AliasConfig]`
- Add `default_alias: str` (required, no default — must be explicit in JSON)
- Validate at load time that `default_alias` is one of the declared alias keys

Add `_KB_INDEX: dict[str, KBConfig] | None = None` alongside `_KB_REGISTRY`.
Populate both in `_load_knowledge_bases()`. `_KB_INDEX` is a flat dict keyed by
`kb_name`, so duplicate names across tasks are a startup error. Change
`get_kb_config()` from a linear scan to an O(1) dict lookup. Add
`clear_knowledge_base_caches()` that resets `_KB_REGISTRY` and `_KB_INDEX`, and
have `clear_settings_caches()` call it.

Add two shared helpers that replace five near-identical copies scattered across
the ops files:

```python
def get_kb_names() -> list[str]:
    """Flat list of all KB names across all tasks."""

def validate_kb_alias(kb: str, alias: str) -> None:
    """Raise ValueError with a consistent message if kb or alias is unknown."""
```

Do not delete `_KBProxy` in the first patch. First migrate the active UI caller
in `src/ui/app.py` to `get_knowledge_bases()` / `get_kb_config()` (Phase 2),
then delete `_KBProxy` class and `KNOWLEDGE_BASES = _KBProxy()` in the same
refactor. No long-lived backward-compat layer remains afterwards.

Remove from `RagSettings`: `top_k`, `score_threshold`, `context_max_length`,
`default_alias`. Add `rag_strict_startup: bool = False` (used in Phase 3).
`RagSettings` retains only infrastructure: `rag_enabled`, `embedding_model`,
`embedding_device`, `embedding_batch_size`, `knowledge_bases_path`.

**`src/shared/knowledge_bases.json`**

Update both KBs to the new alias dict format with all fields explicit. The full
file structure is shown in the contract section above. This is a breaking
schema change; external overrides must be updated in lockstep.

---

## Phase 2 — Decouple `Retriever` from `Settings` and migrate active callers

`Retriever` currently receives a `Settings` object in its constructor and falls
back to `settings.top_k` / `settings.score_threshold` / `settings.context_max_length`
when callers pass `None`. After this phase all params are required — the
retriever has no settings knowledge at all.

**`src/rag/retriever.py`**

- `__init__(embedding_service, vector_store, settings)` →
  `__init__(embedding_service, vector_store)`. Drop `self.settings`.
- `retrieve(query, top_k, score_threshold, task=None)` — `top_k` and
  `score_threshold` become required, non-Optional.
- `format_context(documents, max_length)` — `max_length` becomes required.

**`src/gateway/services/rag_service.py`**

Delete `format_documents()` — it truncates context mid-string. Replace all
usages with `Retriever.format_context()`, which respects document boundaries.

**Formatting change (intentional):** The current `format_documents()` emits
`[Document N] (Source: ..., Score: 0.xxx)\n{content}` headers and then hard-
truncates the concatenated string at `context_max_length`, potentially cutting
a document mid-sentence. `Retriever.format_context()` respects document
boundaries: it appends whole documents until the next one would exceed the
limit, then stops. The metadata header format may also differ. This is a
deliberate quality improvement — the LLM receives complete documents rather
than a truncated blob.

**`src/ui/app.py`**

- Replace the `KNOWLEDGE_BASES` import with `get_knowledge_bases()` /
  `get_kb_config()` or a tiny flat helper derived from them
- Preserve the current sidebar UX (KB labels + descriptions) while removing the
  proxy dependency
- After this lands, delete `_KBProxy` and `KNOWLEDGE_BASES`

**`src/gateway/schemas/openai_chat.py`**

- Change the `RAGSource.alias` description to say `None` uses the selected KB's
  `default_alias`, not a global `GATEWAY_DEFAULT_ALIAS`

---

## Phase 3 — Wire query config and build config into the gateway

**`src/gateway/services/rag_service.py`**

Add `self._build_configs: dict[str, BuildConfig] = {}` to `__init__`.

In `validate_knowledge_bases()`: for each alias that resolves in Qdrant, read
`BuildConfig` from `_meta` and cache in `self._build_configs[f"{kb}_{alias}"]`.
If `_meta` is missing or the build config is legacy/incomplete (missing
`sparse_encoder` or `retrieval_strategy`), treat the alias as invalid — log and
mark it unavailable, or raise if `rag_strict_startup` is `True`. Also compare
`build_config.embedding_model` dimension against `embedding_service.dimension`
— log an error, or raise if `rag_strict_startup` is `True`. This gives staging
environments a fast-fail check without breaking local dev where collections may
not be fully bootstrapped.

In `retrieve_documents()`:
```python
alias = alias or kb_cfg.default_alias
alias_cfg = kb_cfg.aliases[alias]
build_cfg = self._build_configs[f"{kb}_{alias}"]
retriever.retrieve(
    query,
    top_k=alias_cfg.top_k,
    score_threshold=alias_cfg.score_threshold,
    strategy=build_cfg.retrieval_strategy,
    reranker=alias_cfg.reranker,
)
```

In `retrieve_context()`: pass `alias_cfg.context_max_length` to
`retriever.format_context()`.

In `available_knowledge_bases()`: return the full alias config dict
(`kb_cfg.aliases` serialized) and add `"default_alias": kb_cfg.default_alias`.
The `GET /v1/knowledge-bases` endpoint exposes the complete per-alias query
config so admin UIs and evaluation scripts can introspect what each alias is
configured with.

**`src/gateway/api/v1/openai_compat.py`** and **`src/gateway/services/processing.py`**

Replace `src.alias or settings.default_alias` with
`src.alias or get_kb_config(src.knowledge_base).default_alias`.

**`src/rag/ops/` — consolidate duplicated helpers**

`aliases.py`, `create/arxiv.py`, `create/pytorch_docs.py`, `update/arxiv.py`,
`update/pytorch_docs.py` each contain a verbatim copy of `_available_kbs()` and
`_validate_kb_alias()`. Delete all local copies; import `get_kb_names` and
`validate_kb_alias` from `shared.config`. In the `update/` files replace
`settings.default_alias` with `get_kb_config(kb).default_alias`.

Note on alias promotion: `promote_alias` (Qdrant pointer swap) is already the
complete operation — no JSON sync needed. The promoted collection carries its
own `BuildConfig` with `retrieval_strategy`; the JSON query alias is independent.

**`src/gateway/api/v1/knowledge_bases.py` — authenticated config reload endpoint**

```python
@router.post("/admin/reload-config")
async def reload_config(request: Request):
  if request.app.state.session_manager is None:
    raise HTTPException(
      status_code=503,
      detail="Config reload is unavailable when auth is disabled",
    )

  _ = request.state.user_id  # guaranteed by auth middleware
  clear_knowledge_base_caches()
    return {"status": "reloaded"}
```

This endpoint lives under the existing `/v1` router, so the full path is
`POST /v1/admin/reload-config`. Reuse the existing authenticated-user session
flow; do not add a separate admin key. This is safe because `AliasConfig`
contains only post-search numbers with no Qdrant structural dependency.
`clear_knowledge_base_caches()` sets `_KB_REGISTRY = None` and `_KB_INDEX =
None`; the next request re-reads the JSON from disk. The reload endpoint also
invalidates `RAGService._build_configs` and `RAGService._retrievers` so that
the next request per alias re-reads `BuildConfig` from Qdrant `_meta` and
re-creates the retriever. This means that if a collection was rebuilt while the
gateway was running, a single reload picks up both the new JSON query config
and the new build config. Cost: one extra Qdrant `_meta` read per alias after
reload. At worst one concurrent request completes with the old registry, which
is acceptable.

Hot-reload is practical now precisely because query aliases are structurally
independent of collections. Previously, `retrieval_strategy` in the JSON would
have required cross-validating against Qdrant `_meta` on every reload — making
it genuinely complex. That constraint no longer exists.

---

## Phase 4 — Skeleton classes for retrieval strategy and reranker

The dispatch code ships in the next iteration, but the architectural seams are
established here so that iteration is purely additive — no refactoring needed.

**`src/rag/reranker.py` (new file)**

```python
class Reranker:
    """Post-retrieval cross-encoder reranker. Not yet implemented."""
    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        raise NotImplementedError

def get_reranker(model_name: str) -> Reranker:
    """Factory — mirrors get_chunker(). Implement model dispatch here."""
    raise NotImplementedError(f"Reranker '{model_name}' not yet implemented")
```

**`src/rag/sparse_encoder.py` (new file)**

```python
class SparseEncoderService:
    """Sparse vector encoder (BM25 / SPLADE). Not yet implemented."""
    def encode(self, texts: list[str]) -> list[SparseVector]:
        raise NotImplementedError
```

**`src/rag/retriever.py`**

`retrieve()` gains `strategy` and `reranker` params. For now only `"dense"` is
implemented; all other values raise `NotImplementedError`:

```python
def retrieve(self, query, top_k, score_threshold, strategy="dense", reranker=None, task=None):
    if strategy == "dense":
        candidates = self._vector_store.search(...)
    else:
        raise NotImplementedError(f"retrieval_strategy '{strategy}' not yet implemented")

    if reranker is not None:
        candidates = reranker.rerank(query, candidates, top_k)
    return candidates[:top_k]
```

`__init__` accepts `reranker: Reranker | None = None`.

**`src/rag/ops/meta.py`**

Add to `BuildConfig` as required fields. Do not silently default legacy
collections to dense; collections missing either field must be rebuilt:

```python
sparse_encoder: str | None
retrieval_strategy: Literal["dense", "hybrid", "sparse"]
```

**`experiments/rag/rag_ops.ipynb`**

Update the notebook explicitly. It constructs `BuildConfig` inline during the
initial create flow, so ops-layer changes alone are not enough.

- In the initial build section, pass `sparse_encoder` and
  `retrieval_strategy` when instantiating `BuildConfig`
- For the dense default path, make the example explicit:
  `sparse_encoder=None`, `retrieval_strategy="dense"`
- Update the markdown around the `_meta`-driven refresh path to say that
  legacy collections created before this refactor cannot be refreshed in-place;
  they require a one-time rebuild so `_meta.build_config` satisfies the new
  contract
- Update any inspection / example output notes so the notebook shows the new
  `build_config` fields and the rebuild requirement clearly

**`src/gateway/services/rag_service.py`**

In `_get_retriever()`: if `alias_cfg.reranker` is set, call
`get_reranker(alias_cfg.reranker)` and pass to `Retriever`. For now `reranker`
is `null` in all JSON entries so this branch is never reached.

---

## Phase 5 — Tests and contract cleanup


**`tests/api/test_rag_lifecycle.py`**

Update `kb_json_file` fixture to the new alias dict format with all fields
explicit:

```python
"default_alias": "champion",
"aliases": {
    "champion":   {"top_k": 5, "score_threshold": 0.35, "context_max_length": 4000, "reranker": None},
    "challenger": {"top_k": 5, "score_threshold": 0.35, "context_max_length": 4000, "reranker": None},
},
```

Replace the cache reset pattern:
```python
# new
cfg.clear_knowledge_base_caches()
```

**`tests/rag/test_ops_aliases.py`**, **`tests/rag/test_ops_meta.py`**

Check for inline KB JSON fixtures and update alias format if present.

**New tests**

- `AliasConfig` rejects incomplete JSON entries (missing any field raises
  `ValidationError` at load time, not at query time)
- `KBConfig.default_alias` must point to a declared alias
- Duplicate KB names across tasks are rejected when building `_KB_INDEX`
- `validate_kb_alias()` raises `ValueError` for unknown KB and unknown alias
- `RAGService.retrieve_documents()` passes `top_k` / `score_threshold` from
  `AliasConfig` and `retrieval_strategy` from `BuildConfig` to `Retriever`
- Legacy collection metadata missing `sparse_encoder` or `retrieval_strategy`
  is rejected; in non-strict mode the alias is marked unavailable, and with
  `rag_strict_startup=True` startup raises
- `POST /v1/admin/reload-config` clears only KB caches, requires an
  authenticated user, and is unavailable when auth is disabled
- `experiments/rag/rag_ops.ipynb` constructs `BuildConfig` with the new fields
  and documents the one-time rebuild requirement for legacy collections

**Docs / contract cleanup**

- `src/gateway/README.md`: remove `GATEWAY_TOP_K` / `GATEWAY_SCORE_THRESHOLD`
  from the active runtime contract and document alias-owned query config
- `README-SYSTEM-DESIGN.md`: update production policy so query params are
  alias-owned config rather than a single global gateway config
- `src/ui/app.py`: no longer imports `KNOWLEDGE_BASES`

---

## Experimentation workflow (end-to-end)

**Tuning query params only — no collection rebuild**

Edit the challenger alias in `knowledge_bases.json`, then reload without restart:
```bash
curl -X POST https://gateway/v1/admin/reload-config \
  --cookie "session_id=..."
```
The next request with `"alias": "challenger"` uses the new params immediately.
If auth is disabled locally, this endpoint is intentionally unavailable; in
that mode restart the gateway after editing the JSON.

**Changing chunking, embedding model, or enabling hybrid search — rebuild required**

Collections created before this refactor need a one-time rebuild because legacy
`_meta.build_config` lacks `sparse_encoder` / `retrieval_strategy`.

1. Run section 3 of `experiments/rag/rag_ops.ipynb` with `alias="challenger"`
   and the new `BuildConfig`. For a hybrid experiment set
   `sparse_encoder="bm25"` and `retrieval_strategy="hybrid"` in `BuildConfig`.
   The notebook chunks, embeds dense vectors, encodes sparse vectors, upserts
   everything into Qdrant, writes `_meta`, and attaches the Qdrant alias.
2. Edit the challenger JSON entry for `top_k`, `score_threshold`, etc.
3. Reload config (`/v1/admin/reload-config`) or restart gateway.
4. Evaluate with `experiments/eval/` notebooks — send requests with
   `"alias": "challenger"` and compare against champion.
5. Promote: `promote_alias(from_alias="challenger", to_alias="champion")` —
   atomic Qdrant pointer swap, zero downtime. Update `default_alias` in JSON
   if needed and reload config.

**Decision table**

| Goal | Edit JSON | Rebuild collection |
|---|---|---|
| Tune `top_k` / `score_threshold` / `context_max_length` | Yes | No |
| Enable reranker on challenger | Yes (`"reranker": "model-name"`) | No |
| Try different chunking or embedding model | No | Yes |
| Enable hybrid search | No (strategy comes from `BuildConfig`) | Yes — set `retrieval_strategy="hybrid"` + `sparse_encoder` at build time |
