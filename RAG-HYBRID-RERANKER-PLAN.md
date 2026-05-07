# RAG Improvements: Hybrid Search + Reranker

## Current state

| Component | File | Status |
|---|---|---|
| Dense retrieval | `src/rag/retriever.py` | Working — `strategy="dense"` only |
| Hybrid / sparse strategies | `src/rag/retriever.py` | Raises `NotImplementedError` |
| Sparse encoder | `src/rag/sparse_encoder.py` | Stub — `NotImplementedError` |
| Reranker | `src/rag/reranker.py` | Stub — `NotImplementedError` |
| Vector store | `src/rag/vector_store.py` | Dense-only, **unnamed** vector format |
| Collection metadata | `src/rag/ops/meta.py` | Build metadata already carries retrieval capability and sparse encoder facts |
| Alias config | `src/shared/config.py` | `AliasConfig.reranker: Optional[str]` already exists but always `null` |
| Materialize helpers | `src/rag/ops/materialize.py` | Dense-only; no sparse indexing |
| Collection creation | `src/rag/ops/create/{arxiv,pytorch_docs}.py` | Pass `dense` strategy only |

---

## Architecture decisions

### Hybrid search: BM25 via fastembed

**Choice:** extend the existing `embeddings` microservice to also serve sparse (BM25) vectors.

Rationale:
- Mirrors the existing dense-embedding pattern; `SparseEncoderService` becomes an HTTP client like `EmbeddingService`.
- `fastembed` provides a `SparseTextEmbedding` class with the `Qdrant/bm25` model, which outputs `SparseVector(indices, values)` directly consumable by `qdrant-client`.
- Keeps heavy model I/O out of the gateway process.
- No new Docker service needed for the sparse path.

Alternative considered and rejected: in-process `rank_bm25` — requires building and persisting a shared vocabulary between index time and query time; fragile across rebuilds.

### Reranker: new microservice

**Choice:** a new `reranker` Docker service that mirrors the `embeddings` service pattern.

Rationale:
- Cross-encoder models are large and slow; isolating them keeps the gateway response-time profile unchanged for calls that don't hit the reranker.
- The `reranker` field in `AliasConfig` is already `Optional[str]`; `null` means disabled, a model name means "call the reranker service".
- Allows the reranker to be scaled independently and skipped entirely in low-resource deployments.

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` — small, fast, strong on passage-retrieval benchmarks.

### Qdrant collection schema change

Hybrid search requires **named vectors**:
- Dense leg: named `"dense"` (`VectorParams`)
- Sparse leg: named `"sparse"` (`SparseVectorParams`)

This is a **breaking change** vs the current unnamed-vector schema.

All collections — dense and hybrid — use named vectors. The unnamed-vector path is removed entirely; there is no conditional branching on schema format in `create_collection()`, `add_documents()`, or `search()`.

Migration: both `arxiv` and `pytorch_docs` must be rebuilt before this feature goes live. This rebuild is required anyway to add the sparse leg; migrating the dense leg from unnamed to named `"dense"` in the same pass costs nothing extra.

---

## Implementation plan

### Phase 1 — Config & data model

*Pure model/config changes. No logic changes. Existing tests must be green at the end of this phase.*

#### Step 1.1 — Rename `BuildConfig.retrieval_strategy` → `retrieval_capability`

**File:** `src/rag/ops/meta.py`

Rename the `retrieval_strategy` field to `retrieval_capability` everywhere in `BuildConfig`: the dataclass field, `to_payload()`, `from_payload()`, and the validation error message. Update all call sites that read `build_config.retrieval_strategy`.

Files affected: `src/rag/ops/meta.py`, `src/rag/ops/create/arxiv.py`, `src/rag/ops/create/pytorch_docs.py`, `src/rag/ops/materialize.py`, `src/gateway/services/rag_service.py`, and any existing tests that set `retrieval_strategy` in a `BuildConfig` constructor or payload dict.

---

#### Step 1.2 — Update `AliasConfig` and `knowledge_bases.json`

**File:** `src/shared/config.py`

`AliasConfig` carries no Python defaults — every field must be explicit in `knowledge_bases.json` (existing policy). Add two new fields without defaults:
- `retrieval_strategy: Literal["dense", "hybrid"]` — controls which query path `RAGService` uses. A collection built as `hybrid` can still be queried as `dense` for A/B comparisons.
- `reranker_multiplier: int` — number of first-stage candidates expressed as a multiple of `top_k` (`candidates = top_k * reranker_multiplier`). Only used when `reranker` is non-null; a value of `1` effectively disables candidate expansion.

`score_threshold` remains a single alias-owned field, applied at the end of the active pipeline:
- without reranker: passed directly to `vector_store.search()` so Qdrant filters below-threshold points;
- with reranker: first-stage search uses `score_threshold=None` (no pre-filtering) and retrieves `top_k * reranker_multiplier` candidates; after `reranker.rerank()` scores and sorts them, `Retriever.retrieve()` filters out any document whose final reranker score is below `score_threshold` and then truncates to `top_k`.

**File:** `src/shared/knowledge_bases.json`

Add `"retrieval_strategy": "dense"` and `"reranker_multiplier": 4` to all existing alias entries (no behaviour change today — `reranker_multiplier` is ignored unless `reranker` is non-null). When promoting a hybrid collection to `champion`, set `"retrieval_strategy": "hybrid"` and recalibrate `score_threshold` — DBSF scores are `[0, 1]`, so values of `0.15`–`0.20` are a reasonable starting range; tune from eval results.

---

### Phase 2 — Vector store

*Self-contained to `src/rag/vector_store.py`. Testable with `qdrant_client` `InMemoryClient` in isolation. Depends on Phase 1 (uses `retrieval_capability` from `BuildConfig`).*

#### Step 2.1 — Update `QdrantVectorStore` for named vectors and hybrid search

**File:** `src/rag/vector_store.py`

**`create_collection()`** — replace the single-`VectorParams` signature with `vectors_config={"dense": VectorParams(size=dimension, distance=Distance.COSINE)}` unconditionally. When `retrieval_capability="hybrid"`, also pass `sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams())}`. No `sparse: bool` flag — the build capability drives the schema.

**`add_documents()`** — add `sparse_vectors: list[SparseVector] | None = None` parameter. Always upsert with `vector={"dense": embedding}`. When `sparse_vectors` is provided, extend to `{"dense": embedding, "sparse": sparse_vec}`.

**`search()`** — add `strategy: Literal["dense", "hybrid"] = "dense"` and `sparse_query: SparseVector | None = None`:
- `"dense"`: `query_points(query=dense_vec, using="dense", ...)` — switches from the unnamed default to explicit named-vector targeting.
- `"hybrid"`: `query_points` with `prefetch=[Prefetch(query=dense_vec, using="dense"), Prefetch(query=NamedSparseVector(...), using="sparse")]` and `query=FusionQuery(fusion=Fusion.DBSF)`. DBSF (Distribution-Based Score Fusion) normalizes each retrieval leg's score distribution before combining, producing a native `[0, 1]` output with no application-side normalization. The deployed server is v1.17.0 (DBSF requires ≥ 1.11). The caller-supplied `score_threshold` is passed directly to Qdrant and is meaningful against these `[0, 1]` scores.

The `meta_exclusion` filter (sentinel point exclusion) applies to both paths.

**`_extract_vector_size()`** — remove the plain-attribute fallback branch (`getattr(params, "size", None)`). After this step every collection uses named vectors, so `params` is always a `dict`; the scalar path is dead code and should be deleted to avoid masking future schema errors.

Imports to add: `SparseVector`, `SparseVectorParams`, `SparseIndexParams`, `NamedSparseVector`, `Prefetch`, `FusionQuery`, `Fusion` from `qdrant_client.models`.

---

### Phase 3 — Sparse encoding

*Server/client pair — always implemented together. Independent of Phase 4 (reranker). Depends on Phase 1 (`settings.sparse_encoder_model`).*

#### Step 3.1 — Extend embeddings service with sparse endpoint

**File:** `src/embeddings/main.py`

Add a `POST /v1/sparse-embeddings` endpoint that accepts `{"input": [...]}` and returns `[{"indices": [...], "values": [...], "index": i}, ...]`. Load `fastembed.sparse.SparseTextEmbedding` alongside the existing dense model at startup.

The sparse model name is read from `settings.sparse_encoder_model` (env var `SPARSE_ENCODER_MODEL`, default `"Qdrant/bm25"`). Add this field to `Settings` in `src/shared/config.py` following the same `Field(..., validation_alias=...)` pattern as `embedding_model`.

New dependency: `fastembed` added to the `embeddings` extras in `pyproject.toml` and to `infra/docker/embeddings/requirements.txt`.

---

#### Step 3.2 — Implement `SparseEncoderService`

**File:** `src/rag/sparse_encoder.py`

```
SparseEncoderService(embeddings_url: str | None)
  .encode_documents(texts: list[str]) -> list[SparseVector]
  .encode_query(text: str)            -> SparseVector
```

Where `SparseVector = qdrant_client.models.SparseVector`.

HTTP client pattern identical to `EmbeddingService` (uses `httpx.Client`, respects `settings.embeddings_url`). Batching mirrors `EmbeddingService._EMBED_BATCH`.

---

### Phase 4 — Reranker

*Server/client pair — always implemented together. Independent of Phase 3 (sparse encoding). Depends on Phase 1 (`settings.reranker_model`, `settings.reranker_url`).*

#### Step 4.1 — Implement `Reranker` as an HTTP client

**File:** `src/rag/reranker.py`

```
CrossEncoderReranker(reranker_url: str)
  .rerank(query: str, docs: list[Document], top_k: int) -> list[Document]
```

Calls `POST /v1/rerank` with `{"query": "...", "passages": ["doc1", "doc2", ...]}` and receives scores back, then overwrites `Document.score` with the reranker output and sorts descending by that score. Returns the full sorted list — **no thresholding inside `rerank()`**. `score_threshold` filtering is the caller's responsibility (done in `Retriever.retrieve()` after `rerank()` returns).

`get_reranker(model_name: str) -> Reranker` factory remains the public entry point; it reads `settings.reranker_url` and constructs a `CrossEncoderReranker`.

---

#### Step 4.2 — New `reranker` microservice

**New files:**
- `src/reranker/main.py` — FastAPI service, mirrors `src/embeddings/main.py` structure:
  - `POST /v1/rerank` → `{"query": str, "passages": list[str]}` → `{"scores": list[float]}`
  - `GET /v1/health`
- `infra/docker/reranker/Dockerfile` — mirrors `infra/docker/embeddings/Dockerfile`; installs `sentence-transformers`.

**`pyproject.toml`:** add `reranker` optional-dependency group with `sentence-transformers`, `fastapi`, `uvicorn`.

**`infra/compose/docker-compose.yaml`:** add `reranker` service; expose port 8101. Add `RERANKER_URL: http://reranker:8101` to the `x-shared-endpoints` fragment and pass it to `gateway`.

**`src/shared/config.py`:** add to `Settings`:
- `reranker_url: str = "http://reranker:8101"` — bound to env var `RERANKER_URL`.
- `reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"` — bound to env var `RERANKER_MODEL`. The reranker service reads this at startup (same pattern as `settings.embedding_model` in the embeddings service).
- `sparse_encoder_model: str = "Qdrant/bm25"` — bound to env var `SPARSE_ENCODER_MODEL` (also referenced from Phase 3).

All three fields use `Field(..., validation_alias=AliasChoices(...))` consistent with existing `Settings` fields.

---

### Phase 5 — Build pipeline

*Updates the index-build path to produce sparse vectors and named-vector collections. Depends on Phases 1–3.*

#### Step 5.1 — Update collection-creation helpers

**File:** `src/rag/ops/materialize.py`

**`create_collection_with_meta()`** — pass `retrieval_capability` from `meta.build_config` to `vector_store.create_collection()` so the correct `sparse_vectors_config` is added when capability is `"hybrid"`. No `sparse: bool` flag.

**`batch_embed_and_upsert()`** — add optional `sparse_encoder_service: SparseEncoderService | None` parameter. When provided, compute sparse vectors per batch and pass them to `vector_store.add_documents()`.

**Files:** `src/rag/ops/create/arxiv.py`, `src/rag/ops/create/pytorch_docs.py`

Each `create_*_collection()` function should:
1. Instantiate `SparseEncoderService` when `build_config.retrieval_capability == "hybrid"`.
2. Pass it through to `batch_embed_and_upsert`.
3. `create_collection_with_meta` derives the schema from `build_config.retrieval_capability` internally — no extra flag needed at the call site.

---

### Phase 6 — Query pipeline + tests

*Wires all previous phases into the live query path and adds test coverage. Steps within this phase must be done in order: 6.1 → 6.2 → 6.3. Depends on all prior phases.*

#### Step 6.1 — Update `Retriever.retrieve()` for hybrid path

**File:** `src/rag/retriever.py`

Replace the `if strategy != "dense": raise NotImplementedError` guard with a real implementation:

```python
fetch_k = top_k * self.reranker_multiplier if self.reranker is not None else top_k

if strategy == "hybrid":
    sparse_query = self.sparse_encoder_service.encode_query(query)
    candidates = self.vector_store.search(
        query_embedding=query_embedding,
        top_k=fetch_k,
        score_threshold=None if self.reranker is not None else score_threshold,
        filter_dict=filter_dict,
        strategy="hybrid",
        sparse_query=sparse_query,
    )
else:  # dense
    candidates = self.vector_store.search(
        query_embedding=query_embedding,
        top_k=fetch_k,
        score_threshold=None if self.reranker is not None else score_threshold,
        filter_dict=filter_dict,
    )

if self.reranker is not None:
    candidates = self.reranker.rerank(query, candidates, top_k)
    candidates = [d for d in candidates if d.score >= score_threshold]

return candidates[:top_k]
```

`Retriever.__init__` gains `sparse_encoder_service: SparseEncoderService | None = None` and `reranker_multiplier: int = 1`.

For aliases with reranking enabled, the same rule applies to all retrieval strategies:
- first-stage retrieval fetches `top_k * reranker_multiplier` candidates with no score filter;
- reranker output becomes the final `Document.score`;
- `score_threshold` is applied by `Retriever.retrieve()` after reranking.

For aliases without reranking, `score_threshold` is passed directly to Qdrant and the result is truncated to `top_k`.

This keeps a single public score field whose meaning is always "final score of the active alias pipeline".

---

#### Step 6.2 — Wire everything into `RAGService`

**File:** `src/gateway/services/rag_service.py`

In `_get_retriever()`, after reading `kb_cfg`, check `kb_cfg.aliases[alias].reranker`:
- `None` → pass `reranker=None` to `Retriever` (current behaviour).
- Non-null string → call `get_reranker(model_name)` and pass result to `Retriever`.

Also pass `reranker_multiplier=alias_cfg.reranker_multiplier` and `sparse_encoder_service` (when `alias_cfg.retrieval_strategy == "hybrid"`) to `Retriever.__init__`.

Also update `retrieve_documents()` to pass `strategy=alias_cfg.retrieval_strategy` to `retriever.retrieve()` instead of `build_cfg.retrieval_strategy` (the alias config owns query-time strategy; `build_cfg.retrieval_capability` is the build-time capability used only for compatibility checks).

**Multi-KB queries:** when a request supplies multiple `rag_sources`, `_retrieve_rag_chunks()` in `processing.py` already iterates over them and calls `retrieve_documents()` once per source. This behaviour is **unchanged** — each KB is retrieved independently using its own alias config (including its own `retrieval_strategy`, `reranker`, and `reranker_multiplier`). There is no cross-KB merging, deduplication, or shared reranking pass. A single request can therefore simultaneously hit one dense KB and one hybrid KB; each runs its own full pipeline.

---

#### Step 6.3 — Tests

**New test files to add:**
- `tests/rag/test_sparse_encoder.py` — mock the HTTP endpoint; verify batching, `SparseVector` output type.
- `tests/rag/test_reranker.py` — mock the HTTP endpoint; verify score attachment and top-k truncation.
- `tests/rag/test_vector_store_hybrid.py` — integration test using `qdrant_client` `InMemoryClient`; verify that named-vector collections are created and queried correctly.
- `tests/rag/test_retriever_hybrid.py` — unit test with mocked `QdrantVectorStore` and `SparseEncoderService`; verify the hybrid branch calls both and applies reranking.

Existing dense tests must remain green. The strategy default (`"dense"`) is unchanged; the named-vector format now applies to all paths, so test fixtures must create collections using `vectors_config={"dense": ...}` rather than an unnamed single vector.

---

## Dependency summary

| Package | Added to | Notes |
|---|---|---|
| `fastembed` | `pyproject.toml [embeddings]`, embeddings Dockerfile | Sparse BM25 model download on first start |
| `sentence-transformers` | `pyproject.toml [reranker]`, reranker Dockerfile | Already in airflow-worker-gpu; do not add to gateway |

No new gateway Python dependencies.

---

## Collection rebuild sequence

Once all code is in place, the rollout sequence per KB is:

1. Build a new collection with `retrieval_capability="hybrid"` → new timestamped name.
2. Assign it to the `challenger` alias.
3. Run eval to compare NDCG/MRR against the current `champion` (dense).
4. If metrics improve, reassign `champion` alias to the hybrid collection.
5. Delete the old dense collection.

The `champion` alias stays dense throughout this process; no user-facing regression risk.

---

## Open questions

Resolved direction for now:
- Query-time policy lives in `src/shared/knowledge_bases.json`.
- `knowledge_bases.json` is the RAG runtime registry, not just a KB list.
- Alias policy stays flat for now.
- Build-side `retrieval_strategy` is renamed to `retrieval_capability` (covered by Phase 1, Step 1.1).
- The gateway materializes a derived effective RAG config per query from alias policy + current Qdrant alias target + `_meta.build_config`, without re-reading every backing source on every request.
- Expensive build-capability reads are cached by resolved physical collection and invalidated on config reload or alias-binding changes.
- When hybrid-aware code meets legacy unnamed-vector collections, affected aliases are marked unavailable.
- `_meta` sentinel data migrates to the named-vector schema too.
- All RAG-related operations are in scope, including update flows and related scripts.
- Fallback policy is strict for the first iteration: if any required retrieval leg or reranker is unavailable, the whole request fails, including eval runs.
- Sparse-only remains part of the supported contract.
- Compatibility checks should validate whether the query-time policy can run against the resolved build capability.
- `Document.score` remains a single public field representing the final score of the active alias pipeline. For reranked aliases, first-stage retrieval fetches `top_k * reranker_multiplier` candidates with no score filter; `score_threshold` is applied by `Retriever.retrieve()` to the reranker's output scores only.
- `qdrant_client` `InMemoryClient` is sufficient for the planned hybrid integration tests.
- The new reranker service and the embeddings sparse-endpoint work should follow the repo's existing microservice conventions exactly, including `/health`, lockfile-based Docker requirements, Compose healthchecks, and current packaging conventions.

No remaining open questions are recorded in this plan for now.

---

## Polishing and bugfix plan

This follow-up section captures the post-implementation review gaps on the current feature branch. The goal is to close them in small, testable slices, one phase at a time.

### Phase P1 — Named-vector metadata parity

*Goal: make the named-vector migration complete and make `_meta` writes work on all rebuilt collections.*

Files:
- `src/rag/vector_store.py`
- `src/rag/ops/materialize.py`
- `tests/rag/test_vector_store_hybrid.py`
- `tests/rag/test_ops_update.py`

Plan:
- Update `QdrantVectorStore.write_meta()` so the sentinel point uses the same named-vector schema as the collection (`{"dense": [0.0] * dimension}`), instead of the old unnamed vector format.
- Verify that both dense and hybrid collections can write and read `_meta` successfully after creation.
- Add a regression test that exercises `create_collection_with_meta()` end to end and proves that metadata writes succeed for rebuilt collections.
- Extend update-path coverage so a refresh flow that reads/writes metadata on named-vector collections is tested directly.

Exit criteria:
- Fresh collection creation no longer fails on `_meta` upsert.
- `_meta` write/read regression is covered by tests.

---

### Phase P2 — Strict fail-closed runtime behavior

*Goal: make the live query path match the agreed strict fallback policy.*

Files:
- `src/gateway/services/rag_service.py`
- `src/gateway/services/processing.py`
- `tests/api/test_rag_lifecycle.py`
- `tests/eval/test_eval_workflow.py`

Plan:
- Stop converting retrieval-pipeline failures into silent empty lists in `RAGService.retrieve_documents()`.
- Stop swallowing RAG retrieval errors in `_retrieve_rag_chunks()` and continuing with partial or empty context.
- Preserve the distinction between "no results matched" and "the configured RAG pipeline failed / is unavailable".
- Apply the same fail-closed rule to eval paths so evals fail when a configured RAG source cannot complete its pipeline.

Exit criteria:
- Unavailable alias, sparse encoder failure, or reranker failure causes the full request to fail.
- Valid zero-hit retrieval still returns an empty result without being treated as an error.

---

### Phase P3 — Query/build compatibility enforcement

*Goal: reject impossible runtime configurations before serving traffic.*

Files:
- `src/gateway/services/rag_service.py`
- `src/rag/ops/meta.py`
- `tests/rag/test_config_contracts.py`
- `tests/rag/test_rag_lifecycle.py`

Plan:
- Add explicit compatibility validation between alias query strategy and `_meta.build_config.retrieval_capability`.
- Add explicit compatibility validation between the runtime sparse encoder configuration and `_meta.build_config.sparse_encoder` when sparse retrieval is required.
- Mark incompatible aliases unavailable during startup / retriever creation with actionable error messages.
- Cover both valid and invalid combinations with tests.

Compatibility matrix to enforce:
- Query strategy `dense` requires a dense leg, so it is valid only when the build capability includes dense vectors.
- Query strategy `hybrid` requires both dense and sparse legs, so it is valid only for hybrid-capable collections.
- Query strategy `sparse` requires a sparse leg, so it is valid only when the build capability includes sparse vectors.

Exit criteria:
- Alias/build mismatches are caught before query execution.
- Operator-facing logs explain why an alias was marked unavailable.

---

### Phase P4 — Sparse-only contract completion

*Goal: either fully implement the already-approved sparse-only contract or remove any remaining accidental partial state. Current plan assumes implementation, not rollback.*

Files:
- `src/shared/config.py`
- `src/shared/knowledge_bases.json`
- `src/rag/vector_store.py`
- `src/rag/retriever.py`
- `src/gateway/services/rag_service.py`
- `tests/rag/test_vector_store_hybrid.py`
- `tests/rag/test_retriever_hybrid.py`

Plan:
- Extend alias query config so `retrieval_strategy` can actually be `"sparse"` if sparse-only remains supported.
- Implement sparse-only collection/query behavior consistently across vector store, retriever, and runtime compatibility checks.
- Ensure the collection schema path supports collections that expose only the sparse leg when that is the declared build capability.
- Add dedicated tests for sparse-only retrieval so it is not represented only in type hints and metadata validation.

Exit criteria:
- Sparse-only is either fully operational end to end or explicitly removed from all remaining contracts in a later conscious decision.

---

### Phase P5 — Cache correctness for alias rebinds

*Goal: make the derived effective config and retriever caches reflect alias promotions / rebinds reliably.*

Files:
- `src/gateway/services/rag_service.py`
- `src/rag/ops/aliases.py`
- `tests/api/test_rag_lifecycle.py`
- `tests/rag/test_ops_aliases.py`

Plan:
- Rework build-capability caching so it is keyed by resolved physical collection, not only by `(kb, alias)`.
- Ensure the gateway notices when an alias now points to a different physical collection and refreshes the retriever/build-config state accordingly.
- Keep `reload-config` invalidation, but do not rely on it as the only way to observe alias rebinds.
- Add tests around alias promotion / reassignment so the next request uses the new collection metadata instead of stale cached state.

Exit criteria:
- Alias promotion changes are reflected without stale build-config reuse.
- Effective config materialization remains consistent with the resolved Qdrant alias target.

---

### Phase P6 — Regression coverage and focused validation

*Goal: leave the RAG slice with targeted tests that protect the agreed contract.*

Files:
- `tests/rag/test_vector_store_hybrid.py`
- `tests/rag/test_retriever_hybrid.py`
- `tests/rag/test_config_contracts.py`
- `tests/api/test_rag_lifecycle.py`
- `tests/eval/test_eval_workflow.py`

Plan:
- Add explicit regression tests for named-vector `_meta` writes.
- Add fail-closed request-path and eval-path tests.
- Add compatibility-matrix tests for alias strategy vs build capability.
- Add sparse-only tests if Phase P4 is implemented.
- Add an alias-rebind cache test if Phase P5 is implemented.

Validation command set after this polishing pass:
- `PYTHONPATH=src pytest tests/rag/test_config_contracts.py tests/rag/test_sparse_encoder.py tests/rag/test_reranker.py tests/rag/test_retriever_hybrid.py tests/rag/test_vector_store_hybrid.py tests/rag/test_ops_meta.py tests/rag/test_ops_update.py -q`
- plus the focused API / eval tests added or updated by Phases P2 and P5.

Exit criteria:
- The reviewed gaps are covered by executable regression tests, not just manual reasoning.
