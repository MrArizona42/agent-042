# Agentic Selection: Embedding-Based Task Routing and KB Auto-Selection

## Goal

Replace the rule-based task router and manual UI KB selection with a two-layer
embedding-based decision pipeline. The user sends a query — the system figures
out the task, the LoRA adapter, and which KBs are relevant, all without the user
touching any control.

---

## Current state

| Decision | Current mechanism | Problem |
|---|---|---|
| Task (chat / code / summarize) | Keyword regex in `RuleBasedTaskRouter.decide()` | Fragile, trivially wrong on mixed or implicit queries |
| LoRA adapter | `req.model or settings.default_model` — client must name it explicitly | Gateway never auto-selects a LoRA; eval runner does `lora-{task}` manually |
| KB selection | `req.rag_sources` — UI radio button or explicit API field | Manual; user must know what KBs exist |

The `knowledge_bases.json` groups KBs by task and carries a `description` per KB.
`EmbeddingService` is already instantiated at gateway startup inside `RAGService`.
Neither is currently used for routing.

This plan makes task routing the primary decision. The selected task determines:
- which system prompt shape to use
- which LoRA adapter to activate
- which KBs are even eligible for auto-selection

Layer 2 only chooses KBs from that task's pool. It never changes the task.

---

## Task entry configuration in `knowledge_bases.json`

Each task entry becomes the single source of truth for:
- how the task is recognized by the embedding router
- whether a LoRA is active for that task
- which KBs are eligible for auto-selection after the task is chosen

```json
{
    "task": "chat",
    "label": "General knowledge",
    "routing_description": "Open-ended ML/DL/AI/LLM research discussion, conceptual Q&A, comparisons, brainstorming, and general assistance that is not primarily code debugging and not a request to summarize provided text.",
    "adapter": {
        "name": "",
        "alias": "",
        "enabled": false
    },
    "knowledge_bases": [
        {
            "name": "arxiv",
            "label": "ArXiv papers (ML / AI theory)",
            "description": "Deep discussions about ML/AI theory and latest trends",
            "selection_description": "Research papers and literature-grounded ML/AI content: architectures, training methods, evaluation results, benchmarks, ablations, and theory-heavy discussion.",
            "default_alias": "champion",
            "aliases": {...}
        }
    ]
}
```

The summarization task is represented explicitly in the same file, but its KB
pool is intentionally empty:

```json
{
  "task": "summarize",
  "label": "Summarization",
  "routing_description": "Summarize, condense, rewrite, or produce TL;DR / bullet recap of user-supplied text, article, transcript, notes, or conversation. This task works from the provided content and should not use external retrieval.",
  "adapter": {
    "name": "",
    "alias": "",
    "enabled": false
  },
  "knowledge_bases": []
}
```

Recommended seed texts for the embedding router / selector:

- `chat.routing_description`: `Open-ended ML/DL/AI/LLM research discussion, conceptual explanation, comparison, brainstorming, planning, and general Q&A that is not mainly code debugging and not a request to summarize provided text.`
- `code.routing_description`: `Programming help for ML systems: writing code, debugging tracebacks, refactoring, explaining APIs, fixing integration issues, and reasoning about implementation details.`
- `summarize.routing_description`: `Summarize or condense user-provided content into a shorter form such as TL;DR, bullets, outline, recap, or structured summary without relying on external knowledge retrieval.`
- `arxiv.selection_description`: `Research papers, recent ML/AI methods, architectures, experiments, benchmarks, theory, and literature-grounded answers.`
- `pytorch_docs.selection_description`: `PyTorch API reference, tutorials, tensors, autograd, modules, distributed training, and library usage guidance for implementation questions.`

`label` stays human-readable. `routing_description` and
`selection_description` are the texts embedded for Layer 1 and Layer 2.

`summarize` is a first-class task even though it has `knowledge_bases: []`. It
participates in task routing, prompt selection, and optional LoRA selection,
but Layer 2 KB auto-selection is skipped for it by design.

**Default state:** `enabled: false`, `name: ""`, `alias: ""`. This means the task
uses the base model and produces no warnings. The empty strings are intentional
— they make the disabled state explicit in the file rather than relying on
field absence.

**Enabled state example:**

```json
"adapter": {
    "name": "lora-chat",
    "alias": "champion",
    "enabled": true
}
```

The `alias` field maps directly to the MLflow alias (`champion`, `challenger`, etc.)
and determines the vLLM adapter name as `{name}-{alias}`, e.g. `lora-chat-champion`.
This means one registered MLflow model (`lora-chat`) can have multiple aliases
promoted simultaneously (`champion`, `challenger`), and the JSON controls which
one the gateway uses for live serving. Switching from `challenger` back to
`champion` is a one-line JSON edit + config reload.

**Config validation rules:**

- `enabled: false` allows `name: ""` and `alias: ""`.
- `enabled: true` requires both `name` and `alias` to be non-empty strings.
- The effective vLLM adapter id is always `{name}-{alias}`. The gateway does not
  special-case aliases; the config must match whatever adapter-sync actually loaded.
- These rules apply equally to tasks with KBs and tasks with `knowledge_bases: []`
  such as `summarize`.

**Startup and reload validation:** the gateway calls vLLM's `/v1/models` once at
startup and again after `/v1/admin/reload-config`, then checks that every
`enabled: true` task adapter resolves to an existing model id `{name}-{alias}`.
A missing adapter logs a WARNING — it means the adapter-sync container hasn't
loaded it yet (transient, expected) or the config is wrong (permanent,
operator error). The gateway still starts and falls back to `default_model` for
that task.

**No per-request I/O.** Model selection at request time is a pure config lookup:
```python
adapter_cfg = task_cfg.adapter
if adapter_cfg and adapter_cfg.enabled and adapter_cfg.name and adapter_cfg.alias:
    model_name = f"{adapter_cfg.name}-{adapter_cfg.alias}"
else:
    model_name = settings.default_model
generation_payload["model"] = req.model or model_name
```

---

## LoRA lifecycle: from training to enabled in JSON

```
1. Train
   Researcher runs lora_training.ipynb or the Airflow train_lora DAG.
   The run is logged to MLflow with adapter artifacts.

2. Register
   In the operations notebook (lora_ops.ipynb):
       registry.register_adapter(run_id, artifact_path, model_name="lora-chat")
   The adapter appears in the MLflow Model Registry as a versioned entry
   with no aliases yet.

3. Evaluate as challenger
   Promote to the challenger alias:
       registry.promote("lora-chat", version=N, alias="challenger")
   Adapter-sync picks this up and hot-loads lora-chat-challenger into vLLM.
   Run the eval DAG against challenger to collect metrics.

4. Promote to champion
   If metrics are satisfactory:
       registry.promote("lora-chat", version=N, alias="champion")
   Adapter-sync hot-loads lora-chat-champion into vLLM.
   The old champion version is unloaded automatically.

5. Enable in knowledge_bases.json
   Edit the task entry:
       "adapter": {"name": "lora-chat", "alias": "champion", "enabled": true}
   Reload config (restart gateway or call /v1/admin/reload-config).
   Startup validator confirms lora-chat-champion is present in vLLM.
   From this point, all chat requests use the LoRA automatically.
```

Steps 1–4 are already supported by the existing `AdapterRegistry`,
`AdapterSyncer`, and eval pipeline. Step 5 is the only new operator action
introduced by this feature.

**Disabling without unloading:** set `enabled: false` in the JSON and reload.
The adapter stays in vLLM (no churn, no restart needed) but the gateway stops
routing to it. Useful during incidents or when switching to a new version.

---

## Proposed pipeline

```
user query
    │
    ▼
Layer 1 — Task classification (embedding similarity)
    │  embed(query) vs embed(task routing_description) for each task group
    │  → task ∈ {chat, code, summarize}
    │
    ├──► system prompt shape     (already wired via PromptBuilder)
    ├──► LoRA model name          config lookup: knowledge_bases.json adapter block
    ├──► KB candidate pool        task-scoped list from knowledge_bases.json
    │
    ▼
Layer 2 — KB relevance scoring (embedding similarity)
    │  embed(query) vs embed(kb.selection_description) for each KB in task's pool
    │  score ≥ threshold → include KB for retrieval
    │  skip entirely if selected task has no KBs (summarize)
    │
    ▼
RAG retrieval → prompt assembly → generation
```

Both layers use the same `EmbeddingService` already loaded at startup. No extra
model is needed; no extra LLM call is made.

Static prototype embeddings are computed once per gateway startup and rebuilt on
config reload:

- task embeddings: one vector per `routing_description`
- KB embeddings: one vector per `selection_description`

Only the user query embedding is request-time data. The task and KB prototype
embeddings should not be recomputed per request.

Yes: recalculating task and KB prototype embeddings once at startup / reload is
the intended design. They are static config-derived data, so caching them in
memory is the correct behavior.

---

## Request contract for `rag_sources`

`ChatCompletionRequest.rag_sources` stays in the schema, but its contract becomes
explicitly tri-state:

- `None` = auto mode. The gateway selects the task first, then automatically
  selects KBs only from that task's pool.
- `[]` = force off. Skip KB auto-selection and skip retrieval entirely.
- non-empty list = explicit override. Use the provided KBs / aliases as-is and
  bypass auto-selection.

Prompt behavior follows the same contract:

- If auto-selection or explicit retrieval was attempted for a task that has KB
  candidates, but no usable context was found, keep the existing message:
  `No relevant context was found in the knowledge base for this query.`
- If the client forces RAG off with `[]`, do not emit that message.
- If the selected task has no KB pool (`summarize`), do not emit that message;
  Layer 2 was intentionally skipped rather than attempted.

---

## What gets removed

- **`RuleBasedTaskRouter`** — replaced entirely. `task_router.py` becomes the
  embedding-based classifier, keeping the same `RouteDecision` interface so
  `processing.py` callsites stay unchanged.
- **UI KB radio button** — the sidebar "Knowledge Base" section in `app.py` and
  the `selected_kb` / `rag_sources` payload logic in the chat send block are
  removed. The gateway decides KB selection now.
- **`req.rag_sources` as the only way to opt into RAG** — omitted `rag_sources`
  now means automatic KB selection. Explicit `rag_sources` from the API remain
  the override path, and `[]` remains the force-off path.

---

## What is NOT affected

- `ChatCompletionRequest.rag_sources` schema — kept as an optional override field.
  Explicit sources passed by a client still bypass auto-selection entirely.
- `ChatCompletionRequest.model` — explicit model from a client bypasses LoRA
  auto-selection (eval runner uses this path).
- Alias resolution logic inside `RAGService` — unchanged.
- `Retriever` and reranker behavior once a KB is chosen — unchanged.
- Eval runner — already sets `model` and `rag_sources` explicitly, so it bypasses
  both new layers completely.
- `knowledge_bases.json` stays the single source of truth, but it now also owns
  task routing text and KB selection text in addition to adapter config.

---

## Layer 1 — Embedding-based task classifier

**File:** `src/gateway/services/task_router.py`

Replace `RuleBasedTaskRouter` with `EmbeddingTaskRouter`.

**Startup / reload (called during gateway startup and after config reload):**

- Wait for the embeddings service to be healthy.
- For each task group in `get_knowledge_bases()`, embed `routing_description`.
- Store as `{task: embedding_vector}`.
- Include `summarize` even though its KB list is empty.
- This cache is static config data; do not rebuild it per request.

**At inference time:**

- Embed the last user message.
- Compute cosine similarity against each task embedding.
- Return the task with the highest score.
- Fallback to `"chat"` only if the embedding router is unavailable.
- Task routing should not depend on Qdrant availability. It only depends on the
  embedding client and the cached task vectors.

**Interface preserved:** `router.decide(user_text: str) -> RouteDecision`

**Config knob (optional):** a `task_classification_threshold: float = 0.0` setting
in `RagSettings` — below threshold, fall back to `"chat"`. At 0.0 it always picks
the closest task, which is fine for 3 classes.

---

## Layer 2 — KB auto-selection

**File:** `src/gateway/services/rag_service.py`

Add a `select_knowledge_bases(query: str, task: str) -> list[RAGSource]` method.

**Startup / reload:**

- For each KB in `_KB_INDEX`, embed `kb.selection_description`.
- Store as `{kb_name: embedding_vector}`.
- Only KBs belonging to the resolved task's pool are candidates at query time
  (filter by `get_knowledge_bases()[task].knowledge_bases`).
- If a task's pool is empty (`summarize`), Layer 2 is skipped entirely.

**At query time:**

- Run only when `req.rag_sources is None`, RAG is enabled, and the selected task
  has at least one KB candidate.
- Embed the user query.
- Score each candidate KB description.
- Return `RAGSource(knowledge_base=kb.name)` for every KB scoring above
  `kb_selection_threshold`.
- If no KB clears the threshold, return an empty list (no RAG for this query),
  and keep the existing `No relevant context was found ...` prompt note.

The dense / hybrid retriever may still compute its own query embedding in the
first implementation. That is acceptable. The required optimization here is to
cache the task and KB prototype embeddings at startup / reload. Reusing the
request embedding across Layer 1, Layer 2, and retrieval can be a later cleanup.

**Config knob:** `kb_selection_threshold: float = 0.3` in `RagSettings`. This is
the main dial to tune aggressiveness. Start conservatively (0.3–0.4).

---

## Changes to `processing.py`

`_prepare_request` currently:
1. Calls `self._router.decide(last_user)` → `decision.task`
2. Uses `req.rag_sources` directly to drive retrieval
3. Sets `generation_payload["model"] = req.model or settings.default_model`

After this change:
1. Same — `decision.task` is still the result, but from the embedding router.
2. Resolve effective RAG behavior using the tri-state contract:
    - `req.rag_sources is None` → auto-select KBs from `decision.task`
    - `req.rag_sources == []` → force off; skip auto-selection and retrieval
    - non-empty `req.rag_sources` → use as explicit override
3. Set `rag_requested` for `PromptBuilder` like this:
    - `True` when explicit retrieval was requested, or auto-selection was attempted
      for a task with KB candidates
    - `False` when the client forced RAG off, or the selected task has no KB pool
      (`summarize`)
4. Derive LoRA from the task config's `adapter` block (pure config lookup, no I/O):
    - `adapter.enabled`, `adapter.name`, and `adapter.alias` non-empty → use `{name}-{alias}`
    - Otherwise → use `settings.default_model`
    - `req.model` set by client always wins (override path for eval runner)

Task selection always happens first. LoRA selection depends on the selected task
whether or not Layer 2 runs.

---

## Changes to `app.py` (UI)

Remove:
- The entire "Knowledge Base" `st.subheader` sidebar block (radio button,
  `kb_options`, `_kb_meta`, `selected_kb_label`, `selected_kb`).
- The `if selected_kb: payload["rag_sources"] = [...]` block in the send path.

The sidebar `get_knowledge_bases()` import can be removed too.

After this change, the UI simply omits `rag_sources`, which means `None = auto`
and lets the gateway perform task-first routing and KB auto-selection.

Keep:
- Everything else (sessions, auth, streaming, prompt preview).

**Optional replacement UI element:** a read-only `st.caption` showing which KBs
were used, sourced from the prompt preview `rag_context` field that is already
returned and displayed in the prompt preview expander.

---

## Changes to `shared/config.py`

Add two fields to `RagSettings`:

```python
kb_selection_threshold: float = Field(
    default=0.3,
    description="Cosine similarity threshold for automatic KB selection",
    ge=0.0,
    le=1.0,
)
task_classification_threshold: float = Field(
    default=0.0,
    description="Minimum cosine similarity for task classification; 0.0 = always pick closest",
    ge=0.0,
    le=1.0,
)
```

Also extend the registry models loaded from `knowledge_bases.json`:

- `TaskConfig.routing_description: str`
- `KBConfig.selection_description: str`
- `TaskConfig.adapter: AdapterConfig`

`AdapterConfig` should validate:

- `enabled=false` allows empty strings for `name` and `alias`
- `enabled=true` requires non-empty `name` and non-empty `alias`

`TaskConfig` should allow `knowledge_bases=[]` so `summarize` can exist as a
task without participating in RAG.

---

## Config reload and cache invalidation

`/v1/admin/reload-config` remains the operator entrypoint for applying edits to
`knowledge_bases.json`.

Reload must clear and rebuild every cache derived from that file:

- parsed task registry / KB index in `shared.config`
- task-router embedding cache
- KB-selection embedding cache
- RAG retriever cache
- cached `BuildConfig` metadata
- resolved Qdrant collection targets
- unavailable-alias cache
- enabled-adapter availability snapshot / validation results

After invalidation, reload should rebuild static task / KB embeddings from the
current JSON and rerun:

- KB alias validation against Qdrant
- enabled task-adapter validation against vLLM `/v1/models`

Any edit to the following fields requires reload before it affects live traffic:

- `routing_description`
- `selection_description`
- task `adapter`
- task membership of a KB
- KB alias metadata
- KB labels / descriptions used for discovery

If rebuild fails and strict startup mode is off, the gateway should stay alive
with safe fallbacks:

- task routing falls back to `chat`
- automatic KB selection returns no KBs
- model selection falls back to `default_model`

---

## Testing

**`tests/` additions:**

- `tests/gateway/test_task_router.py` — unit tests for `EmbeddingTaskRouter` using
  a mocked `EmbeddingService` that returns fixed vectors. Cover: correct task
  selection for representative queries, `summarize` selection, and fallback when
  the embedding router is unavailable.
- `tests/gateway/test_rag_auto_select.py` — unit tests for
  `RAGService.select_knowledge_bases`. Cover: query clearly in one KB's domain,
  query below threshold returns empty, `summarize` skips Layer 2 because its KB
  pool is empty, and explicit `rag_sources` override is not touched by auto-select.
- `tests/gateway/test_processing_lora.py` — test that `_prepare_request` injects
  the correct LoRA model name when `adapter.enabled=true` and correct name/alias
  are set, uses base model when `enabled=false`, respects explicit `req.model`,
  and enforces `None = auto`, `[] = force off`, non-empty list = override.
- `tests/shared/test_kb_config.py` — add cases for `AdapterConfig` validation:
  `enabled=true` with empty `name` should raise, `enabled=true` with empty
  `alias` should raise, `enabled=false` with empty strings is valid, and
  `summarize` with `knowledge_bases=[]` is valid.
- `tests/gateway/test_reload_config.py` — test that reload invalidates and
  rebuilds task / KB embedding caches in addition to the existing RAG caches.

Existing tests for `RAGService`, `PromptBuilder`, and the OpenAI-compat routes
should not need changes.

---

## Implementation order

1. `shared/knowledge_bases.json` + `shared/config.py` — add `AdapterConfig` model
  and `adapter` field to `TaskConfig`; add `routing_description` and
  `selection_description`; add the two threshold fields to `RagSettings`; add
  the explicit `summarize` task with `knowledge_bases: []`; update the JSON with
  disabled-by-default adapter blocks for all tasks.
2. `task_router.py` — replace `RuleBasedTaskRouter` with `EmbeddingTaskRouter`,
  build / invalidate cached task embeddings, and keep the `RouteDecision`
  interface.
3. `rag_service.py` — add `_build_kb_embeddings()` (startup / reload),
  `select_knowledge_bases()` (query time), and enabled-adapter presence checks
  in validation.
4. `processing.py` — wire in the tri-state `rag_sources` contract, task-first
  LoRA lookup, and `rag_requested` semantics for the prompt builder.
5. `knowledge_bases.py` reload endpoint — clear and rebuild all config-derived
  caches, not just the current RAG retrieval caches.
6. `app.py` — remove manual KB selector.
7. Tests.
