# Plan: LoRA Hot-Swap via Stateless Sync

## Problem

Current setup requires restarting vLLM to change which LoRA adapters are loaded.
The adapter-sync init container runs before vLLM, writes `lora-modules.json`,
and vLLM reads it at startup. To deploy a new adapter or change what "champion"
points to, you must restart the whole vLLM container — slow and disruptive.

## Design Decisions (agreed)

1. **Disk paths use MLflow versions**: `/adapters/{model}/v{N}/model/` — immutable, cacheable, multiple versions coexist.
2. **vLLM adapter names use aliases**: `lora-summarize-champion`, `lora-code-challenger` — loaded/unloaded via vLLM hot-load API.
3. **Configurable alias list** (not hardcoded to two): `REGISTRY_SYNC_ALIASES=champion,challenger` — extensible.
4. **Sync is stateless**: no `lora-modules.json`, no `.sync-state.json`. MLflow is source of truth. Sync unconditionally unloads + reloads.
5. **Startup order flips**: vLLM starts first (no adapters), then sync runs against the live vLLM API, then gateway/UI start.
6. **No custom vLLM entrypoint**: drop `vllm-entrypoint.sh` and the wrapper Dockerfile — use stock `vllm/vllm-openai` image.
7. **CLI as primary entry point**: `manage_registry.py sync` for both startup and on-the-fly updates.

## Architecture: Before vs After

### Before
```
mlflow (healthy)
  → adapter-sync (downloads champ, writes lora-modules.json, exits)
    → vllm (reads manifest at startup, pre-loads adapters)
      → gateway → ui
```

### After
```
mlflow (healthy)  ─┐
                   ├→ vllm (starts with --enable-lora, zero adapters, becomes healthy)
                   │    → adapter-sync (queries MLflow, downloads, hot-loads via vLLM API, exits)
                   │      → gateway → ui
```

## Naming Conventions

### Disk
```
/adapters/lora-summarize/v3/model/adapter_config.json   ← from MLflow v3
/adapters/lora-summarize/v5/model/adapter_config.json   ← from MLflow v5
/adapters/lora-code/v2/model/adapter_config.json
```

### vLLM (loaded adapters)
```
lora-summarize-champion   →  /adapters/lora-summarize/v3/model
lora-summarize-challenger →  /adapters/lora-summarize/v5/model
lora-code-champion        →  /adapters/lora-code/v2/model
```

### API usage
```
"model": "lora-summarize-champion"    ← eval runner, gateway
"model": "lora-code-champion"         ← eval runner
"model": null                          ← UI (uses base model by default)
```

## Sync Algorithm

`manage_registry.py sync --vllm-url <URL>`:

```
1. Query MLflow: for each registered model × each alias in REGISTRY_SYNC_ALIASES
   → build desired_state: {("lora-summarize", "champion"): v3, ...}

2. Download missing versions to /adapters/{model}/v{N}/
   → skip if directory already exists (immutable)

3. Query vLLM: GET /v1/models → get list of currently loaded adapter names

4. Unload all adapters managed by sync (names matching {model}-{alias} pattern)
   → DELETE /v1/unload_lora_adapter {"lora_name": "lora-summarize-champion"}

5. Load desired adapters
   → POST /v1/load_lora_adapter {
       "lora_name": "lora-summarize-champion",
       "lora_path": "/adapters/lora-summarize/v3/model"
     }

6. Log summary: what was loaded, what was unloaded, any errors
```

## Typical Workflows

### Deploy first-ever adapter
```bash
# Train, register, promote in MLflow:
python scripts/manage_registry.py register lora-summarize --run-id <RUN_ID>
python scripts/manage_registry.py promote lora-summarize 1 --alias champion

# Hot-load into running vLLM:
python scripts/manage_registry.py sync --vllm-url http://localhost:8000
# → downloads v1, loads lora-summarize-champion
```

### Swap champion to newer version
```bash
python scripts/manage_registry.py promote lora-summarize 5 --alias champion
python scripts/manage_registry.py sync --vllm-url http://localhost:8000
# → downloads v5 (v3 stays on disk), unloads lora-summarize-champion, reloads pointing to v5
```

### A/B test champion vs challenger
```bash
python scripts/manage_registry.py promote lora-summarize 3 --alias champion
python scripts/manage_registry.py promote lora-summarize 5 --alias challenger
python scripts/manage_registry.py sync --vllm-url http://localhost:8000
# → both lora-summarize-champion and lora-summarize-challenger available

# Eval:
python experiments/scripts/eval/runner.py --lora-aliases champion,challenger
```

### Remove all adapters (go back to base model only)
```bash
python scripts/manage_registry.py demote lora-summarize --alias champion
python scripts/manage_registry.py sync --vllm-url http://localhost:8000
# → unloads everything, vLLM serves base model only
```

## Changes by File

### 1. `src/shared/config.py`
- Add `sync_aliases: list[str]` to `ModelRegistrySettings` (default: `["champion", "challenger"]`)
  loaded from `REGISTRY_SYNC_ALIASES` env var (comma-separated).
- Add `vllm_base_url: str` to `ModelRegistrySettings` (default: `"http://localhost:8000"`),
  loaded from `REGISTRY_VLLM_BASE_URL`.
- `production_alias` field stays for backward compat (used by `promote`/`demote` default).

### 2. `src/shared/model_registry.py`
- **`VllmLoraModule`** — remove or keep as internal-only (no longer serialized to JSON file).
- **`AdapterSyncer.sync()`** — rewrite:
  - Accept `vllm_base_url` parameter.
  - For each model × alias: query MLflow, download version dir, call vLLM hot-load API.
  - Use `requests` or `httpx` to call vLLM `/v1/load_lora_adapter`, `/v1/unload_lora_adapter`, `/v1/models`.
  - Remove `lora-modules.json` writing.
  - Remove `adapters-summary.json` writing.
- **`AdapterSyncer.discover_production_adapters()`** — rename/refactor to `discover_aliased_adapters()`,
  iterate over `sync_aliases` list instead of single `production_alias`.
- **CLI `sync` subcommand** — add `--vllm-url` flag.
- **`AdapterRegistry.download_adapter()`** — update path scheme to `{model}/v{version}/`.

### 3. `infra/docker/adapter-sync/sync-adapters.sh`
- Pass `--vllm-url http://vllm:8000` to sync command.
- No longer sets `VLLM_BASE_MODEL` (not needed without `lora-modules.json`).

### 4. `infra/docker/adapter-sync/Dockerfile`
- Add `requests` (or `httpx`) to dependencies if not already present
  (needed for vLLM API calls from inside the container).

### 5. `infra/docker/vllm/Dockerfile`
- **Delete** (use stock `vllm/vllm-openai:v0.16.0` image directly in compose).

### 6. `infra/docker/vllm/vllm-entrypoint.sh`
- **Delete**.

### 7. `infra/compose/docker-compose.yaml`

**`vllm` service:**
```yaml
vllm:
  image: vllm/vllm-openai:v0.16.0          # ← stock image, no custom build
  command: [
    "--model", "${VLLM_MODEL}",
    "--enable-lora",
    "--max-loras", "${VLLM_MAX_LORAS}",
    "--max-lora-rank", "${VLLM_MAX_LORA_RANK}",
    # ... other flags unchanged
  ]
  # NO depends_on adapter-sync
  volumes:
    - ${PROJECT_ROOT}/assets/models:/models:rw
    - ${PROJECT_ROOT}/assets/adapters:/adapters:rw
```

**`vllm-adapter-sync` service:**
```yaml
vllm-adapter-sync:
  # ... same build
  environment:
    # ... MLflow + S3 creds unchanged
    REGISTRY_SYNC_ALIASES: ${REGISTRY_SYNC_ALIASES:-champion,challenger}
    REGISTRY_VLLM_BASE_URL: http://vllm:8000     # ← NEW
  depends_on:
    mlflow:
      condition: service_healthy
    vllm:                                         # ← FLIPPED
      condition: service_healthy
```

**`gateway` service:**
```yaml
gateway:
  depends_on:
    vllm-adapter-sync:
      condition: service_completed_successfully   # ← wait for sync
    vllm:
      condition: service_healthy
    # ... rest unchanged
```

### 8. `scripts/manage_registry.py`
- Add `sync` subcommand:
  ```
  manage_registry.py sync [--vllm-url URL] [--aliases champion,challenger]
  ```
  Calls `AdapterSyncer.sync()` with vLLM URL.
- Keep all existing commands (`list`, `register`, `versions`, `promote`, `demote`, `download`, `production`).

### 9. `experiments/scripts/eval/runner.py`
- Simplify `_resolve_lora_alias()`:
  ```python
  def _resolve_lora_alias(lora_alias: str, task: str) -> dict:
      if lora_alias == "none":
          return {"adapter_name": None, ...}
      adapter_name = f"lora-{task}-{lora_alias}"  # ← deterministic, no MLflow call
      # Optionally still query MLflow for version/run_id metadata (for logging only)
      ...
  ```
- The `model` field sent to vLLM becomes `"lora-summarize-champion"` instead of `"lora-summarize"`.

### 10. `dags/eval_dags.py`
- No structural changes needed — it already passes `lora_aliases` to the runner.
  The runner change (item 9) handles the new naming transparently.

### 11. `experiments/notebooks/register_pretrained_loras.ipynb`
- Update Part III (deployment cells) to use `!python scripts/manage_registry.py sync`.
- Remove Option A (full restart) / Option B (manual curl) distinction — there's only one path now.
- Update curl examples to show the new adapter naming (`lora-summarize-champion`).

### 12. `assets/adapters/lora-modules.json`
- **Delete** (no longer generated or consumed).

### 13. `README-SYSTEM-DESIGN.md`
- Update LoRA adapter section to reflect new sync flow, naming, and startup order.

### 14. `src/gateway/services/processing.py`
- No changes needed. It already passes `req.model` straight through to vLLM.
  Clients wanting a LoRA just set `"model": "lora-summarize-champion"`.

## Configuration (.env additions)

```bash
# Aliases to sync from MLflow to vLLM (comma-separated)
REGISTRY_SYNC_ALIASES=champion,challenger

# vLLM URL for hot-load API (used by sync command)
REGISTRY_VLLM_BASE_URL=http://vllm:8000
```

## Migration / Backward Compatibility

- The old `lora-modules.json` flow is fully removed. No fallback.
- `production_alias` setting stays in config for `promote`/`demote` default argument.
  It's no longer used for sync (replaced by `sync_aliases` list).
- Old adapter dirs (`/adapters/lora-summarize/v1/`) remain on disk — no cleanup needed.
  New sync just ignores them and writes to the same structure.
- First run after migration: `docker compose up` starts vLLM clean, sync loads
  whatever aliases exist in MLflow — or nothing if no aliases are set.

## Testing Checklist

- [ ] `manage_registry.py sync` with zero aliases in MLflow → vLLM has no adapters
- [ ] `promote` + `sync` → adapter appears in `GET /v1/models`
- [ ] Request with `"model": "lora-summarize-champion"` → uses adapter
- [ ] Request with `"model": null` → uses base model
- [ ] `promote` to new version + `sync` → champion swaps without restart
- [ ] `demote` + `sync` → adapter unloaded
- [ ] Full `docker compose up` from scratch → correct startup order, adapters loaded
- [ ] `docker compose up` with no MLflow aliases → clean start, base model only
- [ ] Eval runner with `--lora-aliases champion,challenger` → correct adapter names in results
- [ ] `sync` when vLLM is unreachable → clear error message, no partial state
