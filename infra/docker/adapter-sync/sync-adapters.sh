#!/usr/bin/env bash
# sync-adapters.sh — download champion adapters from MLflow and write
# /adapters/lora-modules.json for vLLM.
#
# Exits 0 even when no adapters are found so the dependent vLLM service
# can still start (just without LoRA modules loaded at boot).
set -euo pipefail

ADAPTERS_DIR="${REGISTRY_ADAPTERS_DIR:-/adapters}"

echo "=== adapter-sync: starting ==="
echo "  MLflow URI : ${MLFLOW_BACKEND_URI:-<not set>}"
echo "  Adapters dir: ${ADAPTERS_DIR}"

python -m shared.model_registry sync \
    --adapters-dir "${ADAPTERS_DIR}" \
    --base-model "${VLLM_BASE_MODEL:-}" \
    || {
        echo "WARNING: adapter sync failed (exit $?). vLLM will start without pre-loaded adapters."
        # Ensure a valid (empty) manifest so the vLLM entrypoint doesn't choke
        mkdir -p "${ADAPTERS_DIR}"
        echo "[]" > "${ADAPTERS_DIR}/lora-modules.json"
    }

echo "=== adapter-sync: done ==="
