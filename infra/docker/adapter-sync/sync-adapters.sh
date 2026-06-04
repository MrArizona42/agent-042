#!/usr/bin/env bash
# sync-adapters.sh — download aliased adapters from MLflow and
# hot-load them into the running vLLM instance via its REST API.
#
# Exits 0 when no adapters are found (vLLM runs without LoRA modules).
# Exits non-zero when MLflow or vLLM is unreachable (hard failure).
set -euo pipefail

echo "=== adapter-sync: starting ==="
echo "  MLflow URI env   : ${PLATFORM__MLFLOW_TRACKING_URI:-<not set>}"
echo "  Adapters dir env : ${ADAPTER_REGISTRY__ADAPTERS_DIR:-<shared default>}"
echo "  vLLM URL env     : ${PLATFORM__VLLM_BASE_URL:-<shared default>}"
echo "  Sync aliases env : ${ADAPTER_REGISTRY__SYNC_ALIASES:-<shared default>}"

python -m shared.model_registry sync

echo "=== adapter-sync: done ==="
