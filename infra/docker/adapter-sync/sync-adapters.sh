#!/usr/bin/env bash
# sync-adapters.sh — download aliased adapters from MLflow and
# hot-load them into the running vLLM instance via its REST API.
#
# Exits 0 when no adapters are found (vLLM runs without LoRA modules).
# Exits non-zero when MLflow or vLLM is unreachable (hard failure).
set -euo pipefail

echo "=== adapter-sync: starting ==="
echo "  MLflow URI       : ${MLFLOW_TRACKING_URI:-<not set>}"
echo "  Adapters dir env : ${REGISTRY_ADAPTERS_DIR:-<shared default>}"
echo "  vLLM URL env     : ${GATEWAY_VLLM_BASE_URL:-${REGISTRY_VLLM_BASE_URL:-<shared default>}}"
echo "  Sync aliases env : ${REGISTRY_SYNC_ALIASES:-<shared default>}"

python -m shared.model_registry sync

echo "=== adapter-sync: done ==="
