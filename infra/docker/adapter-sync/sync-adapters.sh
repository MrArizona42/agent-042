#!/usr/bin/env bash
# sync-adapters.sh — download aliased adapters from MLflow and
# hot-load them into the running vLLM instance via its REST API.
#
# Exits 0 when no adapters are found (vLLM runs without LoRA modules).
# Exits non-zero when MLflow or vLLM is unreachable (hard failure).
set -euo pipefail

ADAPTERS_DIR="${REGISTRY_ADAPTERS_DIR:-/adapters}"
VLLM_URL="${REGISTRY_VLLM_BASE_URL:-http://vllm:8000}"
SYNC_ALIASES="${REGISTRY_SYNC_ALIASES:-champion,challenger}"

echo "=== adapter-sync: starting ==="
echo "  MLflow URI       : ${MLFLOW_TRACKING_URI:-<not set>}"
echo "  Adapters dir     : ${ADAPTERS_DIR}"
echo "  vLLM URL         : ${VLLM_URL}"
echo "  Sync aliases     : ${SYNC_ALIASES}"

python -m shared.model_registry sync \
    --adapters-dir "${ADAPTERS_DIR}" \
    --vllm-url "${VLLM_URL}" \
    --aliases "${SYNC_ALIASES}"

echo "=== adapter-sync: done ==="
