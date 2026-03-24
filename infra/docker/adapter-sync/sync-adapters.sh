#!/usr/bin/env bash
# sync-adapters.sh — download production adapters from MLflow and write
# /adapters/lora-modules.json for vLLM.
#
# Exits 0 when no adapters are found (vLLM starts without LoRA modules).
# Exits non-zero when MLflow is unreachable (hard failure).
set -euo pipefail

ADAPTERS_DIR="${REGISTRY_ADAPTERS_DIR:-/adapters}"

echo "=== adapter-sync: starting ==="
echo "  MLflow URI       : ${MLFLOW_BACKEND_URI:-<not set>}"
echo "  Adapters dir     : ${ADAPTERS_DIR}"
echo "  Production alias : from shared.config.ModelRegistrySettings"

SYNC_ARGS=("--adapters-dir" "${ADAPTERS_DIR}" "--base-model" "${VLLM_BASE_MODEL:-}")

python -m shared.model_registry sync "${SYNC_ARGS[@]}"

echo "=== adapter-sync: done ==="
