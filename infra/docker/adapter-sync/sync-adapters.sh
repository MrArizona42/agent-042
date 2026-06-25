#!/usr/bin/env bash
# sync-adapters.sh — download aliased adapters from MLflow and
# hot-load them into the running vLLM instance via its REST API.
#
# Exits 0 when no adapters are found (vLLM runs without LoRA modules).
# Exits non-zero when MLflow or vLLM is unreachable (hard failure).
set -euo pipefail

echo "=== adapter-sync: starting ==="
echo "  Runtime config   : ${CONFIG__RUNTIME_PATH:-<not set>}"
echo "  vLLM service     : ${NETWORK__VLLM__INTERNAL_HOST:-<not set>}:${NETWORK__VLLM__INTERNAL_PORT:-<not set>}"
echo "  MLflow service   : ${NETWORK__MLFLOW__INTERNAL_HOST:-<not set>}:${NETWORK__MLFLOW__INTERNAL_PORT:-<not set>}"

python -m services.adapter_sync.model_registry sync --adapters-dir /adapters

echo "=== adapter-sync: done ==="
