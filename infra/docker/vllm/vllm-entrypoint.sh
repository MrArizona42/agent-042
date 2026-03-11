#!/usr/bin/env bash
# vllm-entrypoint.sh — conditionally inject --lora-modules when a
# manifest exists, then exec the upstream vLLM OpenAI server.
set -euo pipefail

LORA_MANIFEST="/adapters/lora-modules.json"

EXTRA_ARGS=()

# If the adapter-sync init container produced a non-empty manifest, tell
# vLLM to pre-load those adapters at startup.
if [ -f "${LORA_MANIFEST}" ]; then
    # Check the file is a non-empty JSON array (not just "[]")
    entry_count=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(len(d))" "${LORA_MANIFEST}" 2>/dev/null || echo 0)
    if [ "${entry_count}" -gt 0 ]; then
        echo "vllm-entrypoint: loading ${entry_count} LoRA adapter(s) from ${LORA_MANIFEST}"
        EXTRA_ARGS+=("--lora-modules" "@${LORA_MANIFEST}")
    else
        echo "vllm-entrypoint: manifest exists but is empty — no adapters to pre-load"
    fi
else
    echo "vllm-entrypoint: no lora-modules.json found — starting without pre-loaded adapters"
fi

# exec the upstream entrypoint (vllm serve), forwarding all original args + extras
exec python3 -m vllm.entrypoints.openai.api_server "$@" "${EXTRA_ARGS[@]}"
