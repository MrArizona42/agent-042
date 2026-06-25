#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash ops/model_registry_ops.sh <args...>

Runs services.adapter_sync.model_registry's own CLI
(python -m services.adapter_sync.model_registry) inside the vllm-adapter-sync
container, which already has Compose-injected env vars -- no host-side .env
loading needed.

Examples:
  bash ops/model_registry_ops.sh list
  bash ops/model_registry_ops.sh sync --adapters_dir=/adapters
EOF
}

[[ $# -gt 0 ]] || {
  usage >&2
  exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
compose_file="$project_root/infra/compose/docker-compose.yaml"
env_file="${MODEL_REGISTRY_OPS_ENV_FILE:-$project_root/.env}"

[[ -f "$env_file" ]] || {
  echo "error: env file not found: $env_file" >&2
  exit 1
}

[[ -f "$compose_file" ]] || {
  echo "error: compose file not found: $compose_file" >&2
  exit 1
}

compose_project_name="$(awk -F= '/^[[:space:]]*COMPOSE_PROJECT_NAME=/{print substr($0, index($0, "=") + 1); exit}' "$env_file")"
if [[ -z "$compose_project_name" ]]; then
  compose_project_name="$(basename "$PWD")"
fi

COMPOSE_PROJECT_NAME="$compose_project_name" docker compose \
  --project-name "$compose_project_name" \
  --env-file "$env_file" \
  -f "$compose_file" \
  run --rm --entrypoint python vllm-adapter-sync -m services.adapter_sync.model_registry "$@"
