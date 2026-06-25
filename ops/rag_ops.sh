#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash ops/rag_ops.sh python -m rag.cli.app <args...>

Examples:
  bash ops/rag_ops.sh python -m rag.cli.app catalog validate

  bash ops/rag_ops.sh python -m rag.cli.app alias diff pytorch_reference challenger

  bash ops/rag_ops.sh python -m rag.cli.app alias apply pytorch_reference challenger

  bash ops/rag_ops.sh python -m rag.cli.app alias apply pytorch_reference champion \
    --allow-build-default --allow-unevaluated

  bash ops/rag_ops.sh python -m rag.cli.app release list --kb pytorch_reference

  bash ops/rag_ops.sh python -m rag.cli.app benchmark run --kb pytorch_reference --alias challenger
EOF
}

[[ $# -gt 0 ]] || {
  usage >&2
  exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
compose_file="$project_root/infra/compose/docker-compose.yaml"
env_file="${RAG_OPS_ENV_FILE:-$project_root/.env}"

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

compose_args=(
  --project-name "$compose_project_name"
  --env-file "$env_file"
  -f "$compose_file"
)

interactive_args=()
if [[ ! -t 0 || ! -t 1 ]]; then
  interactive_args=(-T)
fi

if container_id="$(COMPOSE_PROJECT_NAME="$compose_project_name" docker compose "${compose_args[@]}" ps -q ops)" \
  && [[ -n "$container_id" ]] \
  && [[ "$(docker inspect --format '{{.State.Running}}' "$container_id" 2>/dev/null || true)" == "true" ]]; then
  COMPOSE_PROJECT_NAME="$compose_project_name" docker compose "${compose_args[@]}" \
    exec "${interactive_args[@]}" ops "$@"
else
  COMPOSE_PROJECT_NAME="$compose_project_name" docker compose "${compose_args[@]}" \
    run --rm "${interactive_args[@]}" ops "$@"
fi
