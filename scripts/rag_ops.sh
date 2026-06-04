#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/rag_ops.sh python -m rag.sources.cli <args...>

Examples:
  bash scripts/rag_ops.sh python -m rag.sources.cli build-source \
    --catalog src/shared/catalog.toml \
    --kb pytorch_reference \
    --source docs \
    --rag-data-root assets/rag_data \
    --limit 1

  bash scripts/rag_ops.sh python -m rag.sources.cli collect-bundle \
    --catalog src/shared/catalog.toml \
    --kb pytorch_reference \
    --source docs \
    --rag-data-root assets/rag_data \
    --limit 1

  bash scripts/rag_ops.sh python -m rag.sources.cli materialize \
    --catalog src/shared/catalog.toml \
    --kb pytorch_reference \
    --source docs \
    --alias challenger \
    --rag-data-root assets/rag_data \
    --limit 1

  bash scripts/rag_ops.sh python -m rag.sources.cli promote-alias \
    --kb pytorch_reference \
    --alias challenger \
    --collection <collection_name>
EOF
}

[[ $# -gt 0 ]] || {
  usage >&2
  exit 1
}

env_file=.env
compose_file=infra/compose/docker-compose.yaml

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
  --profile ops \
  --env-file "$env_file" \
  -f "$compose_file" \
  run --rm rag-ops "$@"