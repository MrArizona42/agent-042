#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Dump current Docker Compose logs into artifacts/infra/compose_logs/<service>.log
#
# Usage:
#   bash scripts/dump_docker_logs.sh            # all services
#   bash scripts/dump_docker_logs.sh gateway ui  # specific services
#
# Uses the canonical repo-root `.env` by default. Override with
# COMPOSE_ENV_FILE=/absolute/path/to/.env if needed.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${COMPOSE_ENV_FILE:-$REPO_ROOT/.env}"

[[ -f "$ENV_FILE" ]] || {
    echo "Env file not found: $ENV_FILE" >&2
    exit 1
}

read_env_value() {
    local name="$1"
    awk -F= -v key="$name" '
        $0 !~ /^[[:space:]]*#/ && $1 == key {
            value = substr($0, index($0, "=") + 1)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            gsub(/^"|"$/, "", value)
            print value
            exit
        }
    ' "$ENV_FILE"
}

compose_project_name="$(read_env_value COMPOSE_PROJECT_NAME)"
project_root="$(read_env_value PROJECT_ROOT)"

[[ -n "$compose_project_name" ]] || {
    echo "COMPOSE_PROJECT_NAME is missing in $ENV_FILE" >&2
    exit 1
}
[[ -n "$project_root" ]] || {
    echo "PROJECT_ROOT is missing in $ENV_FILE" >&2
    exit 1
}

LOG_DIR="$project_root/artifacts/infra/compose_logs"
compose_args=(--env-file "$ENV_FILE" --project-name "$compose_project_name")

mkdir -p "$LOG_DIR"

# Resolve services: args or all running services
if [[ $# -gt 0 ]]; then
    services=("$@")
else
    mapfile -t services < <(
        docker compose "${compose_args[@]}" ps -a --format '{{.Service}}' 2>/dev/null
    )
fi

if [[ ${#services[@]} -eq 0 ]]; then
    echo "No running services found."
    exit 0
fi

for svc in "${services[@]}"; do
    outfile="$LOG_DIR/${svc}.log"
    echo "Dumping $svc → $outfile"
    docker compose "${compose_args[@]}" \
        logs --no-color --no-log-prefix "$svc" > "$outfile" 2>/dev/null || \
        echo "  (no logs for $svc)"
done

echo "Done. Logs saved to $LOG_DIR/"
