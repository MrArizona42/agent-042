#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Dump current Docker Compose logs into artifacts/infra/logs/<service>.log
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
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-$PROJECT_ROOT/infra/compose/docker-compose.yaml}"
COMPOSE_ENV_FILE="${COMPOSE_ENV_FILE:-$PROJECT_ROOT/.env}"
LOG_DIR="$PROJECT_ROOT/artifacts/infra/logs"

if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo "Compose file not found: $COMPOSE_FILE" >&2
    exit 1
fi

if [[ ! -f "$COMPOSE_ENV_FILE" ]]; then
    echo "Compose env file not found: $COMPOSE_ENV_FILE" >&2
    echo "Expected the canonical repo-root .env or an explicit COMPOSE_ENV_FILE override." >&2
    exit 1
fi

compose_args=(--env-file "$COMPOSE_ENV_FILE" -f "$COMPOSE_FILE")

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
