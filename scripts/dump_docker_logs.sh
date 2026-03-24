#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Dump current Docker Compose logs into infra/compose/logs/<service>.log
#
# Usage:
#   bash scripts/dump_docker_logs.sh            # all services
#   bash scripts/dump_docker_logs.sh gateway ui  # specific services
#
# Requires: docker compose, must be run from the project root or with
# COMPOSE_FILE pointing to infra/compose/docker-compose.yaml.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_DIR="$PROJECT_ROOT/infra/compose"
LOG_DIR="$COMPOSE_DIR/logs"

mkdir -p "$LOG_DIR"

# Resolve services: args or all running services
if [[ $# -gt 0 ]]; then
    services=("$@")
else
    mapfile -t services < <(
        docker compose -f "$COMPOSE_DIR/docker-compose.yaml" ps -a --format '{{.Service}}' 2>/dev/null
    )
fi

if [[ ${#services[@]} -eq 0 ]]; then
    echo "No running services found."
    exit 0
fi

for svc in "${services[@]}"; do
    outfile="$LOG_DIR/${svc}.log"
    echo "Dumping $svc → $outfile"
    docker compose -f "$COMPOSE_DIR/docker-compose.yaml" \
        logs --no-color --no-log-prefix "$svc" > "$outfile" 2>/dev/null || \
        echo "  (no logs for $svc)"
done

echo "Done. Logs saved to $LOG_DIR/"
