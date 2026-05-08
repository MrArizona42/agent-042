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

LOG_DIR="/home/anton-m/agent-042/artifacts/infra/compose_logs"

compose_args=(--project-name "agent-042")

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
