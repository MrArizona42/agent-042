#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

COMPOSE_FILE="${COMPOSE_FILE:-$REPO_ROOT/infra/compose/docker-compose.yaml}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Set ENV_FILE=/path/to/.env if the deployment env file lives elsewhere." >&2
  exit 1
fi

CLICKHOUSE_DB="$(awk -F= '/^[[:space:]]*CLICKHOUSE_DB=/ {print substr($0, index($0, "=") + 1); exit}' "${ENV_FILE}")"
if [[ -z "${CLICKHOUSE_DB}" ]]; then
  echo "Missing CLICKHOUSE_DB in ${ENV_FILE}." >&2
  exit 1
fi

docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" exec -T clickhouse \
  sh -lc 'clickhouse-client --multiquery < /docker-entrypoint-initdb.d/001_inference_events.sql'

docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" exec -T clickhouse \
  clickhouse-client --database "${CLICKHOUSE_DB}" --query "SHOW TABLES"
