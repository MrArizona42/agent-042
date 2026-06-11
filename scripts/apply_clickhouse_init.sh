#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-infra/compose/docker-compose.yaml}"
ENV_FILE="${ENV_FILE:-.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Set ENV_FILE=/path/to/.env if the deployment env file lives elsewhere." >&2
  exit 1
fi

docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" exec -T clickhouse \
  sh -lc 'clickhouse-client --multiquery < /docker-entrypoint-initdb.d/001_inference_events.sql'

docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" exec -T clickhouse \
  clickhouse-client --database agent042_analytics --query "SHOW TABLES"
