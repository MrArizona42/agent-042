#!/usr/bin/env bash
# Apply the agent042 Postgres schema: control-plane tables and the eval_runs
# release/deployment column migration. Creates the agent042 database itself
# if it doesn't exist yet.
#
# Idempotent -- safe to run unconditionally, including against a database
# that already has some or all of these tables/columns. Run this before
# gateway, RAG operations, or Airflow use the declarative alias workflow's
# schema. `Base.metadata.create_all()` creates tables for fresh databases but
# never alters existing ones, so this explicit migration step stays required
# even after the ORM models exist. `deploy_release.sh` calls this
# automatically on every release deploy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

COMPOSE_FILE="${COMPOSE_FILE:-$REPO_ROOT/infra/compose/docker-compose.yaml}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"
DB_DIR="$REPO_ROOT/migrations/postgres"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Set ENV_FILE=/path/to/.env if the deployment env file lives elsewhere." >&2
  exit 1
fi

POSTGRES_APP_DB="$(awk -F= '/^[[:space:]]*POSTGRES_APP_DB=/ {print substr($0, index($0, "=") + 1); exit}' "${ENV_FILE}")"
POSTGRES_USER="$(awk -F= '/^[[:space:]]*POSTGRES_USER=/ {print substr($0, index($0, "=") + 1); exit}' "${ENV_FILE}")"

if [[ -z "${POSTGRES_APP_DB}" || -z "${POSTGRES_USER}" ]]; then
  echo "Missing POSTGRES_APP_DB or POSTGRES_USER in ${ENV_FILE}." >&2
  exit 1
fi

# POSTGRES_APP_DB has no creation path elsewhere: the official postgres image
# only auto-creates the database named by POSTGRES_DB (mlflow's backend
# store), and Airflow creates its own database itself. Connect to the
# always-present "postgres" maintenance database to check/create it so this
# script is safe to call unconditionally on a brand-new server.
db_exists="$(docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" exec -T postgres \
  psql -v ON_ERROR_STOP=1 -U "${POSTGRES_USER}" -d postgres -tAc \
  "SELECT 1 FROM pg_database WHERE datname = '${POSTGRES_APP_DB}'")"

if [[ "${db_exists}" == "1" ]]; then
  echo "[apply] database ${POSTGRES_APP_DB} already exists"
else
  echo "[apply] creating database ${POSTGRES_APP_DB}"
  docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" exec -T postgres \
    createdb -U "${POSTGRES_USER}" "${POSTGRES_APP_DB}"
fi

# Dependency order: rag_releases and rag_release_builds have no foreign
# keys; rag_alias_deployments references rag_releases; the eval_runs
# migration references both.
MIGRATIONS=(
  "eval_runs.sql"
  "eval_samples.sql"
  "eval_runs_add_rag_observability_columns.sql"
  "chat_messages_add_usage_columns.sql"
  "rag_release_builds.sql"
  "rag_releases.sql"
  "rag_alias_deployments.sql"
  "eval_runs_add_release_columns.sql"
)

for migration in "${MIGRATIONS[@]}"; do
  path="$DB_DIR/$migration"
  if [[ ! -f "$path" ]]; then
    echo "Missing migration file: $path" >&2
    exit 1
  fi
  echo "[apply] $migration"
  docker compose --env-file "${ENV_FILE}" -f "${COMPOSE_FILE}" exec -T postgres \
    psql -v ON_ERROR_STOP=1 -U "${POSTGRES_USER}" -d "${POSTGRES_APP_DB}" < "$path"
done

echo "All agent042 DB migrations applied."
