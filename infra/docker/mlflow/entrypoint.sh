#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Wait for PostgreSQL to be fully ready before running migrations.
# pg_isready only checks TCP; we also verify we can connect to the target DB.
# ---------------------------------------------------------------------------
PG_MAX_RETRIES=30
MIGRATION_MAX_RETRIES=5
RETRY_INTERVAL=2

echo "Waiting for PostgreSQL to accept connections..."
for i in $(seq 1 "$PG_MAX_RETRIES"); do
    if pg_isready -h postgres -p 5432 -q 2>/dev/null; then
        echo "PostgreSQL is ready (attempt $i/$PG_MAX_RETRIES)."
        break
    fi
    if [ "$i" -eq "$PG_MAX_RETRIES" ]; then
        echo "ERROR: PostgreSQL did not become ready in time." >&2
        exit 1
    fi
    sleep "$RETRY_INTERVAL"
done

# ---------------------------------------------------------------------------
# Run database schema migration before starting the server.
# This is idempotent — if the schema is already up-to-date, it's a no-op.
# ---------------------------------------------------------------------------
echo "Running MLflow DB schema migration..."
for i in $(seq 1 "$MIGRATION_MAX_RETRIES"); do
    if mlflow db upgrade "${MLFLOW_BACKEND_URI}"; then
        echo "Schema migration complete."
        break
    fi
    if [ "$i" -eq "$MIGRATION_MAX_RETRIES" ]; then
        echo "ERROR: MLflow DB migration failed after $MIGRATION_MAX_RETRIES attempts." >&2
        exit 1
    fi
    echo "Migration attempt $i failed, retrying in ${RETRY_INTERVAL}s..."
    sleep "$RETRY_INTERVAL"
done

# Hand off to the original command (mlflow server ...)
exec mlflow "$@"
