#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Wait for PostgreSQL to be fully ready before starting the server.
# Docker Compose healthcheck already gates startup, but a belt-and-suspenders
# check avoids transient issues on slow hosts.
# ---------------------------------------------------------------------------
PG_MAX_RETRIES=30
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
# NOTE: We intentionally do NOT run "mlflow db upgrade" here.
# The "mlflow server" command calls _safe_initialize_tables() on startup,
# which correctly handles BOTH fresh databases (creates base tables first,
# then applies Alembic migrations) and existing databases (applies only
# pending migrations).  Running "mlflow db upgrade" directly would fail on
# a fresh database because Alembic migrations assume the base tables
# already exist (e.g. ALTER TABLE metrics ADD COLUMN step …).
# ---------------------------------------------------------------------------

# Hand off to the original command (mlflow server ...)
exec mlflow "$@"
