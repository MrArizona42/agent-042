#!/bin/bash
set -e

# Run database schema migration before starting the server.
# This is idempotent — if the schema is already up-to-date, it's a no-op.
echo "Running MLflow DB schema migration..."
mlflow db upgrade "${MLFLOW_BACKEND_URI}"
echo "Schema migration complete."

# Hand off to the original command (mlflow server ...)
exec mlflow "$@"
