# Lightweight init-container that syncs champion LoRA adapters from MLflow
# Model Registry into the shared /adapters volume.
# vLLM picks up the generated lora-modules.json on startup.
#
# To update lock file:
#   uv --no-config pip compile pyproject.toml --extra mlflow --python-version 3.12 --python-platform linux -o infra/docker/adapter-sync/requirements-adapter-sync.lock
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY infra/docker/adapter-sync/requirements-adapter-sync.lock /app/
RUN pip install -r requirements-adapter-sync.lock

# Only the shared module is needed
COPY src/shared /app/src/shared

ENV PYTHONPATH=/app/src

COPY infra/docker/adapter-sync/sync-adapters.sh /app/sync-adapters.sh
RUN chmod +x /app/sync-adapters.sh

ENTRYPOINT ["/app/sync-adapters.sh"]
