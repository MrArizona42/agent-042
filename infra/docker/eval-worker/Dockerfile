FROM python:3.12-slim

WORKDIR /app

# Minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python deps from lock file for reproducible builds.
# torch is installed from the CPU-only index to avoid pulling CUDA wheels.
# To update:
#   uv --no-config pip compile pyproject.toml --extra eval-worker \
#       --python-version 3.12 --python-platform linux \
#       --extra-index-url https://download.pytorch.org/whl/cpu \
#       --index-strategy unsafe-best-match \
#       -o infra/docker/eval-worker/requirements-eval-worker.lock
COPY infra/docker/eval-worker/requirements-eval-worker.lock /app/
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-eval-worker.lock \
    && rm requirements-eval-worker.lock

# Copy source and experiment scripts
COPY src /app/src
COPY experiments /app/experiments

ENV PYTHONPATH=/app/src:/app

# Run Celery worker
# -Q eval: listen only to the eval queue
# -c 1: single concurrency (CPU-bound metric computation)
# -P solo: solo pool (simpler, no fork overhead)
CMD ["celery", "-A", "eval_worker.celery_app", "worker", "--loglevel=info", "-Q", "eval", "-c", "1", "-P", "solo"]
