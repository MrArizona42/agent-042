FROM python:3.12-slim

WORKDIR /app

# Minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python deps
COPY infra/docker/celery/requirements-celery.txt /app/
RUN pip install --no-cache-dir -r requirements-celery.txt

# Copy source
COPY src /app/src

ENV PYTHONPATH=/app/src

# Run Celery worker
# -Q default: listen to default queue
# -c 1: single worker (GPU bound)
# -P solo: solo pool (simpler, no fork issues with GPU)
CMD ["celery", "-A", "worker.celery_app", "worker", "--loglevel=info", "-Q", "celery", "-c", "1", "-P", "solo"]
