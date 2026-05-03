FROM python:3.12-slim

WORKDIR /app

# Minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python deps from lock file for reproducible builds.
# To update: uv --no-config pip compile pyproject.toml --extra reranker --python-version 3.12 --python-platform linux -o infra/docker/reranker/requirements-reranker.lock
COPY infra/docker/reranker/requirements-reranker.lock /app/
RUN pip install --no-cache-dir -r requirements-reranker.lock

# Copy source
COPY src /app/src

ENV PYTHONPATH=/app/src
EXPOSE 8101

CMD ["python", "-m", "uvicorn", "reranker.main:app", "--host", "0.0.0.0", "--port", "8101"]
