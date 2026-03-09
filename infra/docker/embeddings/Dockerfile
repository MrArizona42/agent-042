FROM python:3.12-slim

WORKDIR /app

# Minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python deps from lock file for reproducible builds.
# To update: uv --no-config pip compile pyproject.toml --extra embeddings --python-version 3.12 --python-platform linux -o infra/docker/embeddings/requirements-embeddings.lock
COPY infra/docker/embeddings/requirements-embeddings.lock /app/
RUN pip install -r requirements-embeddings.lock

# Copy source
COPY src /app/src

ENV PYTHONPATH=/app/src
EXPOSE 8100

CMD ["python", "-m", "uvicorn", "embeddings.main:app", "--host", "0.0.0.0", "--port", "8100"]
