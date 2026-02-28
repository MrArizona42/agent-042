FROM python:3.12-slim

WORKDIR /app

# Minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python deps from lock file for reproducible builds.
# The lock file pins torch to a +cpu wheel; the extra index lets pip find it.
# To update: uv --no-config pip compile pyproject.toml --extra gateway --torch-backend cpu --python-version 3.12 -o infra/docker/gateway/requirements-gateway.lock
COPY infra/docker/gateway/requirements-gateway.lock /app/
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-gateway.lock

# Copy source
COPY src /app/src

ENV PYTHONPATH=/app/src
EXPOSE 9000

CMD ["python", "-m", "uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "9000"]
