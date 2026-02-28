# Base Python image (small, stable, well-supported)
FROM python:3.12-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure logs appear immediately (important for containers)
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Install system dependencies
# curl → healthchecks, postgresql-client → pg_isready for wait logic
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from lock file for reproducible builds.
# To update: uv --no-config pip compile --python-version 3.12 requirements-mlflow.txt -o requirements-mlflow.lock
COPY infra/docker/mlflow/requirements-mlflow.lock /app/
RUN pip install --no-cache-dir -r requirements-mlflow.lock

# Expose MLflow default port (documentation only)
EXPOSE 5000

# Copy entrypoint script that runs DB migrations before starting
COPY infra/docker/mlflow/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ENTRYPOINT runs migrations then exec's "mlflow <args>"
ENTRYPOINT ["/app/entrypoint.sh"]
