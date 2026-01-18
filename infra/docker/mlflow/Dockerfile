# Base Python image (small, stable, well-supported)
FROM python:3.12-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure logs appear immediately (important for containers)
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Install system dependencies
# curl → healthchecks
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# mlflow → core service
# psycopg2-binary → PostgreSQL backend store
# boto3 → S3 / Yandex Object Storage support
RUN pip install --no-cache-dir \
    mlflow \
    psycopg2-binary \
    boto3

# Expose MLflow default port (documentation only)
EXPOSE 5000

# ENTRYPOINT defines the executable
# This makes the container behave like "mlflow <args>"
ENTRYPOINT ["mlflow"]
