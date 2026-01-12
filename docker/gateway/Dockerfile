FROM python:3.12-slim

WORKDIR /app

# Minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python deps
COPY requirements-gateway.txt /app/
RUN pip install --no-cache-dir -r requirements-gateway.txt

# Copy source
COPY src /app/src

ENV PYTHONPATH=/app/src
EXPOSE 9000

CMD ["python", "-m", "uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "9000"]

