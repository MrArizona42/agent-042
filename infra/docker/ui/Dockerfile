FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python deps from lock file for reproducible builds.
# To update: uv --no-config pip compile pyproject.toml --extra ui --python-version 3.12 --python-platform linux -o infra/docker/ui/requirements-ui.lock
COPY infra/docker/ui/requirements-ui.lock /app/
RUN pip install --no-cache-dir -r requirements-ui.lock

COPY src /app/src

# Streamlit configuration (CORS, domain settings, etc.)
COPY src/ui/.streamlit /app/.streamlit

ENV PYTHONPATH=/app/src
EXPOSE 8501

# Note: server.address and port are also in .streamlit/config.toml
# but we keep them here for clarity and as override capability
CMD ["python", "-m", "streamlit", "run", "src/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
