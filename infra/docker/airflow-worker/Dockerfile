# Airflow Celery worker image with ALL DAG task dependencies.
#
# Extends the base Airflow image with packages required to execute
# every task type: evaluation (bert-score / torch CPU), RAG data
# pipelines (arxiv, qdrant-client, dvc), and collection cleanup.
#
# The scheduler, webserver, and dag-processor continue to use the slim
# base image — only the worker needs the heavy dependencies.
#
# Base version must stay in sync with infra/docker/airflow/Dockerfile.
FROM apache/airflow:3.1.8

# Install Firejail for sandboxed HumanEval code execution.
# Firejail is a SUID-root binary that uses Linux namespaces + seccomp-bpf to
# isolate each generated-code subprocess (--net=none, rlimits).  No Docker
# socket is required, so the worker container stays unprivileged.
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends firejail \
    && rm -rf /var/lib/apt/lists/*
USER airflow

# Install DAG task dependencies for the Celery worker.
# torch is installed from the CPU-only index to avoid pulling CUDA wheels.
# The --constraint file prevents overriding Airflow-critical packages
# (protobuf, grpcio, opentelemetry, cryptography) that are shipped with
# the base image.
#
# To update the lock file:
#   scripts/update_locks.sh airflow-worker
COPY --chown=airflow:root infra/docker/airflow-worker/requirements-airflow-worker.lock /tmp/requirements.txt
COPY --chown=airflow:root infra/docker/airflow/airflow-core-constraints.txt /tmp/constraints.txt
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --constraint /tmp/constraints.txt \
    -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt /tmp/constraints.txt
