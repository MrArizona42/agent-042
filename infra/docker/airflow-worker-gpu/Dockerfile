# Airflow Celery GPU worker image with training dependencies.
#
# Extends the base Airflow image with packages required for GPU-based
# training tasks (LoRA fine-tuning via PyTorch Lightning, PEFT, etc.)
# as well as all CPU worker dependencies (eval metrics, RAG pipelines).
#
# Unlike the CPU worker, this image does NOT use the PyTorch CPU-only
# index — it installs the full CUDA-enabled torch from PyPI.
#
# The scheduler, webserver, and dag-processor continue to use the slim
# base image — only workers need the heavy dependencies.
#
# Base version must stay in sync with infra/docker/airflow/Dockerfile.
FROM apache/airflow:3.1.8

# Install DAG task dependencies for the GPU Celery worker.
# The --constraint file prevents overriding Airflow-critical packages
# (protobuf, grpcio, opentelemetry, cryptography) that are shipped with
# the base image.
#
# To update the lock file:
#   scripts/update_locks.sh airflow-worker-gpu
COPY --chown=airflow:root infra/docker/airflow-worker-gpu/requirements-airflow-worker-gpu.lock /tmp/requirements.txt
COPY --chown=airflow:root infra/docker/airflow/airflow-core-constraints.txt /tmp/constraints.txt
RUN pip install --no-cache-dir \
    --constraint /tmp/constraints.txt \
    -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt /tmp/constraints.txt
