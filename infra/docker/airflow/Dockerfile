# Custom Airflow image with pre-installed DAG dependencies.
# Base version must stay in sync with the one used in docker-compose.yaml.
FROM apache/airflow:3.1.8

# Install DAG Python dependencies at build time so containers start fast.
# --chown is required because the base image runs as the non-root "airflow"
# user, so the copied file must be owned by that user for the subsequent
# rm to succeed.
# Install from lock file for reproducible builds.
# The lock file MUST be compiled with Airflow's core constraints to prevent
# overriding critical packages (protobuf, grpcio, opentelemetry, etc.) that
# break the Execution API and JWT authentication.
# To update:
#   uv --no-config pip compile pyproject.toml --extra airflow \
#       --python-version 3.12 --python-platform linux \
#       --constraint infra/docker/airflow/airflow-core-constraints.txt \
#       -o infra/docker/airflow/requirements.lock
COPY --chown=airflow:root infra/docker/airflow/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock \
    && rm /tmp/requirements.lock
