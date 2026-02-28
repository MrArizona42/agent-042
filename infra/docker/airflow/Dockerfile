# Custom Airflow image with pre-installed DAG dependencies.
# Base version must stay in sync with the one used in docker-compose.yaml.
FROM apache/airflow:2.10.4

# Install DAG Python dependencies at build time so containers start fast.
# --chown is required because the base image runs as the non-root "airflow"
# user, so the copied file must be owned by that user for the subsequent
# rm to succeed.
# Install from lock file for reproducible builds.
# The lock file pins torch to a +cpu wheel; the extra index lets pip find it.
# To update: uv --no-config pip compile --torch-backend cpu --python-version 3.12 dags/requirements.txt -o dags/requirements.lock
COPY --chown=airflow:root dags/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /tmp/requirements.lock \
    && rm /tmp/requirements.lock
