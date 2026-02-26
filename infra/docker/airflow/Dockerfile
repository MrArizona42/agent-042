# Custom Airflow image with pre-installed DAG dependencies.
# Base version must stay in sync with the one used in docker-compose.yaml.
FROM apache/airflow:2.10.4

# Install DAG Python dependencies at build time so containers start fast.
# --chown is required because the base image runs as the non-root "airflow"
# user, so the copied file must be owned by that user for the subsequent
# rm to succeed.
COPY --chown=airflow:root dags/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
