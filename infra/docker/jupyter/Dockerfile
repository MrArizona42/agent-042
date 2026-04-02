FROM quay.io/jupyter/minimal-notebook:python-3.13.12

# Install as root so system-level pip deps land in the shared site-packages
USER root

# Install from lock file for reproducible builds.
# To update: uv --no-config pip compile pyproject.toml --extra training --extra rag --extra dev --extra mlflow --python-version 3.13 --python-platform linux -o infra/docker/jupyter/requirements-jupyter.lock
COPY infra/docker/jupyter/requirements-jupyter.lock /tmp/requirements-jupyter.lock

RUN pip install --no-cache-dir -r /tmp/requirements-jupyter.lock

# Restore unprivileged notebook user
USER ${NB_USER}
