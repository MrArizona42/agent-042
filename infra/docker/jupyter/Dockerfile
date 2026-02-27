FROM quay.io/jupyter/minimal-notebook:latest

# Install as root so system-level pip deps land in the shared site-packages
USER root

COPY infra/docker/jupyter/requirements-jupyter.txt /tmp/requirements-jupyter.txt

RUN pip install --no-cache-dir -r /tmp/requirements-jupyter.txt

# Restore unprivileged notebook user
USER ${NB_USER}
