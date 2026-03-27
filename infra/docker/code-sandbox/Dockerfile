FROM python:3.13-slim

# No extra packages — stdlib only.  A minimal image reduces attack surface
# for any code that escapes the subprocess.
COPY infra/docker/code-sandbox/server.py /app/server.py

# Run as a non-root user; the container filesystem is mounted read-only by
# compose, so /tmp must be a tmpfs (also configured in compose).
RUN useradd --no-create-home --shell /bin/false sandbox
USER sandbox

EXPOSE 8200
CMD ["python3", "/app/server.py"]
