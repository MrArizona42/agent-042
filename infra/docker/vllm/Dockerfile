# Thin wrapper around the official vLLM image.
# Adds an entrypoint that conditionally injects --lora-modules when
# the adapter-sync init container has produced a manifest.
FROM vllm/vllm-openai:v0.16.0

COPY infra/docker/vllm/vllm-entrypoint.sh /opt/vllm-entrypoint.sh
RUN chmod +x /opt/vllm-entrypoint.sh

ENTRYPOINT ["/opt/vllm-entrypoint.sh"]
