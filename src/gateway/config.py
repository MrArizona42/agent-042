from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    """Runtime config for the gateway.

    All settings are configurable via environment variables.
    """

    model_config = SettingsConfigDict(env_prefix="GATEWAY_", extra="ignore")

    # Where vLLM is reachable from the gateway container/process.
    # If you run via docker-compose, this should usually be http://vllm:8000
    # If you run locally, likely http://localhost:8000
    vllm_base_url: str = Field(default="http://localhost:8000")

    # Default model to use when none is specified in the request.
    # This should match the model name served by vLLM.
    default_model: str = Field(default="/models/Qwen/Qwen3-0.6B")

    # Optional safety/auth.
    api_key: str | None = Field(default=None)

    # CORS configuration (comma-separated list in env is supported).
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Metadata for docs.
    service_name: str = Field(default="agent-042-gateway")
    public_base_url: str | None = Field(default=None)


def get_settings() -> GatewaySettings:
    return GatewaySettings()

