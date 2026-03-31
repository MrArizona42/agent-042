"""Configuration for the embeddings microservice."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingsSettings(BaseSettings):
    """Service-specific settings for the standalone embeddings HTTP server.

    Only contains fields unique to the embeddings microservice (host, port).
    Shared fields (model, device, batch_size) are read from
    ``shared.config.Settings`` via ``get_settings()``.

    Environment Variables:
        EMBEDDINGS_HOST: Host to bind the HTTP server to.
        EMBEDDINGS_PORT: Port to bind the HTTP server to.
    """

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDINGS_",
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    host: str = Field(
        default="0.0.0.0",
        description="Host to bind the HTTP server to",
    )
    port: int = Field(
        default=8100,
        description="Port to bind the HTTP server to",
        ge=1,
        le=65535,
    )


@lru_cache
def get_embeddings_settings() -> EmbeddingsSettings:
    """Get cached embeddings service settings."""
    return EmbeddingsSettings()
