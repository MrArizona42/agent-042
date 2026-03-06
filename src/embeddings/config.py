"""Configuration for the embeddings microservice."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingsSettings(BaseSettings):
    """Settings for the standalone embeddings service.

    Environment Variables:
        EMBEDDINGS_MODEL: HuggingFace model identifier for embeddings.
        EMBEDDINGS_DEVICE: Device to run model on (cpu, cuda, mps).
        EMBEDDINGS_BATCH_SIZE: Maximum batch size for encoding.
        EMBEDDINGS_HOST: Host to bind the HTTP server to.
        EMBEDDINGS_PORT: Port to bind the HTTP server to.
    """

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDINGS_",
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model for embeddings",
    )
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Device for embedding model (cpu, cuda, mps)",
    )
    batch_size: int = Field(
        default=32,
        description="Maximum batch size for encoding",
        ge=1,
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
def get_settings() -> EmbeddingsSettings:
    """Get cached embeddings service settings."""
    return EmbeddingsSettings()
