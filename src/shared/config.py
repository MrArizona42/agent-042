"""Unified configuration for all services.

This module provides a single source of truth for all configuration settings
across the gateway, RAG, and UI services. Configuration is loaded from
environment variables with sensible defaults for local development.

Usage:
    from shared.config import get_settings

    settings = get_settings()
    print(settings.qdrant_host)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Unified settings for all services.

    All settings are configurable via environment variables with the GATEWAY_ prefix.
    This class consolidates previously scattered configuration from:
    - gateway/config.py (GatewaySettings)
    - RAG/config.py (RAGSettings)
    - ui/app.py (hardcoded defaults)
    - ui/client.py (hardcoded timeouts)

    Environment Variables:
        GATEWAY_VLLM_BASE_URL: vLLM server URL
        GATEWAY_DEFAULT_MODEL: Default model for inference
        GATEWAY_API_KEY: Optional API key for vLLM
        GATEWAY_CORS_ALLOW_ORIGINS: Comma-separated list of allowed origins
        GATEWAY_SERVICE_NAME: Service name for API docs
        GATEWAY_PUBLIC_BASE_URL: Public URL for the gateway
        GATEWAY_QDRANT_HOST: Qdrant server host
        GATEWAY_QDRANT_PORT: Qdrant server port
        GATEWAY_RAG_ENABLED: Enable/disable RAG functionality
        GATEWAY_EMBEDDING_MODEL: Model for generating embeddings
        GATEWAY_EMBEDDING_DEVICE: Device for embedding model (cpu, cuda, mps)
        GATEWAY_TOP_K: Default number of documents to retrieve
        GATEWAY_SCORE_THRESHOLD: Minimum similarity score for retrieval
        GATEWAY_VLLM_TIMEOUT: Timeout for vLLM requests in seconds
        GATEWAY_EMBEDDING_BATCH_SIZE: Batch size for embedding generation
        GATEWAY_CONTEXT_MAX_LENGTH: Maximum context length for RAG
        GATEWAY_CHUNK_SIZE: Default chunk size for document splitting
        GATEWAY_CHUNK_OVERLAP: Default overlap between chunks
        GATEWAY_URL: Full URL to the gateway (used by UI)
        UI_HEALTH_TIMEOUT: Timeout for health check requests
        UI_MODELS_TIMEOUT: Timeout for models list requests
        UI_CHAT_TIMEOUT: Timeout for chat completion requests
    """

    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        extra="ignore",
        # Support loading from .env file if present
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # =========================================================================
    # vLLM / Inference Settings
    # =========================================================================
    vllm_base_url: str = Field(
        default="http://localhost:8000",
        description="URL where vLLM server is reachable",
    )
    default_model: str = Field(
        default="/models/Qwen/Qwen3-0.6B",
        description="Default model when none specified in request",
    )
    api_key: str | None = Field(
        default=None,
        description="Optional API key for vLLM authentication",
    )
    vllm_timeout: float = Field(
        default=60.0,
        description="Timeout for vLLM requests in seconds",
        ge=1.0,
    )

    # =========================================================================
    # Async Inference Settings (Phase 1)
    # =========================================================================
    async_enabled: bool = Field(
        default=True,
        description="Enable async inference via Celery workers",
    )

    # =========================================================================
    # Gateway Service Settings
    # =========================================================================
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed CORS origins (comma-separated in env)",
    )
    service_name: str = Field(
        default="agent-042-gateway",
        description="Service name displayed in API docs",
    )
    public_base_url: str | None = Field(
        default=None,
        description="Public URL for the gateway API",
    )

    # =========================================================================
    # Qdrant / Vector Store Settings
    # =========================================================================
    qdrant_host: str = Field(
        default="localhost",
        description="Qdrant server hostname",
    )
    qdrant_port: int = Field(
        default=6333,
        description="Qdrant server port",
        ge=1,
        le=65535,
    )

    # =========================================================================
    # RAG Settings
    # =========================================================================
    rag_enabled: bool = Field(
        default=True,
        description="Enable RAG functionality",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model for embeddings",
    )
    embedding_device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Device for embedding model",
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation",
        ge=1,
    )
    top_k: int = Field(
        default=5,
        description="Number of documents to retrieve",
        ge=1,
    )
    score_threshold: float = Field(
        default=0.0,
        description="Minimum similarity score for retrieval",
        ge=0.0,
        le=1.0,
    )
    context_max_length: int = Field(
        default=4000,
        description="Maximum character length of RAG context",
        ge=100,
    )

    # =========================================================================
    # Chunking Settings
    # =========================================================================
    chunk_size: int = Field(
        default=512,
        description="Default chunk size for document splitting",
        ge=100,
    )
    chunk_overlap: int = Field(
        default=50,
        description="Default overlap between chunks",
        ge=0,
    )
    code_chunk_size: int = Field(
        default=1000,
        description="Chunk size for code documents",
        ge=100,
    )
    code_chunk_overlap: int = Field(
        default=100,
        description="Overlap for code chunks",
        ge=0,
    )
    section_chunk_size: int = Field(
        default=1024,
        description="Chunk size for section-aware splitting",
        ge=100,
    )
    section_chunk_overlap: int = Field(
        default=100,
        description="Overlap for section chunks",
        ge=0,
    )

    # =========================================================================
    # UI Settings (uses different prefix for some settings)
    # =========================================================================
    # Note: GATEWAY_URL is the full URL to access the gateway from UI
    # This is separate from gateway's internal settings
    url: str = Field(
        default="http://localhost:9001",
        alias="GATEWAY_URL",
        description="Full URL to the gateway (used by UI)",
    )

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse comma-separated CORS origins string to list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


class ModelRegistrySettings(BaseSettings):
    """Settings for MLflow Model Registry / adapter sync.

    Environment Variables:
        REGISTRY_MLFLOW_TRACKING_URI: MLflow tracking server URL.
        REGISTRY_ADAPTERS_DIR: Local directory for downloaded LoRA adapters.
        REGISTRY_AUTO_SYNC: Pull production adapters on service startup.
    """

    model_config = SettingsConfigDict(
        env_prefix="REGISTRY_",
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    mlflow_tracking_uri: str = Field(
        default="http://localhost:5050",
        description="MLflow tracking server URL",
    )
    adapters_dir: str = Field(
        default="./adapters",
        description="Local directory for downloaded LoRA adapters",
    )
    auto_sync: bool = Field(
        default=False,
        description="Automatically sync production adapters on startup",
    )


class UISettings(BaseSettings):
    """UI-specific settings with UI_ prefix.

    These are separate because they use a different environment variable prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="UI_",
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    health_timeout: float = Field(
        default=10.0,
        description="Timeout for health check requests in seconds",
        ge=1.0,
    )
    models_timeout: float = Field(
        default=30.0,
        description="Timeout for models list requests in seconds",
        ge=1.0,
    )
    chat_timeout: float = Field(
        default=300.0,
        description="Timeout for chat completion requests in seconds",
        ge=1.0,
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    Settings are loaded once and cached for the lifetime of the process.
    This ensures consistent configuration and avoids repeated parsing.

    Returns:
        Settings: Validated application settings

    Raises:
        ValidationError: If environment variables contain invalid values
    """
    return Settings()


@lru_cache
def get_registry_settings() -> ModelRegistrySettings:
    """Get cached model registry settings."""
    return ModelRegistrySettings()


@lru_cache
def get_ui_settings() -> UISettings:
    """Get cached UI-specific settings.

    Returns:
        UISettings: Validated UI settings
    """
    return UISettings()


def validate_settings_on_startup() -> None:
    """Validate all settings at application startup.

    Call this function early in your application's lifecycle to fail fast
    if configuration is invalid.

    Raises:
        ValidationError: If any settings are invalid
    """
    # Force settings to be loaded and validated
    settings = get_settings()
    ui_settings = get_ui_settings()

    # Log configuration summary (without sensitive values)
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Configuration loaded successfully:")
    logger.info(f"  vLLM URL: {settings.vllm_base_url}")
    logger.info(f"  Default model: {settings.default_model}")
    logger.info(f"  Async inference enabled: {settings.async_enabled}")
    logger.info(f"  Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    logger.info(f"  RAG enabled: {settings.rag_enabled}")
    logger.info(f"  Embedding model: {settings.embedding_model}")
    logger.info(f"  Embedding device: {settings.embedding_device}")
    logger.info(f"  Gateway URL (for UI): {settings.url}")
    logger.info(
        f"  UI timeouts: health={ui_settings.health_timeout}s, chat={ui_settings.chat_timeout}s"
    )
