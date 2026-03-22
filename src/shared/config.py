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

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# =========================================================================
# Knowledge Base Registry (loaded from JSON config file)
# =========================================================================

_DEFAULT_KB_PATH = Path(__file__).resolve().parent / "knowledge_bases.json"


class KnowledgeBaseConfig(BaseModel):
    """Single knowledge-base entry from shared/knowledge_bases.json."""

    knowledge_base: str
    aliases: list[str] = Field(default_factory=lambda: ["champion"])
    update_strategy: Literal["incremental", "replace"] = "replace"
    label: str = ""
    description: str = ""
    chunking_strategy: str = Field(
        description="Chunking strategy: fixed_token, code, or section_aware",
    )
    chunk_size: int = Field(
        description="Chunk size for document splitting",
        ge=100,
    )
    chunk_overlap: int = Field(
        description="Overlap between chunks",
        ge=0,
    )


def _load_knowledge_bases(path: Path | str) -> dict[str, KnowledgeBaseConfig]:
    """Load the knowledge-bases registry from a JSON file.

    Args:
        path: Path to the ``knowledge_bases.json`` file.

    Returns:
        Mapping of ``kb_name`` → ``KnowledgeBaseConfig``.
    """
    path = Path(path)

    if not path.exists():
        logger.warning("Knowledge-bases config not found at %s — using empty registry", path)
        return {}

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    registry: dict[str, KnowledgeBaseConfig] = {}
    for entry in raw:
        cfg = KnowledgeBaseConfig(**entry)
        registry[cfg.knowledge_base] = cfg
    return registry


# Module-level registry (populated lazily via get_knowledge_bases())
_KB_REGISTRY: dict[str, KnowledgeBaseConfig] | None = None


def get_knowledge_bases() -> dict[str, KnowledgeBaseConfig]:
    """Return the knowledge-base registry (cached after first call).

    Path is resolved from ``GATEWAY_KNOWLEDGE_BASES_PATH`` env var or
    the bundled default ``knowledge_bases.json``.
    """
    global _KB_REGISTRY  # noqa: PLW0603
    if _KB_REGISTRY is None:
        import os

        env_path = os.environ.get("GATEWAY_KNOWLEDGE_BASES_PATH", "").strip()
        path = Path(env_path) if env_path else _DEFAULT_KB_PATH
        _KB_REGISTRY = _load_knowledge_bases(path)
    return _KB_REGISTRY


# ---------------------------------------------------------------------------
# Backward-compatible KNOWLEDGE_BASES dict
# ---------------------------------------------------------------------------
# Legacy callers that import ``KNOWLEDGE_BASES`` from this module get a
# lazy-loading proxy that returns the same dict structure as before:
#   { "arxiv": { "collection": ..., "label": ..., "description": ... }, ... }
# The proxy loads the JSON config on first access.


class _KBProxy(dict):
    """Lazy dict that loads KB config on first access."""

    _loaded: bool = False

    def _ensure(self) -> None:
        if not self._loaded:
            for name, cfg in get_knowledge_bases().items():
                super().__setitem__(
                    name,
                    {
                        "label": cfg.label,
                        "description": cfg.description,
                        "aliases": cfg.aliases,
                        "update_strategy": cfg.update_strategy,
                        "chunking_strategy": cfg.chunking_strategy,
                        "chunk_size": cfg.chunk_size,
                        "chunk_overlap": cfg.chunk_overlap,
                    },
                )
            self._loaded = True

    def __getitem__(self, key):
        self._ensure()
        return super().__getitem__(key)

    def __contains__(self, key):
        self._ensure()
        return super().__contains__(key)

    def __iter__(self):
        self._ensure()
        return super().__iter__()

    def __len__(self):
        self._ensure()
        return super().__len__()

    def keys(self):
        self._ensure()
        return super().keys()

    def values(self):
        self._ensure()
        return super().values()

    def items(self):
        self._ensure()
        return super().items()

    def get(self, key, default=None):
        self._ensure()
        return super().get(key, default)


KNOWLEDGE_BASES: dict[str, dict] = _KBProxy()


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
        GATEWAY_DEFAULT_ALIAS: Default alias role for RAG retrieval
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
    max_completion_tokens: int = Field(
        default=512,
        description="Maximum number of tokens the model can generate per response",
        ge=1,
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
    celery_broker_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("CELERY_BROKER_URL", "GATEWAY_CELERY_BROKER_URL"),
        description="RabbitMQ broker URL for Celery (e.g. amqp://user:pass@rabbitmq:5672//)",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        validation_alias=AliasChoices("REDIS_URL", "GATEWAY_REDIS_URL"),
        description="Redis connection URL for token streaming pub/sub",
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
    knowledge_bases_path: str = Field(
        default="",
        description="Override path to knowledge_bases.json (leave empty to use bundled default)",
    )

    # =========================================================================
    # RAG Settings
    # =========================================================================
    rag_enabled: bool = Field(
        default=True,
        description="Enable RAG functionality",
    )
    embeddings_url: str = Field(
        default="http://localhost:8100",
        description="URL of the embeddings microservice",
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
        default=0.35,
        description="Minimum similarity score for retrieval",
        ge=0.0,
        le=1.0,
    )
    context_max_length: int = Field(
        default=4000,
        description="Maximum character length of RAG context",
        ge=100,
    )
    default_alias: str = Field(
        default="champion",
        description="Default alias role for RAG retrieval when none is specified",
    )

    # =========================================================================
    # OAuth2 / OIDC Settings
    # =========================================================================
    google_client_id: str = Field(
        default="",
        description="Google OAuth2 client ID",
    )
    google_client_secret: str = Field(
        default="",
        description="Google OAuth2 client secret",
    )
    google_redirect_uri: str = Field(
        default="",
        description="OAuth2 callback URL (e.g. https://agent.antonlab.ru:8443/auth/callback)",
    )
    google_discovery_url: str = Field(
        default="https://accounts.google.com/.well-known/openid-configuration",
        description="Google OIDC discovery URL",
    )
    agent042_db_url: str | None = Field(
        default=None,
        description="PostgreSQL connection URL for agent042 DB (async: postgresql+asyncpg://...)",
    )
    session_secret_key: str = Field(
        default="",
        description="Secret key for signing session cookies (32-byte hex)",
    )
    session_ttl_seconds: int = Field(
        default=86400,
        description="Session TTL in seconds (default 24 hours)",
        ge=60,
    )

    # =========================================================================
    # Internal Service API Key
    # =========================================================================
    internal_api_key: str = Field(
        default="",
        description="Pre-shared API key for internal service-to-service calls "
        "(e.g. Airflow eval runner)",
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


class EvalSettings(BaseSettings):
    """Settings for the evaluation runner.

    Environment Variables:
        EVAL_GATEWAY_URL: Gateway URL for generation evals.
        EVAL_JUDGE_MODEL: Gemini model name for LLM-as-Judge.
        EVAL_GOOGLE_AI_API_KEY: Google AI Studio API key (Gemini).
        EVAL_BERT_SCORE_MODEL: Model for BERTScore computation.
        EVAL_TEMPERATURE: Temperature for generation requests.
        EVAL_MAX_TOKENS: Max tokens for generation requests.
        EVAL_SAMPLE_LIMIT: Max samples per dataset (0 = unlimited).
    """

    model_config = SettingsConfigDict(
        env_prefix="EVAL_",
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    gateway_url: str = Field(
        default="http://localhost:9001",
        description="Gateway URL for generation evals",
    )
    judge_model: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model name for LLM-as-Judge",
    )
    google_ai_api_key: str = Field(
        default="",
        description="Google AI Studio API key for Gemini judge",
    )
    bert_score_model: str = Field(
        default="microsoft/deberta-base-mnli",
        description="Model for BERTScore computation",
    )
    temperature: float = Field(
        default=0.0,
        description="Temperature for generation requests",
        ge=0.0,
    )
    max_tokens: int = Field(
        default=512,
        description="Max tokens for generation requests",
        ge=1,
    )
    sample_limit: int = Field(
        default=100,
        description="Max samples per dataset (0 = unlimited)",
        ge=0,
    )
    code_exec_timeout: int = Field(
        default=30,
        description="Timeout in seconds for sandboxed code execution",
        ge=1,
    )
    code_exec_image: str = Field(
        default="python:3.11-slim",
        description="Docker image for sandboxed code execution",
    )
    code_exec_mem_limit: str = Field(
        default="512m",
        description="Memory limit for sandboxed code execution containers",
    )
    code_exec_cpus: float = Field(
        default=1.0,
        description="CPU limit for sandboxed code execution containers",
        ge=0.1,
    )
    internal_api_key: str = Field(
        default="",
        description="Internal API key for authenticating with the gateway",
    )
    db_url: str | None = Field(
        default=None,
        description="PostgreSQL connection URL for eval results (sync: postgresql://...)",
    )


@lru_cache
def get_eval_settings() -> EvalSettings:
    """Get cached evaluation settings."""
    return EvalSettings()


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

    # Load and validate the knowledge-base registry
    kb_registry = get_knowledge_bases()

    # Log configuration summary (without sensitive values)
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Configuration loaded successfully:")
    logger.info(f"  vLLM URL: {settings.vllm_base_url}")
    logger.info(f"  Default model: {settings.default_model}")
    logger.info(f"  Async inference enabled: {settings.async_enabled}")
    logger.info(f"  Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
    logger.info(f"  RAG enabled: {settings.rag_enabled}")
    logger.info(f"  Embeddings URL: {settings.embeddings_url}")
    logger.info(f"  Embedding model: {settings.embedding_model}")
    logger.info(f"  Embedding device: {settings.embedding_device}")
    logger.info(f"  Knowledge bases: {list(kb_registry.keys()) or '(none)'}")
    logger.info(f"  Gateway URL (for UI): {settings.url}")
    logger.info(
        f"  UI timeouts: health={ui_settings.health_timeout}s, chat={ui_settings.chat_timeout}s"
    )
