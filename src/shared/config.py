"""Unified configuration for all services.

This module provides a single source of truth for all configuration settings
across the gateway, RAG, and UI services. Configuration is loaded from
environment variables with sensible defaults for local development.

Local-only entrypoints that want repo-root ``.env`` support should call
``bootstrap_local_settings_env()`` before the first settings access.

Usage:
    from shared.config import bootstrap_local_settings_env, get_settings

    bootstrap_local_settings_env()
    settings = get_settings()
    print(settings.qdrant_host)
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.local_env import load_local_env

logger = logging.getLogger(__name__)

# =========================================================================
# Knowledge Base Registry (loaded from JSON config file)
# =========================================================================

_DEFAULT_KB_PATH = Path(__file__).resolve().parent / "knowledge_bases.json"


class KBConfig(BaseModel):
    """Single knowledge-base entry within a task group."""

    name: str
    aliases: list[str] = Field(default_factory=lambda: ["champion"])
    update_strategy: Literal["incremental", "replace"] = "replace"
    label: str = ""
    description: str = ""


class TaskConfig(BaseModel):
    """Top-level task entry from shared/knowledge_bases.json."""

    task: str
    label: str = ""
    knowledge_bases: list[KBConfig] = Field(default_factory=list)


def _load_knowledge_bases(path: Path | str) -> dict[str, TaskConfig]:
    """Load the knowledge-bases registry from a JSON file.

    Args:
        path: Path to the ``knowledge_bases.json`` file.

    Returns:
        Mapping of ``task_name`` → ``TaskConfig``.
    """
    path = Path(path)

    if not path.exists():
        logger.warning("Knowledge-bases config not found at %s — using empty registry", path)
        return {}

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    registry: dict[str, TaskConfig] = {}
    for entry in raw:
        cfg = TaskConfig(**entry)
        registry[cfg.task] = cfg
    return registry


# Module-level registry (populated lazily via get_knowledge_bases())
_KB_REGISTRY: dict[str, TaskConfig] | None = None


def get_knowledge_bases() -> dict[str, TaskConfig]:
    """Return the knowledge-base registry (cached after first call).

    Path is resolved from ``GATEWAY_KNOWLEDGE_BASES_PATH`` env var or
    the bundled default ``knowledge_bases.json``.

    Returns:
        Mapping of ``task_name`` → ``TaskConfig``.
    """
    global _KB_REGISTRY  # noqa: PLW0603
    if _KB_REGISTRY is None:
        import os

        env_path = os.environ.get("GATEWAY_KNOWLEDGE_BASES_PATH", "").strip()
        path = Path(env_path) if env_path else _DEFAULT_KB_PATH
        _KB_REGISTRY = _load_knowledge_bases(path)
    return _KB_REGISTRY


def get_kb_config(kb_name: str) -> KBConfig | None:
    """Look up a KB by name across all tasks.

    Returns the ``KBConfig`` for *kb_name* or ``None`` if not found.
    """
    for task_cfg in get_knowledge_bases().values():
        for kb_cfg in task_cfg.knowledge_bases:
            if kb_cfg.name == kb_name:
                return kb_cfg
    return None


# ---------------------------------------------------------------------------
# Backward-compatible KNOWLEDGE_BASES dict
# ---------------------------------------------------------------------------
# Legacy callers that import ``KNOWLEDGE_BASES`` from this module get a
# lazy-loading proxy that returns the same dict structure as before:
#   { "arxiv": { "collection": ..., "label": ..., "description": ... }, ... }
# The proxy loads the JSON config on first access.


class _KBProxy(dict):
    """Lazy dict that loads KB config on first access.

    Provides backward-compatible flat ``{kb_name: info_dict}`` access.
    """

    _loaded: bool = False

    def _ensure(self) -> None:
        if not self._loaded:
            for task_cfg in get_knowledge_bases().values():
                for kb_cfg in task_cfg.knowledge_bases:
                    super().__setitem__(
                        kb_cfg.name,
                        {
                            "label": kb_cfg.label,
                            "description": kb_cfg.description,
                            "aliases": kb_cfg.aliases,
                            "update_strategy": kb_cfg.update_strategy,
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

    def reset(self) -> None:
        super().clear()
        self._loaded = False


KNOWLEDGE_BASES = _KBProxy()

PLATFORM_VLLM_BASE_URL_ENV = "VLLM_BASE_URL"
PLATFORM_EMBEDDINGS_URL_ENV = "EMBEDDINGS_URL"
PLATFORM_QDRANT_HOST_ENV = "QDRANT_HOST"
PLATFORM_QDRANT_PORT_ENV = "QDRANT_PORT"
PLATFORM_MLFLOW_TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
PLATFORM_REDIS_URL_ENV = "REDIS_URL"
PLATFORM_CELERY_BROKER_URL_ENV = "CELERY_BROKER_URL"


class PlatformSettings(BaseSettings):
    """Canonical shared endpoint settings used across services.

    Canonical environment variable names:
        VLLM_BASE_URL
        EMBEDDINGS_URL
        QDRANT_HOST
        QDRANT_PORT
        MLFLOW_TRACKING_URI
        REDIS_URL
        CELERY_BROKER_URL

    """

    model_config = SettingsConfigDict(extra="ignore")

    vllm_base_url: str = Field(
        default="http://localhost:8000",
        validation_alias=PLATFORM_VLLM_BASE_URL_ENV,
        description="URL where the shared vLLM server is reachable",
    )
    embeddings_url: str = Field(
        default="http://localhost:8100",
        validation_alias=PLATFORM_EMBEDDINGS_URL_ENV,
        description="URL of the shared embeddings microservice",
    )
    qdrant_host: str = Field(
        default="localhost",
        validation_alias=PLATFORM_QDRANT_HOST_ENV,
        description="Shared Qdrant server hostname",
    )
    qdrant_port: int = Field(
        default=6333,
        validation_alias=PLATFORM_QDRANT_PORT_ENV,
        description="Shared Qdrant server port",
        ge=1,
        le=65535,
    )
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5050",
        validation_alias=PLATFORM_MLFLOW_TRACKING_URI_ENV,
        description="Shared MLflow tracking server URL",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        validation_alias=PLATFORM_REDIS_URL_ENV,
        description="Redis connection URL for shared streaming and coordination",
    )
    celery_broker_url: str | None = Field(
        default=None,
        validation_alias=PLATFORM_CELERY_BROKER_URL_ENV,
        description="RabbitMQ broker URL for shared Celery-based workflows",
    )


class GatewayBehaviorSettings(BaseModel):
    """Gateway request handling and service behavior settings."""

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
    streaming_timeout: float = Field(
        default=300.0,
        description="Timeout for Redis Pub/Sub streaming in seconds",
        ge=1.0,
    )
    embeddings_timeout: float = Field(
        default=120.0,
        description="Timeout for embeddings service HTTP requests in seconds",
        ge=1.0,
    )
    async_enabled: bool = Field(
        default=True,
        description="Enable async inference via Celery workers",
    )
    cors_allow_origins_csv: str = Field(
        default="*",
        validation_alias=AliasChoices("GATEWAY_CORS_ALLOW_ORIGINS"),
        description="Allowed CORS origins (comma-separated in env)",
    )

    @computed_field
    @property
    def cors_allow_origins(self) -> list[str]:
        """CORS allowed origins, parsed from comma-separated string."""
        return [o.strip() for o in self.cors_allow_origins_csv.split(",") if o.strip()]

    service_name: str = Field(
        default="agent-042-gateway",
        description="Service name displayed in API docs",
    )
    url: str = Field(
        default="http://localhost:9001",
        alias="GATEWAY_URL",
        description="Full URL to the gateway (used by UI)",
    )


class RagSettings(BaseModel):
    """Gateway RAG behavior and embedding model settings."""

    knowledge_bases_path: str = Field(
        default="",
        description="Override path to knowledge_bases.json (leave empty to use bundled default)",
    )
    rag_enabled: bool = Field(
        default=True,
        description="Enable RAG functionality",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias=AliasChoices(
            "GATEWAY_EMBEDDING_MODEL",
            "EMBEDDINGS_MODEL",
        ),
        description="HuggingFace model for embeddings",
    )
    embedding_device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        validation_alias=AliasChoices(
            "GATEWAY_EMBEDDING_DEVICE",
            "EMBEDDINGS_DEVICE",
        ),
        description="Device for embedding model",
    )
    embedding_batch_size: int = Field(
        default=32,
        validation_alias=AliasChoices(
            "GATEWAY_EMBEDDING_BATCH_SIZE",
            "EMBEDDINGS_BATCH_SIZE",
        ),
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


class AuthSettings(BaseModel):
    """Gateway auth, session, and internal caller authentication settings."""

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
    internal_api_key: str = Field(
        default="",
        description="Pre-shared API key for internal service-to-service calls "
        "(e.g. Airflow eval runner)",
    )


class GatewaySettings(PlatformSettings, GatewayBehaviorSettings, RagSettings, AuthSettings):
    """Gateway-facing settings composed from smaller concern groups.

    Shared platform endpoints prefer canonical names such as VLLM_BASE_URL and
    QDRANT_HOST, while gateway-specific behavior uses the GATEWAY_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        extra="ignore",
    )


# Backward compatibility alias used across the existing codebase.
Settings = GatewaySettings


class RegistrySettings(BaseSettings):
    """Settings for MLflow Model Registry / adapter sync.

    Environment Variables:
        MLFLOW_TRACKING_URI: Preferred MLflow tracking server URL.
        VLLM_BASE_URL: Preferred shared vLLM server URL.
        REGISTRY_ADAPTERS_DIR: Local directory for downloaded LoRA adapters.
        REGISTRY_AUTO_SYNC: Pull production adapters on service startup.
    """

    model_config = SettingsConfigDict(
        env_prefix="REGISTRY_",
        extra="ignore",
    )

    mlflow_tracking_uri: str = Field(
        default="http://localhost:5050",
        validation_alias=PLATFORM_MLFLOW_TRACKING_URI_ENV,
        description="MLflow tracking server URL",
    )
    adapters_dir: str = Field(
        default="./adapters",
        description="Local directory for downloaded LoRA adapters",
    )
    production_alias: str | None = Field(
        default=None,
        description="MLflow alias that marks an adapter as production-ready. "
        "Used as the default alias for promote/demote commands. "
        "None means no production adapters are synced (base model only).",
    )
    sync_aliases_csv: str = Field(
        default="champion,challenger",
        validation_alias=AliasChoices("REGISTRY_SYNC_ALIASES"),
        description="Comma-separated MLflow aliases to sync to vLLM. "
        "Each (model, alias) pair becomes a vLLM adapter named '{model}-{alias}'.",
    )

    @computed_field
    @property
    def sync_aliases(self) -> list[str]:
        """MLflow aliases to sync, parsed from comma-separated string."""
        return [a.strip() for a in self.sync_aliases_csv.split(",") if a.strip()]

    vllm_base_url: str = Field(
        default="http://localhost:8000",
        validation_alias=PLATFORM_VLLM_BASE_URL_ENV,
        description="vLLM OpenAI-compatible server URL for hot-loading adapters.",
    )
    auto_sync: bool = Field(
        default=False,
        description="Automatically sync production adapters on startup",
    )


# Backward compatibility alias used across the existing codebase.
ModelRegistrySettings = RegistrySettings


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
    code_exec_mem_limit: str = Field(
        default="512m",
        description=(
            "Memory limit string for sandboxed code execution. "
            "Accepted for config compatibility; not currently enforced "
            "by bwrap (reserved for future cgroup-based limits)."
        ),
    )
    code_exec_cpus: float = Field(
        default=1.0,
        description="CPU share for sandboxed code execution (used to derive rlimit-cpu)",
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


@lru_cache
def get_platform_settings() -> PlatformSettings:
    """Get cached canonical shared endpoint settings."""
    return PlatformSettings()


class UISettings(BaseSettings):
    """UI-specific settings with UI_ prefix.

    These are separate because they use a different environment variable prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="UI_",
        extra="ignore",
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
def get_settings() -> GatewaySettings:
    """Get cached application settings.

    Settings are loaded once and cached for the lifetime of the process.
    This ensures consistent configuration and avoids repeated parsing.

    Returns:
        GatewaySettings: Validated application settings

    Raises:
        ValidationError: If environment variables contain invalid values
    """
    return GatewaySettings()


@lru_cache
def get_registry_settings() -> RegistrySettings:
    """Get cached model registry settings."""
    return RegistrySettings()


@lru_cache
def get_ui_settings() -> UISettings:
    """Get cached UI-specific settings.

    Returns:
        UISettings: Validated UI settings
    """
    return UISettings()


def clear_settings_caches() -> None:
    """Clear cached settings and derived config state."""
    global _KB_REGISTRY  # noqa: PLW0603

    get_platform_settings.cache_clear()
    get_settings.cache_clear()
    get_registry_settings.cache_clear()
    get_eval_settings.cache_clear()
    get_ui_settings.cache_clear()
    _KB_REGISTRY = None
    KNOWLEDGE_BASES.reset()


def bootstrap_local_settings_env(
    *,
    repo_root: Path | None = None,
    env_file: str | Path | None = None,
    legacy_fallbacks: tuple[str | Path, ...] = (),
) -> Path | None:
    """Explicitly load the repo-root ``.env`` for local entrypoints.

    Containerized deployments should inject env vars directly and may call this
    helper safely; it becomes a no-op when no local env file exists.
    """
    resolved_root = (
        repo_root.resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    )
    loaded_env = load_local_env(
        env_file,
        repo_root=resolved_root,
        legacy_fallbacks=legacy_fallbacks,
    )
    clear_settings_caches()
    return loaded_env


def validate_settings_on_startup() -> None:
    """Validate all settings at application startup.

    Call this function early in your application's lifecycle to fail fast
    if configuration is invalid.

    Raises:
        ValidationError: If any settings are invalid
    """
    get_platform_settings()

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
    kb_names = [kb.name for tc in kb_registry.values() for kb in tc.knowledge_bases]
    logger.info(f"  Knowledge bases: {kb_names or '(none)'}")
    logger.info(f"  Gateway URL (for UI): {settings.url}")
    logger.info(
        f"  UI timeouts: health={ui_settings.health_timeout}s, chat={ui_settings.chat_timeout}s"
    )
