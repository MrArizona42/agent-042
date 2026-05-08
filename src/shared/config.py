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
from typing import Literal, Optional

from pydantic import AliasChoices, BaseModel, Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.local_env import load_local_env

logger = logging.getLogger(__name__)

# =========================================================================
# Knowledge Base Registry (loaded from JSON config file)
# =========================================================================

_DEFAULT_KB_PATH = Path(__file__).resolve().parent / "knowledge_bases.json"


class AliasConfig(BaseModel):
    """Per-alias query-time RAG configuration.

    No defaults — every field must be explicit in ``knowledge_bases.json``.
    Adding a new field forces updating every alias entry; Pydantic will
    reject incomplete entries at load time.
    """

    top_k: int
    score_threshold: float
    reranker: Optional[str]  # null today; model name when reranker is implemented
    retrieval_strategy: Literal["dense", "hybrid", "sparse"]
    reranker_multiplier: int


class AdapterConfig(BaseModel):
    """Per-task LoRA routing configuration."""

    name: str = ""
    alias: str = ""
    enabled: bool = False

    @model_validator(mode="after")
    def _enabled_adapter_must_be_complete(self) -> "AdapterConfig":
        if self.enabled and (not self.name.strip() or not self.alias.strip()):
            raise ValueError("enabled adapter requires non-empty name and alias")
        return self


class KBConfig(BaseModel):
    """Single knowledge-base entry within a task group."""

    name: str
    default_alias: str
    aliases: dict[str, AliasConfig]
    update_strategy: Literal["incremental", "replace"] = "replace"
    label: str = ""
    description: str = ""
    selection_description: str

    @model_validator(mode="after")
    def _default_alias_must_exist(self) -> "KBConfig":
        if self.default_alias not in self.aliases:
            raise ValueError(
                f"default_alias '{self.default_alias}' is not a declared alias "
                f"(available: {list(self.aliases.keys())})"
            )
        return self


class TaskConfig(BaseModel):
    """Top-level task entry from shared/knowledge_bases.json."""

    task: str
    label: str = ""
    routing_description: str
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    knowledge_bases: list[KBConfig] = Field(default_factory=list)


def _load_knowledge_bases(
    path: Path | str,
) -> tuple[dict[str, TaskConfig], dict[str, KBConfig]]:
    """Load the knowledge-bases registry from a JSON file.

    Args:
        path: Path to the ``knowledge_bases.json`` file.

    Returns:
        Tuple of (task registry, flat KB index keyed by KB name).

    Raises:
        ValueError: If duplicate KB names are found across tasks.
    """
    path = Path(path)

    if not path.exists():
        logger.warning("Knowledge-bases config not found at %s — using empty registry", path)
        return {}, {}

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    registry: dict[str, TaskConfig] = {}
    index: dict[str, KBConfig] = {}
    for entry in raw:
        cfg = TaskConfig(**entry)
        registry[cfg.task] = cfg
        for kb_cfg in cfg.knowledge_bases:
            if kb_cfg.name in index:
                raise ValueError(
                    f"Duplicate KB name '{kb_cfg.name}' found across tasks. "
                    f"KB names must be unique."
                )
            index[kb_cfg.name] = kb_cfg
    return registry, index


# Module-level caches (populated lazily via get_knowledge_bases())
_KB_REGISTRY: dict[str, TaskConfig] | None = None
_KB_INDEX: dict[str, KBConfig] | None = None


def get_knowledge_bases() -> dict[str, TaskConfig]:
    """Return the knowledge-base registry (cached after first call).

    Path is resolved from ``GATEWAY_KNOWLEDGE_BASES_PATH`` env var or
    the bundled default ``knowledge_bases.json``.

    Returns:
        Mapping of ``task_name`` → ``TaskConfig``.
    """
    global _KB_REGISTRY, _KB_INDEX  # noqa: PLW0603
    if _KB_REGISTRY is None:
        import os

        env_path = os.environ.get("GATEWAY_KNOWLEDGE_BASES_PATH", "").strip()
        path = Path(env_path) if env_path else _DEFAULT_KB_PATH
        _KB_REGISTRY, _KB_INDEX = _load_knowledge_bases(path)
    return _KB_REGISTRY


def get_kb_config(kb_name: str) -> KBConfig | None:
    """Look up a KB by name (O(1) dict lookup).

    Returns the ``KBConfig`` for *kb_name* or ``None`` if not found.
    """
    # Ensure registry is loaded
    get_knowledge_bases()
    if _KB_INDEX is None:
        return None
    return _KB_INDEX.get(kb_name)


def get_kb_names() -> list[str]:
    """Flat list of all KB names across all tasks."""
    get_knowledge_bases()
    if _KB_INDEX is None:
        return []
    return list(_KB_INDEX.keys())


def validate_kb_alias(kb: str, alias: str | None = None) -> None:
    """Raise ValueError with a consistent message if kb or alias is unknown.

    When *alias* is ``None`` only the KB name is validated.
    """
    kb_cfg = get_kb_config(kb)
    if kb_cfg is None:
        raise ValueError(f"KB '{kb}' not found. Available: {get_kb_names()}")
    if alias is not None and alias not in kb_cfg.aliases:
        raise ValueError(
            f"Alias '{alias}' not valid for KB '{kb}'. Available: {list(kb_cfg.aliases.keys())}"
        )


# ---------------------------------------------------------------------------
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
    vllm_timeout: float = Field(
        default=60.0,
        description="Timeout for vLLM requests in seconds",
        ge=1.0,
    )
    repetition_penalty: float = Field(
        default=1.1,
        description="Repetition penalty applied to all generation requests to prevent token loops",
        ge=1.0,
    )
    streaming_timeout: float = Field(
        default=300.0,
        description="Idle timeout for Redis Pub/Sub streaming in seconds",
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


class BudgetSettings(BaseModel):
    """Prompt and response budgeting settings for online inference."""

    model_max_tokens: int = Field(
        default=32768,
        description="Configured max model window used for prompt/response budgeting",
        ge=1,
    )
    chars_per_token: float = Field(
        default=4.0,
        description="Approximate character-to-token ratio used for gateway shaping",
        gt=0.0,
    )
    budget_guard: int = Field(
        default=512,
        description="Reserved safeguard gap for estimation and chat-template overhead",
        ge=0,
    )
    budget_system: int = Field(
        default=768,
        description="Approximate token budget reserved for the system prompt",
        ge=1,
    )
    budget_turn: int = Field(
        default=10240,
        description="Approximate token budget reserved for the current user turn",
        ge=1,
    )
    min_budget_history: int = Field(
        default=4096,
        description="Minimum approximate token budget reserved for chat history",
        ge=0,
    )
    budget_rag: int = Field(
        default=6144,
        description="Approximate token budget reserved for all retrieved RAG context",
        ge=0,
    )
    min_response_budget: int = Field(
        default=256,
        description="Minimum exact response token budget required before generation",
        ge=1,
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
    kb_selection_threshold: float = Field(
        default=0.3,
        description="Cosine similarity threshold for automatic KB selection",
        ge=0.0,
        le=1.0,
    )
    task_classification_threshold: float = Field(
        default=0.0,
        description=(
            "Minimum cosine similarity for task classification; "
            "0.0 means always pick the closest task"
        ),
        ge=0.0,
        le=1.0,
    )
    rag_strict_startup: bool = Field(
        default=False,
        description="If True, raise on legacy / invalid Qdrant collections at startup "
        "instead of logging and marking them unavailable",
    )
    sparse_encoder_model: str = Field(
        default="Qdrant/bm25",
        validation_alias=AliasChoices(
            "SPARSE_ENCODER_MODEL",
            "sparse_encoder_model",
        ),
        description="fastembed model name for sparse (BM25) vector encoding",
    )
    reranker_url: str = Field(
        default="http://reranker:8101",
        validation_alias=AliasChoices(
            "RERANKER_URL",
            "reranker_url",
        ),
        description="URL of the reranker microservice",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        validation_alias=AliasChoices(
            "RERANKER_MODEL",
            "reranker_model",
        ),
        description="Cross-encoder model loaded by the reranker service",
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


class GatewaySettings(
    PlatformSettings,
    GatewayBehaviorSettings,
    BudgetSettings,
    RagSettings,
    AuthSettings,
):
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
        default="microsoft/deberta-v3-base",
        description="Model for BERTScore computation",
    )
    temperature: float = Field(
        default=0.0,
        description="Temperature for generation requests",
        ge=0.0,
    )
    max_completion_tokens: int = Field(
        default=2048,
        description=(
            "Upper bound for one eval prediction. Prevents eval requests from "
            "claiming the full model window for generation."
        ),
        ge=1,
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


class WorkerSettings(BaseSettings):
    """Worker-specific runtime settings.

    Shared infrastructure endpoints such as ``CELERY_BROKER_URL`` still use the
    canonical platform env names, while worker runtime knobs remain worker-local.
    """

    model_config = SettingsConfigDict(extra="ignore")

    celery_broker_url: str = Field(
        validation_alias=PLATFORM_CELERY_BROKER_URL_ENV,
        description="RabbitMQ connection URL (e.g. amqp://user:password@rabbitmq:5672//)",
    )
    task_default_timeout: int = Field(
        default=300,
        description="Default task timeout in seconds",
    )
    task_max_retries: int = Field(
        default=3,
        description="Maximum number of task retries",
    )
    task_retry_delay: int = Field(
        default=5,
        description="Delay between retries in seconds",
    )
    worker_pool: str = Field(
        default="prefork",
        description="Celery execution pool for gateway inference tasks",
    )
    worker_concurrency: int = Field(
        default=2,
        description="Concurrent worker slots for gateway inference tasks",
        ge=1,
    )
    worker_send_task_events: bool = Field(
        default=True,
        description="Emit Celery task events so Flower can observe queued/running tasks",
    )
    worker_cancel_long_running_tasks_on_connection_loss: bool = Field(
        default=True,
        description="Cancel in-flight tasks if the broker connection is lost",
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


@lru_cache
def get_worker_settings() -> WorkerSettings:
    """Get cached worker settings."""
    return WorkerSettings()


def clear_knowledge_base_caches() -> None:
    """Reset KB registry and index so the next access re-reads from disk."""
    global _KB_REGISTRY, _KB_INDEX  # noqa: PLW0603
    _KB_REGISTRY = None
    _KB_INDEX = None


def clear_settings_caches() -> None:
    """Clear cached settings and derived config state."""
    get_platform_settings.cache_clear()
    get_settings.cache_clear()
    get_registry_settings.cache_clear()
    get_eval_settings.cache_clear()
    get_ui_settings.cache_clear()
    get_worker_settings.cache_clear()
    clear_knowledge_base_caches()


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
