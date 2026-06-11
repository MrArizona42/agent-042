"""Unified configuration for all services.

This module provides a single source of truth for all configuration settings
across the gateway, RAG, and UI services. Configuration is loaded from
environment variables with sensible defaults for local development.

Usage:
    from shared.config import get_settings

    settings = get_settings()
    print(settings.platform.qdrant_host)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, PydanticBaseSettingsSource, SettingsConfigDict

from shared import catalog

logger = logging.getLogger(__name__)


class PlatformSettings(BaseModel):
    """Canonical shared endpoint settings used across services.

    Canonical nested environment variable names read by the root settings loader:
        PLATFORM__VLLM_BASE_URL
        PLATFORM__EMBEDDINGS_URL
        PLATFORM__QDRANT_HOST
        PLATFORM__QDRANT_PORT
        PLATFORM__MLFLOW_TRACKING_URI
        PLATFORM__REDIS_URL
        PLATFORM__CELERY_BROKER_URL
        PLATFORM__KAFKA_BOOTSTRAP_SERVERS
        PLATFORM__INFERENCE_EVENTS_TOPIC

    """

    model_config = ConfigDict(populate_by_name=True, frozen=True)
    vllm_base_url: str = Field(
        default="http://localhost:8000",
        description="URL where the shared vLLM server is reachable",
    )
    embeddings_url: str = Field(
        default="http://localhost:8100",
        description="URL of the shared embeddings microservice",
    )
    qdrant_host: str = Field(
        default="localhost",
        description="Shared Qdrant server hostname",
    )
    qdrant_port: int = Field(
        default=6333,
        description="Shared Qdrant server port",
        ge=1,
        le=65535,
    )
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5050",
        description="Shared MLflow tracking server URL",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for shared streaming and coordination",
    )
    celery_broker_url: str | None = Field(
        default=None,
        description="RabbitMQ broker URL for shared Celery-based workflows",
    )
    kafka_bootstrap_servers: str | None = Field(
        default=None,
        description="Kafka-compatible bootstrap servers for durable inference events",
    )
    inference_events_topic: str = Field(
        default="inference.events.v1",
        description="Kafka-compatible topic for durable inference lifecycle events",
    )


class BudgetSettings(BaseModel):
    """Prompt and response budgeting settings for online inference."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

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


class GatewayConfig(BaseModel):
    """Gateway request handling and service behavior settings."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    default_model: str = Field(
        default="/models/Qwen/Qwen3-0.6B",
        description="Default model when none specified in request",
    )
    api_key: SecretStr | None = Field(
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
    cors_allow_origins: Annotated[tuple[str, ...], NoDecode] = Field(
        default=("*",),
        description="Allowed CORS origins.",
    )
    service_name: str = Field(
        default="agent-042-gateway",
        description="Service name displayed in API docs",
    )
    url: str = Field(
        default="http://localhost:9001",
        description="Full URL to the gateway (used by UI)",
    )
    budget: BudgetSettings = Field(default_factory=BudgetSettings)

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _normalize_cors_allow_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(origin.strip() for origin in value.split(",") if origin.strip())
        if isinstance(value, (list, tuple, set)):
            return tuple(str(origin).strip() for origin in value if str(origin).strip())
        return value


class RagBuildSettings(BaseModel):
    """RAG build-time batching settings."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation during RAG builds",
        ge=1,
    )
    qdrant_upsert_batch_size: int = Field(
        default=128,
        description="Batch size for Qdrant upserts during RAG materialization",
        ge=1,
    )


class RagSettings(BaseModel):
    """Gateway RAG behavior and embedding model settings."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

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
    build: RagBuildSettings = Field(default_factory=RagBuildSettings)
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
        description="fastembed model name for sparse (BM25) vector encoding",
    )
    reranker_url: str = Field(
        default="http://reranker:8101",
        description="URL of the reranker microservice",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model loaded by the reranker service",
    )


class AuthSettings(BaseModel):
    """Gateway auth, session, and internal caller authentication settings."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    google_client_id: str = Field(
        default="",
        description="Google OAuth2 client ID",
    )
    google_client_secret: SecretStr = Field(
        default_factory=lambda: SecretStr(""),
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
    session_secret_key: SecretStr = Field(
        default_factory=lambda: SecretStr(""),
        description="Secret key for signing session cookies (32-byte hex)",
    )
    session_ttl_seconds: int = Field(
        default=86400,
        description="Session TTL in seconds (default 24 hours)",
        ge=60,
    )
    internal_api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(""),
        description="Pre-shared API key for internal service-to-service calls "
        "(e.g. Airflow eval runner)",
    )


class CatalogConfig(BaseModel):
    """Settings for the shared task/knowledge-base/source catalog."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    path: Path | None = Field(
        default=None,
        description=(
            "Override path to the catalog TOML file; relative paths resolve from "
            "the repository root, empty uses the bundled default"
        ),
    )

    @field_validator("path", mode="before")
    @classmethod
    def _normalize_path(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            return Path(stripped)
        return value


class AdapterRegistryConfig(BaseModel):
    """Settings for MLflow model registry / adapter sync."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    adapters_dir: Path = Field(
        default=Path("./adapters"),
        description="Local directory for downloaded LoRA adapters",
    )
    production_alias: str | None = Field(
        default=None,
        description="MLflow alias that marks an adapter as production-ready.",
    )
    sync_aliases: Annotated[tuple[str, ...], NoDecode] = Field(
        default=("champion", "challenger"),
        description="MLflow aliases to sync to vLLM.",
    )
    auto_sync: bool = Field(
        default=False,
        description="Automatically sync production adapters on startup",
    )

    @field_validator("sync_aliases", mode="before")
    @classmethod
    def _normalize_sync_aliases(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(alias.strip() for alias in value.split(",") if alias.strip())
        if isinstance(value, (list, tuple, set)):
            return tuple(str(alias).strip() for alias in value if str(alias).strip())
        return value


class JudgeSettings(BaseModel):
    """Resolved LLM-as-judge transport and model configuration."""

    model_config = ConfigDict(frozen=True)

    backend: Literal["local_vllm", "openai_compatible"]
    model: str
    base_url: str
    api_key: str | None = None
    timeout: float = Field(
        default=60.0,
        ge=1.0,
        description="Timeout for one judge request in seconds",
    )
    request_delay_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Optional delay inserted between consecutive judge requests",
    )


class EvalJudgeSettings(BaseModel):
    """Raw judge configuration from the eval section."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    backend: Literal["local_vllm", "openai_compatible"] = Field(default="local_vllm")
    model: str = Field(
        default="/models/Qwen/Qwen3-0.6B",
        description="Judge model name used for LLM-as-judge scoring.",
    )
    base_url: str = Field(
        default="",
        description="Base URL for external OpenAI-compatible judge backends.",
    )
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(""),
        description="Optional API key for external OpenAI-compatible judge backends.",
    )
    timeout: float = Field(default=60.0, ge=1.0)
    request_delay_seconds: float = Field(default=0.0, ge=0.0)


class EvalMetricSettings(BaseModel):
    """Eval metric and generation controls."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    bert_score_model: str = Field(
        default="microsoft/deberta-v3-base",
        description="Model for BERTScore computation",
    )
    temperature: float = Field(default=0.0, ge=0.0)
    max_completion_tokens: int = Field(default=2048, ge=1)


class EvalSandboxSettings(BaseModel):
    """Sandbox execution limits for code evals."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    code_exec_timeout: int = Field(default=30, ge=1)
    code_exec_mem_limit: str = Field(
        default="512m",
        description=(
            "Memory limit string for sandboxed code execution. Accepted for config "
            "compatibility; not currently enforced by bwrap."
        ),
    )
    code_exec_cpus: float = Field(default=1.0, ge=0.1)


class EvalConfig(BaseModel):
    """Settings for the evaluation runner."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    judge: EvalJudgeSettings = Field(default_factory=EvalJudgeSettings)
    metrics: EvalMetricSettings = Field(default_factory=EvalMetricSettings)
    sandbox: EvalSandboxSettings = Field(default_factory=EvalSandboxSettings)
    db_url: str | None = Field(
        default=None,
        description="PostgreSQL connection URL for eval results (sync: postgresql://...)",
    )

    @model_validator(mode="after")
    def _validate_judge_backend_config(self) -> "EvalConfig":
        if self.judge.backend == "openai_compatible" and not self.judge.base_url.strip():
            raise ValueError(
                "eval.judge.base_url must be set when eval.judge.backend=openai_compatible"
            )
        return self

    def resolve_judge_settings(self, platform: PlatformSettings) -> JudgeSettings:
        """Resolve backend-specific judge settings to a concrete transport config."""
        judge = self.judge
        api_key = secret_value(judge.api_key)
        if judge.backend == "local_vllm":
            return JudgeSettings(
                backend="local_vllm",
                model=judge.model.strip(),
                base_url=platform.vllm_base_url,
                api_key=api_key.strip() or None if api_key is not None else None,
                timeout=judge.timeout,
                request_delay_seconds=judge.request_delay_seconds,
            )

        return JudgeSettings(
            backend="openai_compatible",
            model=judge.model.strip(),
            base_url=judge.base_url.strip(),
            api_key=api_key.strip() or None if api_key is not None else None,
            timeout=judge.timeout,
            request_delay_seconds=judge.request_delay_seconds,
        )


class UIConfig(BaseModel):
    """UI-specific request timeout settings."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

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


class WorkerConfig(BaseModel):
    """Worker-specific runtime settings."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    default_timeout: int = Field(
        default=300,
        description="Default task timeout in seconds",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of task retries",
    )
    retry_delay: int = Field(
        default=5,
        description="Delay between retries in seconds",
    )
    pool: str = Field(
        default="prefork",
        description="Celery execution pool for gateway inference tasks",
    )
    concurrency: int = Field(
        default=2,
        description="Concurrent worker slots for gateway inference tasks",
        ge=1,
    )
    send_task_events: bool = Field(
        default=True,
        description="Emit Celery task events so Flower can observe queued/running tasks",
    )
    cancel_long_running_tasks_on_connection_loss: bool = Field(
        default=True,
        description="Cancel in-flight tasks if the broker connection is lost",
    )


class Settings(BaseSettings):
    """Unified runtime configuration root."""

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        populate_by_name=True,
        frozen=True,
    )

    platform: PlatformSettings = Field(default_factory=PlatformSettings)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    rag: RagSettings = Field(default_factory=RagSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    catalog: CatalogConfig = Field(default_factory=CatalogConfig)
    adapter_registry: AdapterRegistryConfig = Field(default_factory=AdapterRegistryConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
            continue
        merged[key] = value
    return merged


def secret_value(value: SecretStr | str | None) -> str | None:
    """Return the raw value for a secret-like field."""

    if value is None:
        return None
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return value


def load_settings(overrides: dict[str, Any] | None = None) -> Settings:
    """Build the runtime settings tree from nested env names and explicit overrides."""

    settings = Settings()
    if not overrides:
        return settings

    payload = settings.model_dump(exclude_none=False, exclude_computed_fields=True)
    payload = _deep_merge(payload, overrides)
    return Settings.model_validate(payload)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached unified runtime settings root."""

    return load_settings()


def clear_knowledge_base_caches() -> None:
    """Reset KB catalog and index so the next access re-reads from disk."""

    get_settings.cache_clear()
    catalog.clear_catalog_caches()


def clear_settings_caches() -> None:
    """Clear cached settings and derived config state."""

    get_settings.cache_clear()
    clear_knowledge_base_caches()


def log_configuration_summary() -> None:
    """Load settings and log a safe startup summary."""

    settings = get_settings()
    kb_catalog = catalog.get_catalog(settings=settings.catalog)

    logger.info("Configuration loaded successfully")
    logger.info("vLLM URL: %s", settings.platform.vllm_base_url)
    logger.info("Default model: %s", settings.gateway.default_model)
    logger.info("Async inference enabled: %s", settings.gateway.async_enabled)
    logger.info(
        "Qdrant: %s:%s",
        settings.platform.qdrant_host,
        settings.platform.qdrant_port,
    )
    logger.info("RAG enabled: %s", settings.rag.rag_enabled)
    logger.info("Embeddings URL: %s", settings.platform.embeddings_url)
    logger.info("Embedding model: %s", settings.rag.embedding_model)
    logger.info("Embedding device: %s", settings.rag.embedding_device)
    logger.info(
        "Knowledge-base catalog path: %s",
        catalog.resolve_catalog_path(settings.catalog),
    )
    kb_names = [kb.name for task_cfg in kb_catalog.values() for kb in task_cfg.knowledge_bases]
    logger.info("Knowledge bases: %s", kb_names or "(none)")
    logger.info("Gateway URL (for UI): %s", settings.gateway.url)
    logger.info(
        "UI timeouts: health=%ss, chat=%ss",
        settings.ui.health_timeout,
        settings.ui.chat_timeout,
    )
