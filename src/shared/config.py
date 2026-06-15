"""Unified configuration for all services.

Runtime behavior is loaded from the explicit TOML file pointed to by
``CONFIG__RUNTIME_PATH``. Infrastructure coordinates and secrets are still
read from process environment variables injected by the caller.
"""

from __future__ import annotations

import logging
import os
import tomllib
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

RUNTIME_CONFIG_PATH_ENV = "CONFIG__RUNTIME_PATH"
CATALOG_CONFIG_PATH_ENV = "CONFIG__CATALOG_PATH"


class VllmSettings(BaseModel):
    """Identity of the local vLLM model served by Compose."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    model: str = Field(description="Model served by the local vLLM container")


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

class BudgetSettings(BaseModel):
    """Prompt and response budgeting settings for online inference."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    model_max_tokens: int = Field(
        description="Configured max model window used for prompt/response budgeting",
        ge=1,
    )
    chars_per_token: float = Field(
        description="Approximate character-to-token ratio used for gateway shaping",
        gt=0.0,
    )
    budget_guard: int = Field(
        description="Reserved safeguard gap for estimation and chat-template overhead",
        ge=0,
    )
    budget_system: int = Field(
        description="Approximate token budget reserved for the system prompt",
        ge=1,
    )
    budget_turn: int = Field(
        description="Approximate token budget reserved for the current user turn",
        ge=1,
    )
    min_budget_history: int = Field(
        description="Minimum approximate token budget reserved for chat history",
        ge=0,
    )
    budget_rag: int = Field(
        description="Approximate token budget reserved for all retrieved RAG context",
        ge=0,
    )
    min_response_budget: int = Field(
        description="Minimum exact response token budget required before generation",
        ge=1,
    )


class GatewayConfig(BaseModel):
    """Gateway request handling and service behavior settings."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    api_key: SecretStr | None = Field(
        default=None,
        description="Optional API key for vLLM authentication",
    )
    vllm_timeout: float = Field(
        description="Timeout for vLLM requests in seconds",
        ge=1.0,
    )
    repetition_penalty: float = Field(
        description="Repetition penalty applied to all generation requests to prevent token loops",
        ge=1.0,
    )
    streaming_timeout: float = Field(
        description="Idle timeout for Redis Pub/Sub streaming in seconds",
        ge=1.0,
    )
    embeddings_timeout: float = Field(
        description="Timeout for embeddings service HTTP requests in seconds",
        ge=1.0,
    )
    async_enabled: bool = Field(
        description="Enable async inference via Celery workers",
    )
    cors_allow_origins: Annotated[tuple[str, ...], NoDecode] = Field(
        description="Allowed CORS origins.",
    )
    service_name: str = Field(
        description="Service name displayed in API docs",
    )
    url: str = Field(
        default="http://localhost:9001",
        description="Full URL to the gateway (used by UI)",
    )
    budget: BudgetSettings

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
        description="Batch size for embedding generation during RAG builds",
        ge=1,
    )
    qdrant_upsert_batch_size: int = Field(
        description="Batch size for Qdrant upserts during RAG materialization",
        ge=1,
    )


class RagSettings(BaseModel):
    """Gateway RAG behavior and embedding model settings."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    enabled: bool = Field(description="Enable RAG functionality")
    embedding_model: str = Field(
        description="HuggingFace model for embeddings",
    )
    embedding_device: Literal["cpu", "cuda", "mps"] = Field(
        description="Device for embedding model",
    )
    build: RagBuildSettings
    kb_selection_threshold: float = Field(
        description="Cosine similarity threshold for automatic KB selection",
        ge=0.0,
        le=1.0,
    )
    task_classification_threshold: float = Field(
        description=(
            "Minimum cosine similarity for task classification; "
            "0.0 means always pick the closest task"
        ),
        ge=0.0,
        le=1.0,
    )
    strict_startup: bool = Field(
        description="If True, raise on legacy / invalid Qdrant collections at startup "
        "instead of logging and marking them unavailable",
    )
    sparse_encoder_model: str = Field(
        description="fastembed model name for sparse (BM25) vector encoding",
    )
    reranker_url: str = Field(
        default="http://reranker:8101",
        description="URL of the reranker microservice",
    )
    reranker_model: str = Field(
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
        description="Local directory for downloaded LoRA adapters",
    )
    production_alias: str | None = Field(
        description="MLflow alias that marks an adapter as production-ready.",
    )
    sync_aliases: Annotated[tuple[str, ...], NoDecode] = Field(
        description="MLflow aliases to sync to vLLM.",
    )
    auto_sync: bool = Field(
        description="Automatically sync production adapters on startup",
    )

    @field_validator("production_alias", mode="before")
    @classmethod
    def _normalize_production_alias(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

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
        ge=1.0,
        description="Timeout for one judge request in seconds",
    )
    request_delay_seconds: float = Field(
        ge=0.0,
        description="Optional delay inserted between consecutive judge requests",
    )


class EvalJudgeSettings(BaseModel):
    """Raw judge configuration from the eval section."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    backend: Literal["local_vllm", "openai_compatible"]
    model: str = Field(
        description="Judge model name used for LLM-as-judge scoring.",
    )
    base_url: str = Field(
        description="Base URL for external OpenAI-compatible judge backends.",
    )
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(""),
        description="Optional API key for external OpenAI-compatible judge backends.",
    )
    timeout: float = Field(ge=1.0)
    request_delay_seconds: float = Field(ge=0.0)


class EvalMetricSettings(BaseModel):
    """Eval metric and generation controls."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    bert_score_model: str = Field(
        description="Model for BERTScore computation",
    )
    temperature: float = Field(ge=0.0)
    max_completion_tokens: int = Field(ge=1)


class EvalSandboxSettings(BaseModel):
    """Sandbox execution limits for code evals."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    code_exec_timeout: int = Field(ge=1)
    code_exec_mem_limit: str = Field(
        description=(
            "Memory limit string for sandboxed code execution. Accepted for config "
            "compatibility; not currently enforced by bwrap."
        ),
    )
    code_exec_cpus: float = Field(ge=0.1)


class EvalConfig(BaseModel):
    """Settings for the evaluation runner."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    judge: EvalJudgeSettings
    metrics: EvalMetricSettings
    sandbox: EvalSandboxSettings
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


class EventsSettings(BaseModel):
    """Runtime event-stream settings."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    inference_topic: str = Field(description="Kafka-compatible inference lifecycle topic")


class UIConfig(BaseModel):
    """UI-specific request timeout settings."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    health_timeout: float = Field(
        description="Timeout for health check requests in seconds",
        ge=1.0,
    )
    models_timeout: float = Field(
        description="Timeout for models list requests in seconds",
        ge=1.0,
    )
    chat_timeout: float = Field(
        description="Timeout for chat completion requests in seconds",
        ge=1.0,
    )


class WorkerConfig(BaseModel):
    """Worker-specific runtime settings."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    default_timeout: int = Field(
        description="Default task timeout in seconds",
    )
    max_retries: int = Field(
        description="Maximum number of task retries",
    )
    retry_delay: int = Field(
        description="Delay between retries in seconds",
    )
    pool: str = Field(
        description="Celery execution pool for gateway inference tasks",
    )
    concurrency: int = Field(
        description="Concurrent worker slots for gateway inference tasks",
        ge=1,
    )
    send_task_events: bool = Field(
        description="Emit Celery task events so Flower can observe queued/running tasks",
    )
    cancel_long_running_tasks_on_connection_loss: bool = Field(
        description="Cancel in-flight tasks if the broker connection is lost",
    )


class RuntimeConfig(BaseModel):
    """Validated non-secret runtime policy loaded from TOML."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    gateway: GatewayConfig
    rag: RagSettings
    auth: AuthSettings
    adapter_registry: AdapterRegistryConfig
    events: EventsSettings
    eval: EvalConfig
    worker: WorkerConfig
    ui: UIConfig

    @model_validator(mode="before")
    @classmethod
    def _reject_non_runtime_keys(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value

        forbidden_by_section = {
            "vllm": {
                "model",
                "dtype",
                "quantization",
                "gpu_utilization",
                "gpu_count",
                "max_num_seqs",
                "max_num_batched_tokens",
                "kv_cache_dtype",
                "max_loras",
                "max_lora_rank",
                "allow_runtime_lora_updating",
            },
            "gateway": {"api_key", "url", "default_model"},
            "rag": {"rag_enabled", "rag_strict_startup", "reranker_url"},
            "auth": {
                "google_client_id",
                "google_client_secret",
                "google_redirect_uri",
                "agent042_db_url",
                "session_secret_key",
                "internal_api_key",
            },
            "eval.judge": {"api_key"},
        }

        violations: list[str] = []
        for section, forbidden_keys in forbidden_by_section.items():
            current: object = value
            for part in section.split("."):
                if not isinstance(current, dict):
                    current = None
                    break
                current = current.get(part)
            if not isinstance(current, dict):
                continue
            for key in forbidden_keys & current.keys():
                violations.append(f"{section}.{key}")

        if violations:
            keys = ", ".join(sorted(violations))
            raise ValueError(f"Runtime TOML contains non-runtime keys: {keys}")

        return value


class Settings(BaseSettings):
    """Unified runtime configuration root."""

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        populate_by_name=True,
        frozen=True,
    )

    vllm: VllmSettings
    platform: PlatformSettings = Field(default_factory=PlatformSettings)
    gateway: GatewayConfig
    rag: RagSettings
    auth: AuthSettings
    catalog: CatalogConfig = Field(default_factory=CatalogConfig)
    adapter_registry: AdapterRegistryConfig
    events: EventsSettings
    eval: EvalConfig
    worker: WorkerConfig
    ui: UIConfig

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


def resolve_runtime_config_path(runtime_path: str | Path | None = None) -> Path:
    """Resolve the explicit runtime TOML path from argument or process env."""

    raw_path = runtime_path if runtime_path is not None else os.getenv(RUNTIME_CONFIG_PATH_ENV)
    if raw_path is None or not str(raw_path).strip():
        raise RuntimeError(
            f"{RUNTIME_CONFIG_PATH_ENV} must point to the runtime TOML config file"
        )

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def load_runtime_config(runtime_path: str | Path | None = None) -> RuntimeConfig:
    """Load and validate the non-secret runtime policy TOML."""

    path = resolve_runtime_config_path(runtime_path)
    if not path.exists():
        raise FileNotFoundError(f"Runtime config file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Runtime config path is not a file: {path}")

    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    return RuntimeConfig.model_validate(raw)


def secret_value(value: SecretStr | str | None) -> str | None:
    """Return the raw value for a secret-like field."""

    if value is None:
        return None
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return value


def load_settings(
    overrides: dict[str, Any] | None = None,
    *,
    runtime_path: str | Path | None = None,
) -> Settings:
    """Build settings from runtime TOML, process env, and explicit overrides."""

    runtime = load_runtime_config(runtime_path)
    runtime_payload = runtime.model_dump(
        exclude={"schema_version"},
        exclude_unset=True,
        exclude_none=False,
    )
    catalog_path = os.getenv(CATALOG_CONFIG_PATH_ENV)
    if catalog_path is not None and catalog_path.strip():
        runtime_payload["catalog"] = {"path": catalog_path}
    settings = Settings(**runtime_payload)
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
    logger.info("vLLM model: %s", settings.vllm.model)
    logger.info("Async inference enabled: %s", settings.gateway.async_enabled)
    logger.info(
        "Qdrant: %s:%s",
        settings.platform.qdrant_host,
        settings.platform.qdrant_port,
    )
    logger.info("RAG enabled: %s", settings.rag.enabled)
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
