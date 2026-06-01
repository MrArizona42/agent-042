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

import logging
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import AliasChoices, BaseModel, Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.local_env import get_repo_root, load_local_env

logger = logging.getLogger(__name__)

# =========================================================================
# Knowledge Base Registry / operator registry
# =========================================================================

_DEFAULT_OPERATOR_REGISTRY_PATH = Path(__file__).resolve().parent / "operator_registry.toml"


class AliasConfig(BaseModel):
    """Per-alias query-time RAG configuration.

    No defaults — every field must be explicit in the operator registry.
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
    """Materialized task entry used at runtime."""

    task: str
    label: str = ""
    routing_description: str
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    knowledge_bases: list[KBConfig] = Field(default_factory=list)


class AliasProfileRefConfig(BaseModel):
    """Reference to a reusable alias profile declared at the registry root."""

    profile: str


class AliasProfileConfig(BaseModel):
    """Reusable alias profile declared at the registry root of the TOML file."""

    top_k: int
    score_threshold: float
    reranker: Optional[str] = None
    retrieval_strategy: Literal["dense", "hybrid", "sparse"]
    reranker_multiplier: int


class RegistryTaskConfig(BaseModel):
    """Normalized task entry in the TOML operator registry."""

    enabled: bool = True
    label: str = ""
    routing_description: str
    kb_refs: list[str] = Field(default_factory=list)
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)


class RegistryKBConfig(BaseModel):
    """Normalized knowledge-base entry in the TOML operator registry."""

    enabled: bool = True
    default_alias: str
    aliases: dict[str, AliasProfileRefConfig]
    update_strategy: Literal["incremental", "replace"] = "replace"
    label: str = ""
    description: str = ""
    selection_description: str


class OperatorRegistryConfig(BaseModel):
    """Root model for the TOML-based operator registry."""

    schema_version: int = 1
    tasks: dict[str, RegistryTaskConfig] = Field(default_factory=dict)
    knowledge_bases: dict[str, RegistryKBConfig] = Field(default_factory=dict)
    alias_profiles: dict[str, AliasProfileConfig] = Field(default_factory=dict)


def _materialize_operator_registry(
    registry_cfg: OperatorRegistryConfig,
) -> tuple[dict[str, TaskConfig], dict[str, KBConfig]]:
    """Build the runtime task registry from the normalized TOML shape."""

    kb_index: dict[str, KBConfig] = {}
    for kb_name, kb_cfg in registry_cfg.knowledge_bases.items():
        if not kb_cfg.enabled:
            continue

        aliases: dict[str, AliasConfig] = {}
        for alias_name, alias_ref in kb_cfg.aliases.items():
            profile_cfg = registry_cfg.alias_profiles.get(alias_ref.profile)
            if profile_cfg is None:
                raise ValueError(
                    f"KB '{kb_name}' alias '{alias_name}' references unknown "
                    f"alias profile '{alias_ref.profile}'"
                )
            aliases[alias_name] = AliasConfig(
                top_k=profile_cfg.top_k,
                score_threshold=profile_cfg.score_threshold,
                reranker=profile_cfg.reranker,
                retrieval_strategy=profile_cfg.retrieval_strategy,
                reranker_multiplier=profile_cfg.reranker_multiplier,
            )

        kb_index[kb_name] = KBConfig(
            name=kb_name,
            default_alias=kb_cfg.default_alias,
            aliases=aliases,
            update_strategy=kb_cfg.update_strategy,
            label=kb_cfg.label,
            description=kb_cfg.description,
            selection_description=kb_cfg.selection_description,
        )

    task_registry: dict[str, TaskConfig] = {}
    for task_name, task_cfg in registry_cfg.tasks.items():
        if not task_cfg.enabled:
            continue

        task_knowledge_bases: list[KBConfig] = []
        for kb_name in task_cfg.kb_refs:
            if kb_name not in registry_cfg.knowledge_bases:
                raise ValueError(f"Task '{task_name}' references unknown KB '{kb_name}'")
            kb_runtime_cfg = kb_index.get(kb_name)
            if kb_runtime_cfg is not None:
                task_knowledge_bases.append(kb_runtime_cfg)

        task_registry[task_name] = TaskConfig(
            task=task_name,
            label=task_cfg.label,
            routing_description=task_cfg.routing_description,
            adapter=task_cfg.adapter.model_copy(deep=True),
            knowledge_bases=task_knowledge_bases,
        )

    return task_registry, kb_index


def _load_knowledge_bases(
    path: Path | str,
) -> tuple[dict[str, TaskConfig], dict[str, KBConfig]]:
    """Load the operator registry from a TOML file.

    Args:
        path: Path to the operator registry file.

    Returns:
        Tuple of (task registry, flat KB index keyed by KB name).

    Raises:
        ValueError: If the registry path is not a TOML file.
    """
    path = Path(path)

    if not path.exists():
        logger.warning("Operator registry not found at %s — using empty registry", path)
        return {}, {}

    if path.suffix.lower() != ".toml":
        raise ValueError(f"Operator registry must be a TOML file (got '{path.name}')")

    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    return _materialize_operator_registry(OperatorRegistryConfig(**raw))


@lru_cache(maxsize=None)
def _load_knowledge_bases_cached(
    path: str,
) -> tuple[dict[str, TaskConfig], dict[str, KBConfig]]:
    """Cached path-based loader for the materialized runtime registry."""

    return _load_knowledge_bases(path)


# Registry loader state
_KB_OVERRIDE_REGISTRY: dict[str, TaskConfig] | None = None
_KB_OVERRIDE_INDEX: dict[str, KBConfig] | None = None


def _build_kb_index(registry: dict[str, TaskConfig]) -> dict[str, KBConfig]:
    """Build a flat KB index from a task-scoped runtime registry."""

    index: dict[str, KBConfig] = {}
    for task_cfg in registry.values():
        for kb_cfg in task_cfg.knowledge_bases:
            if kb_cfg.name in index:
                if index[kb_cfg.name] is not kb_cfg:
                    raise ValueError(
                        f"Duplicate KB name '{kb_cfg.name}' found across tasks. "
                        f"KB names must be unique."
                    )
                continue
            index[kb_cfg.name] = kb_cfg
    return index


def set_knowledge_base_registry_override(
    registry: dict[str, TaskConfig],
    *,
    index: dict[str, KBConfig] | None = None,
) -> None:
    """Install an in-memory registry override used ahead of disk-backed loading.

    This is intended for tests that need deterministic registry contents without
    touching on-disk config files.
    """

    global _KB_OVERRIDE_REGISTRY, _KB_OVERRIDE_INDEX  # noqa: PLW0603

    _KB_OVERRIDE_REGISTRY = registry
    _KB_OVERRIDE_INDEX = index if index is not None else _build_kb_index(registry)


def clear_knowledge_base_registry_override() -> None:
    """Remove any installed in-memory registry override."""

    global _KB_OVERRIDE_REGISTRY, _KB_OVERRIDE_INDEX  # noqa: PLW0603

    _KB_OVERRIDE_REGISTRY = None
    _KB_OVERRIDE_INDEX = None


def _configured_operator_registry_path(settings: RegistrySettings | None = None) -> str:
    """Resolve the configured operator registry path before path normalization."""

    if settings is None:
        return get_registry_settings().operator_registry_path.strip()

    return settings.operator_registry_path.strip()


def _resolve_knowledge_bases_path(settings: RegistrySettings | None = None) -> Path:
    """Resolve the active knowledge-base registry path.

    Explicit overrides support absolute paths or repository-root-relative paths.
    Falling back to the bundled registry keeps local scripts and container runs
    aligned when no override is configured.
    """
    configured_path = _configured_operator_registry_path(settings)
    if not configured_path:
        return _DEFAULT_OPERATOR_REGISTRY_PATH

    path = Path(configured_path).expanduser()
    if path.is_absolute():
        return path

    try:
        return get_repo_root() / path
    except FileNotFoundError:
        return path


def _get_loaded_knowledge_base_state(
    *, settings: RegistrySettings | None = None
) -> tuple[dict[str, TaskConfig], dict[str, KBConfig]]:
    """Return the effective runtime registry and flat index.

    The returned data comes from the explicit in-memory override when installed,
    otherwise from the cached path-based loader.
    """

    if _KB_OVERRIDE_REGISTRY is not None and _KB_OVERRIDE_INDEX is not None:
        return _KB_OVERRIDE_REGISTRY, _KB_OVERRIDE_INDEX

    path = _resolve_knowledge_bases_path(settings).resolve()
    return _load_knowledge_bases_cached(str(path))


def get_knowledge_bases(
    *, settings: RegistrySettings | None = None
) -> dict[str, TaskConfig]:
    """Return the knowledge-base registry (cached after first call).

    Path is resolved from ``RegistrySettings.operator_registry_path`` when
    provided, or from the cached registry settings root otherwise. The cache is
    keyed by the resolved source path so an override change reloads the registry.

    Returns:
        Mapping of ``task_name`` → ``TaskConfig``.
    """
    registry, _ = _get_loaded_knowledge_base_state(settings=settings)
    return registry


def get_kb_config(
    kb_name: str,
    *,
    settings: RegistrySettings | None = None,
) -> KBConfig | None:
    """Look up a KB by name (O(1) dict lookup).

    Returns the ``KBConfig`` for *kb_name* or ``None`` if not found.
    """
    _, index = _get_loaded_knowledge_base_state(settings=settings)
    return index.get(kb_name)


def get_kb_names(*, settings: RegistrySettings | None = None) -> list[str]:
    """Flat list of all KB names across all tasks."""
    _, index = _get_loaded_knowledge_base_state(settings=settings)
    return list(index.keys())


def validate_kb_alias(
    kb: str,
    alias: str | None = None,
    *,
    settings: RegistrySettings | None = None,
) -> None:
    """Raise ValueError with a consistent message if kb or alias is unknown.

    When *alias* is ``None`` only the KB name is validated.
    """
    kb_cfg = get_kb_config(kb, settings=settings)
    if kb_cfg is None:
        raise ValueError(f"KB '{kb}' not found. Available: {get_kb_names(settings=settings)}")
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
        description=(
            "Override path to knowledge_bases.json; relative paths resolve from the "
            "repository root, empty uses the bundled default"
        ),
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
        REGISTRY_OPERATOR_REGISTRY_PATH: Optional override path to the operator registry file.
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
    operator_registry_path: str = Field(
        default="",
        description=(
            "Override path to the operator registry file (JSON or TOML); relative "
            "paths resolve from the repository root, empty uses the bundled default"
        ),
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

    @property
    def platform(self) -> RegistryPlatformSettings:
        return RegistryPlatformSettings(
            mlflow_tracking_uri=self.mlflow_tracking_uri,
            vllm_base_url=self.vllm_base_url,
        )

    @property
    def storage(self) -> RegistryStorageSettings:
        return RegistryStorageSettings(
            adapters_dir=self.adapters_dir,
            operator_registry_path=self.operator_registry_path,
        )

    @property
    def sync(self) -> RegistrySyncSettings:
        return RegistrySyncSettings(
            production_alias=self.production_alias,
            sync_aliases_csv=self.sync_aliases_csv,
            auto_sync=self.auto_sync,
        )


# Backward compatibility alias used across the existing codebase.
ModelRegistrySettings = RegistrySettings


class JudgeSettings(BaseModel):
    """Resolved LLM-as-judge transport and model configuration."""

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


class EvalSettings(BaseSettings):
    """Settings for the evaluation runner.

    Environment Variables:
        EVAL_GATEWAY_URL: Gateway URL for generation evals.
        EVAL_JUDGE_BACKEND: Judge backend (local_vllm or openai_compatible).
        EVAL_JUDGE_MODEL: Judge model name.
        EVAL_JUDGE_BASE_URL: Base URL for external OpenAI-compatible judge backends.
        EVAL_JUDGE_API_KEY: Optional API key for external OpenAI-compatible backends.
        EVAL_JUDGE_TIMEOUT: Timeout for judge HTTP requests.
        EVAL_JUDGE_REQUEST_DELAY_SECONDS: Optional delay between judge requests.
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
    judge_backend: Literal["local_vllm", "openai_compatible"] = Field(
        default="local_vllm",
        description=(
            "Backend used for LLM-as-judge scoring. local_vllm talks directly to the "
            "project's canonical vLLM endpoint."
        ),
    )
    judge_model: str = Field(
        description=(
            "Judge model name. This setting is mandatory so the eval backend does not "
            "silently inherit or reassign models at runtime."
        ),
    )
    judge_base_url: str = Field(
        default="",
        description=(
            "Base URL for external OpenAI-compatible judge backends. Ignored when "
            "judge_backend=local_vllm."
        ),
    )
    judge_api_key: str = Field(
        default="",
        description=(
            "Optional API key for external OpenAI-compatible judge backends. Ignored "
            "when judge_backend=local_vllm."
        ),
    )
    judge_timeout: float = Field(
        default=60.0,
        description="Timeout for judge HTTP requests in seconds",
        ge=1.0,
    )
    judge_request_delay_seconds: float = Field(
        default=0.0,
        description="Optional delay inserted between judge requests in seconds",
        ge=0.0,
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

    @model_validator(mode="after")
    def _validate_judge_backend_config(self) -> "EvalSettings":
        if not self.judge_model.strip():
            raise ValueError("EVAL_JUDGE_MODEL must be set")
        if self.judge_backend == "openai_compatible":
            if not self.judge_base_url.strip():
                raise ValueError(
                    "EVAL_JUDGE_BASE_URL must be set when EVAL_JUDGE_BACKEND=openai_compatible"
                )
        return self

    def resolve_judge_settings(self) -> JudgeSettings:
        """Resolve backend-specific judge settings to a concrete transport config."""
        if self.judge_backend == "local_vllm":
            return JudgeSettings(
                backend="local_vllm",
                model=self.judge_model.strip(),
                base_url=get_platform_settings().vllm_base_url,
                api_key=self.judge_api_key.strip() or None,
                timeout=self.judge_timeout,
                request_delay_seconds=self.judge_request_delay_seconds,
            )

        return JudgeSettings(
            backend="openai_compatible",
            model=self.judge_model.strip(),
            base_url=self.judge_base_url.strip(),
            api_key=self.judge_api_key.strip() or None,
            timeout=self.judge_timeout,
            request_delay_seconds=self.judge_request_delay_seconds,
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
    clear_knowledge_base_registry_override()
    get_registry_settings.cache_clear()
    _load_knowledge_bases_cached.cache_clear()


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
    logger.info(f"  Knowledge-base registry path: {_resolve_knowledge_bases_path(settings)}")
    kb_names = [kb.name for tc in kb_registry.values() for kb in tc.knowledge_bases]
    logger.info(f"  Knowledge bases: {kb_names or '(none)'}")
    logger.info(f"  Gateway URL (for UI): {settings.url}")
    logger.info(
        f"  UI timeouts: health={ui_settings.health_timeout}s, chat={ui_settings.chat_timeout}s"
    )
