"""Runtime settings models and constants for all services."""

from __future__ import annotations

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

RUNTIME_CONFIG_PATH_ENV = "CONFIG__RUNTIME_PATH"
CATALOG_CONFIG_PATH_ENV = "CONFIG__CATALOG_PATH"


def secret_value(value: SecretStr | str | None) -> str | None:
    """Return the raw value for a secret-like field."""

    if value is None:
        return None
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return value


class VllmSettings(BaseModel):
    """Identity of the local vLLM model served by Compose."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    model: str = Field(description="Model served by the local vLLM container")


class NetworkServiceSettings(BaseModel):
    """Network coordinates for one Compose-managed service endpoint."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    internal_host: str = Field(description="Docker-network hostname")
    internal_port: int = Field(description="Port listened on inside Docker", ge=1, le=65535)
    host_port: int | None = Field(
        default=None,
        description="Port published on host loopback, if the service is published",
        ge=1,
        le=65535,
    )
    scheme: str | None = Field(
        default=None,
        description="URL scheme for HTTP-like services",
    )

    def internal_address(self) -> str:
        """Return host:port for Docker-internal clients."""

        return f"{self.internal_host}:{self.internal_port}"

    def internal_url(self) -> str:
        """Return scheme://host:port for Docker-internal clients."""

        if not self.scheme:
            raise ValueError("Service does not define a URL scheme")
        return f"{self.scheme}://{self.internal_address()}"

    def host_url(self, host: str = "localhost") -> str:
        """Return scheme://host:port for host-side clients."""

        if not self.scheme:
            raise ValueError("Service does not define a URL scheme")
        if self.host_port is None:
            raise ValueError("Service does not define a host-published port")
        return f"{self.scheme}://{host}:{self.host_port}"


class NetworkSettings(BaseModel):
    """Compose network coordinates used to derive project-owned endpoints."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    postgres: NetworkServiceSettings
    mlflow: NetworkServiceSettings
    vllm: NetworkServiceSettings
    qdrant_http: NetworkServiceSettings
    qdrant_grpc: NetworkServiceSettings
    rabbitmq_amqp: NetworkServiceSettings
    redis: NetworkServiceSettings
    redpanda_kafka: NetworkServiceSettings
    embeddings: NetworkServiceSettings
    reranker: NetworkServiceSettings
    code_sandbox: NetworkServiceSettings
    gateway: NetworkServiceSettings

    def service(self, name: str) -> NetworkServiceSettings:
        """Return a named service by canonical config name."""

        normalized = name.strip().lower().replace("-", "_")
        try:
            service = getattr(self, normalized)
        except AttributeError as exc:
            raise KeyError(f"Unknown network service: {name}") from exc
        if not isinstance(service, NetworkServiceSettings):
            raise KeyError(f"Unknown network service: {name}")
        return service

    def internal_url(self, name: str) -> str:
        """Return a Docker-internal URL for a named service."""

        return self.service(name).internal_url()

    def host_url(self, name: str, host: str = "localhost") -> str:
        """Return a host-side URL for a named service."""

        return self.service(name).host_url(host=host)


class PostgresSettings(BaseModel):
    """Native PostgreSQL credentials and project database names."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    user: str
    password: SecretStr
    app_db: str


class RabbitMqSettings(BaseModel):
    """Native RabbitMQ credentials."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    default_user: str
    default_pass: SecretStr


class PlatformSettings(BaseModel):
    """Legacy endpoint facade derived from canonical network settings."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)
    vllm_base_url: str = Field(
        description="URL where the shared vLLM server is reachable",
    )
    embeddings_url: str = Field(
        description="URL of the shared embeddings microservice",
    )
    qdrant_host: str = Field(
        description="Shared Qdrant server hostname",
    )
    qdrant_port: int = Field(
        description="Shared Qdrant server port",
        ge=1,
        le=65535,
    )
    mlflow_tracking_uri: str = Field(
        description="Shared MLflow tracking server URL",
    )
    redis_url: str = Field(
        description="Redis connection URL for shared streaming and coordination",
    )
    celery_broker_url: str = Field(
        description="RabbitMQ broker URL for shared Celery-based workflows",
    )
    kafka_bootstrap_servers: str = Field(
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


def _normalize_str_tuple(value: object) -> object:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return value


class GatewayConfig(BaseModel):
    """Gateway request handling and service behavior settings."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    api_key: SecretStr | None = Field(
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
    url: str | None = Field(
        default=None,
        description="Full URL to the gateway (used by UI)",
    )
    budget: BudgetSettings

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _normalize_cors_allow_origins(cls, value: object) -> object:
        return _normalize_str_tuple(value)


class RuntimeGatewayConfig(BaseModel):
    """Gateway runtime policy loaded from TOML."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

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
    budget: BudgetSettings

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _normalize_cors_allow_origins(cls, value: object) -> object:
        return _normalize_str_tuple(value)


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
    reranker_url: str | None = Field(
        default=None,
        description="URL of the reranker microservice",
    )
    reranker_model: str = Field(
        description="Cross-encoder model loaded by the reranker service",
    )
    data_root: Path = Field(
        default=Path("assets/rag_data"),
        description=(
            "Root directory for RAG artifacts (source caches, chunks, release "
            "manifests); relative paths resolve from the current working directory"
        ),
    )


class AuthSettings(BaseModel):
    """Gateway auth, session, and internal caller authentication settings."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    google_client_id: str = Field(
        description="Google OAuth2 client ID",
    )
    google_client_secret: SecretStr = Field(
        description="Google OAuth2 client secret",
    )
    google_redirect_uri: str = Field(
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
        description="Secret key for signing session cookies (32-byte hex)",
    )
    session_ttl_seconds: int = Field(
        description="Session TTL in seconds (default 24 hours)",
        ge=60,
    )
    internal_api_key: SecretStr = Field(
        description="Pre-shared API key for internal service-to-service calls "
        "(e.g. Airflow eval runner)",
    )


class RuntimeAuthSettings(BaseModel):
    """Auth runtime policy loaded from TOML."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    google_discovery_url: str = Field(
        description="Google OIDC discovery URL",
    )
    session_ttl_seconds: int = Field(
        description="Session TTL in seconds (default 24 hours)",
        ge=60,
    )


class CatalogConfig(BaseModel):
    """Settings for the shared task/knowledge-base/source catalog."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    path: Path = Field(
        description=(
            "Explicit path to the catalog TOML file; relative paths resolve from "
            "the current working directory"
        ),
    )

    @field_validator("path", mode="before")
    @classmethod
    def _normalize_path(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("catalog path must not be empty")
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
        return _normalize_str_tuple(value)


class JudgeSettings(BaseModel):
    """Resolved LLM-as-judge transport and model configuration."""

    model_config = ConfigDict(frozen=True)

    backend: Literal["local_vllm", "openai_compatible"]
    model: str
    base_url: str
    api_key: str | None
    context_window: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Context window for LlamaIndex LLM clients (e.g. the RAG benchmark "
            "judge), which need it to avoid depending on OpenAI's hardcoded "
            "model-name allowlist. None for consumers that call the judge "
            "backend directly over HTTP and don't need it; those consumers "
            "must not require this field."
        ),
    )
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
    api_key: SecretStr | None = Field(
        description="Optional API key for external OpenAI-compatible judge backends.",
    )
    context_window: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Required when backend=openai_compatible, since the external model's "
            "context size cannot be inferred from gateway settings. Ignored for "
            "backend=local_vllm, which always uses gateway.budget.model_max_tokens."
        ),
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


def _validate_judge_base_url(judge: Any) -> None:
    if judge.backend == "openai_compatible" and not judge.base_url.strip():
        raise ValueError(
            "eval.judge.base_url must be set when eval.judge.backend=openai_compatible"
        )


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
        _validate_judge_base_url(self.judge)
        return self

    def resolve_judge_settings(
        self,
        platform: PlatformSettings,
        *,
        local_context_window: int | None = None,
    ) -> JudgeSettings:
        """Resolve backend-specific judge settings to a concrete transport config.

        ``context_window`` on the result is ``None`` unless a caller that needs
        it supplies one: ``local_context_window`` backs the ``local_vllm`` case
        (that judge runs on the same locally-served model as generation, so it
        reuses the gateway's declared context window), and ``openai_compatible``
        passes through ``eval.judge.context_window`` as configured. Plain
        HTTP-calling judge consumers can ignore the field entirely; consumers
        that build a LlamaIndex LLM client (e.g. the RAG benchmark judge) must
        validate it is set before doing so.
        """
        judge = self.judge
        api_key = secret_value(judge.api_key)
        if judge.backend == "local_vllm":
            return JudgeSettings(
                backend="local_vllm",
                model=judge.model.strip(),
                base_url=platform.vllm_base_url,
                api_key=api_key.strip() or None if api_key is not None else None,
                context_window=local_context_window,
                timeout=judge.timeout,
                request_delay_seconds=judge.request_delay_seconds,
            )

        return JudgeSettings(
            backend="openai_compatible",
            model=judge.model.strip(),
            base_url=judge.base_url.strip(),
            api_key=api_key.strip() or None if api_key is not None else None,
            context_window=judge.context_window,
            timeout=judge.timeout,
            request_delay_seconds=judge.request_delay_seconds,
        )


class RuntimeEvalJudgeSettings(BaseModel):
    """Eval judge runtime policy loaded from TOML."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    backend: Literal["local_vllm", "openai_compatible"]
    model: str = Field(
        description="Judge model name used for LLM-as-judge scoring.",
    )
    base_url: str = Field(
        description="Base URL for external OpenAI-compatible judge backends.",
    )
    context_window: int | None = Field(
        default=None,
        ge=1,
        description=("Required when backend=openai_compatible; ignored for backend=local_vllm."),
    )
    timeout: float = Field(ge=1.0)
    request_delay_seconds: float = Field(ge=0.0)


class RuntimeEvalConfig(BaseModel):
    """Eval runtime policy loaded from TOML."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    judge: RuntimeEvalJudgeSettings
    metrics: EvalMetricSettings
    sandbox: EvalSandboxSettings

    @model_validator(mode="after")
    def _validate_judge_backend_config(self) -> "RuntimeEvalConfig":
        _validate_judge_base_url(self.judge)
        return self


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
    gateway: RuntimeGatewayConfig
    rag: RagSettings
    auth: RuntimeAuthSettings
    adapter_registry: AdapterRegistryConfig
    events: EventsSettings
    eval: RuntimeEvalConfig
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
    network: NetworkSettings
    postgres: PostgresSettings
    rabbitmq: RabbitMqSettings
    platform: PlatformSettings | None = None
    gateway: GatewayConfig
    rag: RagSettings
    auth: AuthSettings
    catalog: CatalogConfig
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
            file_secret_settings,
        )

    @model_validator(mode="before")
    @classmethod
    def _discard_derived_facade_inputs(cls, value: object) -> object:
        """Ignore legacy facade inputs; they are derived from canonical config."""

        if isinstance(value, dict):
            cleaned = dict(value)
            cleaned.pop("platform", None)
            return cleaned
        return value

    @model_validator(mode="after")
    def _derive_legacy_endpoint_facades(self) -> "Settings":
        """Populate legacy settings fields from canonical network coordinates."""

        platform = PlatformSettings(
            vllm_base_url=self.network.internal_url("vllm"),
            embeddings_url=self.network.internal_url("embeddings"),
            qdrant_host=self.network.qdrant_http.internal_host,
            qdrant_port=self.network.qdrant_http.internal_port,
            mlflow_tracking_uri=self.network.internal_url("mlflow"),
            redis_url=f"redis://{self.network.redis.internal_address()}/0",
            celery_broker_url=(
                "amqp://"
                f"{self.rabbitmq.default_user}:"
                f"{secret_value(self.rabbitmq.default_pass)}@"
                f"{self.network.rabbitmq_amqp.internal_address()}//"
            ),
            kafka_bootstrap_servers=self.network.redpanda_kafka.internal_address(),
        )
        auth = self.auth.model_copy(
            update={
                "agent042_db_url": (
                    "postgresql+asyncpg://"
                    f"{self.postgres.user}:"
                    f"{secret_value(self.postgres.password)}@"
                    f"{self.network.postgres.internal_address()}/"
                    f"{self.postgres.app_db}"
                )
            },
        )
        eval_config = self.eval.model_copy(
            update={
                "db_url": (
                    "postgresql://"
                    f"{self.postgres.user}:"
                    f"{secret_value(self.postgres.password)}@"
                    f"{self.network.postgres.internal_address()}/"
                    f"{self.postgres.app_db}"
                )
            },
        )
        gateway = self.gateway.model_copy(
            update={"url": self.network.internal_url("gateway")},
        )
        rag = self.rag.model_copy(
            update={"reranker_url": self.network.internal_url("reranker")},
        )

        object.__setattr__(self, "platform", platform)
        object.__setattr__(self, "auth", auth)
        object.__setattr__(self, "eval", eval_config)
        object.__setattr__(self, "gateway", gateway)
        object.__setattr__(self, "rag", rag)
        return self
