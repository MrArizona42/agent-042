"""Configuration for Celery worker."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class WorkerSettings(BaseSettings):
    """Worker configuration loaded from environment variables."""

    # Celery broker (RabbitMQ)
    celery_broker_url: str = Field(
        default="amqp://agent:agent@localhost:5672//",
        alias="CELERY_BROKER_URL",
        description="RabbitMQ connection URL",
    )

    # Redis for pub/sub streaming
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        alias="REDIS_URL",
        description="Redis connection URL for token streaming",
    )

    # vLLM server
    vllm_base_url: str = Field(
        default="http://localhost:8000",
        alias="VLLM_BASE_URL",
        description="URL where vLLM server is reachable",
    )

    vllm_model: str = Field(
        default="/models/Qwen/Qwen3-0.6B",
        alias="VLLM_MODEL",
        description="Default model for inference",
    )

    # Task settings
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

    class Config:
        extra = "ignore"


@lru_cache
def get_worker_settings() -> WorkerSettings:
    """Get cached worker settings."""
    return WorkerSettings()
