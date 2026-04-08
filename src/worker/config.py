"""Configuration for Celery worker."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    """Worker-specific configuration loaded from environment variables.

    Only contains fields unique to the Celery worker process.
    Shared fields (redis_url, vllm_base_url, default_model) are read from
    ``shared.config.Settings`` via ``get_settings()``.
    """

    # Celery broker (RabbitMQ) — no default; CELERY_BROKER_URL must be set.
    celery_broker_url: str = Field(
        alias="CELERY_BROKER_URL",
        description="RabbitMQ connection URL (e.g. amqp://user:password@rabbitmq:5672//)",
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

    model_config = SettingsConfigDict(extra="ignore")


@lru_cache
def get_worker_settings() -> WorkerSettings:
    """Get cached worker settings."""
    return WorkerSettings()
