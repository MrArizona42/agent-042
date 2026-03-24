"""Configuration for the eval-worker Celery process."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class EvalWorkerSettings(BaseSettings):
    """Eval-worker configuration loaded from environment variables."""

    celery_broker_url: str = Field(
        alias="CELERY_BROKER_URL",
        description="RabbitMQ connection URL (e.g. amqp://user:password@rabbitmq:5672//)",
    )

    class Config:
        extra = "ignore"


@lru_cache
def get_eval_worker_settings() -> EvalWorkerSettings:
    """Get cached eval-worker settings."""
    return EvalWorkerSettings()
