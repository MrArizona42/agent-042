"""Runtime config loading, caching, and startup summary."""

from __future__ import annotations

import logging
import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from app_config import catalog
from app_config.runtime.models import (
    CATALOG_CONFIG_PATH_ENV,
    RUNTIME_CONFIG_PATH_ENV,
    RuntimeConfig,
    Settings,
)

logger = logging.getLogger(__name__)


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
        raise RuntimeError(f"{RUNTIME_CONFIG_PATH_ENV} must point to the runtime TOML config file")

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


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"{name} must be set")
    return value


def _explicit_optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"{name} must be set; use an empty value when intentionally disabled")
    value = value.strip()
    return value or None


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
    runtime_payload["catalog"] = {"path": _required_env(CATALOG_CONFIG_PATH_ENV)}
    runtime_payload["gateway"]["api_key"] = _explicit_optional_env("GATEWAY__API_KEY")
    runtime_payload["auth"].update(
        {
            "google_client_id": _required_env("AUTH__GOOGLE_CLIENT_ID"),
            "google_client_secret": _required_env("AUTH__GOOGLE_CLIENT_SECRET"),
            "google_redirect_uri": _required_env("AUTH__GOOGLE_REDIRECT_URI"),
            "session_secret_key": _required_env("AUTH__SESSION_SECRET_KEY"),
            "internal_api_key": _required_env("AUTH__INTERNAL_API_KEY"),
        }
    )
    runtime_payload["eval"]["judge"]["api_key"] = _explicit_optional_env("EVAL__JUDGE__API_KEY")
    runtime_payload["postgres"] = {
        "user": _required_env("POSTGRES_USER"),
        "password": _required_env("POSTGRES_PASSWORD"),
        "app_db": _required_env("POSTGRES_APP_DB"),
    }
    runtime_payload["rabbitmq"] = {
        "default_user": _required_env("RABBITMQ_DEFAULT_USER"),
        "default_pass": _required_env("RABBITMQ_DEFAULT_PASS"),
    }
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
