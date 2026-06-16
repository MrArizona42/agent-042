"""Compatibility re-export — real implementation is in app_config.catalog.models."""

from app_config.catalog.models import AdapterConfig, AliasConfig, KBConfig, TaskConfig

__all__ = ["AdapterConfig", "AliasConfig", "KBConfig", "TaskConfig"]
