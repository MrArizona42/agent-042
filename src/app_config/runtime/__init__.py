"""Application runtime configuration facade.

The implementation still lives in ``shared.config`` during this phase. New code
should prefer importing through this package when it needs application config.
"""

from shared.config import (
    AdapterRegistryConfig,
    CatalogConfig,
    RuntimeConfig,
    Settings,
    clear_knowledge_base_caches,
    clear_settings_caches,
    get_settings,
    load_runtime_config,
    load_settings,
    log_configuration_summary,
    resolve_runtime_config_path,
    secret_value,
)

__all__ = [
    "AdapterRegistryConfig",
    "CatalogConfig",
    "RuntimeConfig",
    "Settings",
    "clear_knowledge_base_caches",
    "clear_settings_caches",
    "get_settings",
    "load_runtime_config",
    "load_settings",
    "log_configuration_summary",
    "resolve_runtime_config_path",
    "secret_value",
]
