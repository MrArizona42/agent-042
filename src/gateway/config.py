"""Gateway service configuration.

This module re-exports the unified settings from the shared config module.
For backward compatibility, GatewaySettings is an alias for Settings.

Usage:
    from gateway.config import get_settings

    settings = get_settings()
"""

from __future__ import annotations

from shared.config import Settings, get_settings, validate_settings_on_startup

# Backward compatibility alias
GatewaySettings = Settings

__all__ = [
    "GatewaySettings",
    "Settings",
    "get_settings",
    "validate_settings_on_startup",
]
