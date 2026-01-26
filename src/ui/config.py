"""UI service configuration.

Re-exports shared settings for UI service usage.
"""

from __future__ import annotations

from shared.config import Settings, UISettings, get_settings, get_ui_settings

__all__ = [
    "Settings",
    "UISettings",
    "get_settings",
    "get_ui_settings",
]
