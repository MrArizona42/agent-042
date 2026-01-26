"""RAG system configuration.

This module re-exports the unified settings from the shared config module.
The RAG module uses the same Settings class as the gateway, since RAG
is always used as a library by the gateway service.

For backward compatibility, RAGSettings is an alias for Settings.

Usage:
    from rag.config import get_settings

    settings = get_settings()
    print(settings.embedding_model)
"""

from __future__ import annotations

from shared.config import Settings, get_settings

# Backward compatibility alias
RAGSettings = Settings


def get_rag_settings() -> Settings:
    """Get RAG settings from environment.

    Deprecated: Use get_settings() instead.
    """
    return get_settings()


__all__ = [
    "RAGSettings",
    "Settings",
    "get_settings",
    "get_rag_settings",
]
