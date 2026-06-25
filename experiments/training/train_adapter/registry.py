"""MLflow Model Registry utilities for LoRA adapter management.

This module re-exports the canonical implementation from
``src/services/adapter_sync/model_registry.py`` so that the ``train_adapter`` package
(and any other code under ``experiments/``) can continue to use::

    from train_adapter.registry import AdapterRegistry
"""

from __future__ import annotations

from services.adapter_sync.model_registry import (  # noqa: F401
    AdapterRegistry,
    RegisteredAdapter,
)
