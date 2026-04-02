"""MLflow Model Registry utilities for LoRA adapter management.

This module re-exports the canonical implementation from
``src/shared/model_registry.py`` so that the ``train_adapter`` package
(and any other code under ``experiments/``) can continue to use::

    from train_adapter.registry import AdapterRegistry
"""

from __future__ import annotations

from shared.model_registry import (  # noqa: F401
    AdapterRegistry,
    RegisteredAdapter,
)
