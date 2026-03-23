"""MLflow Model Registry utilities for LoRA adapter management.

This module re-exports the canonical implementation from
``src/shared/model_registry.py`` so that the ``train_adapter`` package
(and any other code under ``experiments/``) can continue to use::

    from train_adapter.registry import AdapterRegistry
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure ``src/`` is importable when running from the experiments tree.
_SRC_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from shared.model_registry import (  # noqa: E402, F401
    AdapterRegistry,
    RegisteredAdapter,
)
