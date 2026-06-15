from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("CONFIG__RUNTIME_PATH", str(PROJECT_ROOT / "runtime.toml"))
os.environ.setdefault("CONFIG__CATALOG_PATH", str(PROJECT_ROOT / "catalog.toml"))
os.environ.setdefault("VLLM__MODEL", "/models/Qwen/Qwen3-0.6B")
