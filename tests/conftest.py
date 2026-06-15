from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("CONFIG__RUNTIME_PATH", str(PROJECT_ROOT / "runtime.toml"))
os.environ.setdefault("CONFIG__CATALOG_PATH", str(PROJECT_ROOT / "catalog.toml"))
os.environ.setdefault("VLLM__MODEL", "/models/Qwen/Qwen3-0.6B")

_TEST_ENV_KEYS = {
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_APP_DB",
    "RABBITMQ_DEFAULT_USER",
    "RABBITMQ_DEFAULT_PASS",
}

for line in (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        continue
    key, value = stripped.split("=", 1)
    if key.startswith("NETWORK__") or key in _TEST_ENV_KEYS:
        os.environ.setdefault(key, value)
