from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("CONFIG__RUNTIME_PATH", str(PROJECT_ROOT / "config" / "runtime.toml"))
