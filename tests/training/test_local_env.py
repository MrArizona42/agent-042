from __future__ import annotations

import sys
from pathlib import Path

import shared.local_env as local_env

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def test_load_local_env_returns_none_when_repo_root_unavailable(monkeypatch):
    def _raise(_start: Path | None = None) -> Path:
        raise FileNotFoundError("repo root unavailable")

    monkeypatch.setattr(local_env, "get_repo_root", _raise)

    assert local_env.load_local_env() is None
