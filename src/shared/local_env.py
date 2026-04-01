"""Helpers for explicit local dotenv loading.

These helpers are for local Python entrypoints, notebooks, and scripts.
Containerized deployments should inject environment variables directly and
should not rely on implicit dotenv discovery.
"""

from __future__ import annotations

import logging
from pathlib import Path

import dotenv

logger = logging.getLogger(__name__)


def get_repo_root(start: Path | None = None) -> Path:
    """Return the repository root directory.

    The repo root is identified by the presence of both ``pyproject.toml`` and
    the top-level ``src`` directory.
    """
    anchor = start.resolve() if start is not None else Path(__file__).resolve()
    current = anchor if anchor.is_dir() else anchor.parent

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
            return candidate

    raise FileNotFoundError(f"Could not determine repository root from {anchor}")


def resolve_local_env_path(env_file: str | Path, *, repo_root: Path | None = None) -> Path:
    """Resolve *env_file* relative to the repository root when needed."""
    path = Path(env_file)
    if path.is_absolute():
        return path

    root = repo_root.resolve() if repo_root is not None else get_repo_root()
    return root / path


def load_local_env(
    env_file: str | Path | None = None,
    *,
    repo_root: Path | None = None,
    override: bool = False,
    legacy_fallbacks: tuple[str | Path, ...] = (),
) -> Path | None:
    """Load local env variables from the canonical repo-root `.env`.

    Args:
        env_file: Optional explicit env file path. Relative paths are resolved
            from the repository root.
        repo_root: Optional repository root override.
        override: Whether loaded env vars should override existing process env.
        legacy_fallbacks: Additional legacy env file paths to try if the
            canonical path is missing.

    Returns:
        The env file path that was loaded, or ``None`` if no candidate exists.
    """
    root = repo_root.resolve() if repo_root is not None else get_repo_root()
    canonical_env = root / ".env"

    candidates: list[Path] = []
    if env_file is not None:
        candidates.append(resolve_local_env_path(env_file, repo_root=root))
    else:
        candidates.append(canonical_env)

    candidates.extend(resolve_local_env_path(path, repo_root=root) for path in legacy_fallbacks)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)

        if not candidate.exists():
            continue

        dotenv.load_dotenv(candidate, override=override)
        if candidate == canonical_env:
            logger.info("Loaded local env from %s", candidate)
        else:
            logger.warning(
                "Loaded legacy env file %s; migrate these values to %s",
                candidate,
                canonical_env,
            )
        return candidate

    return None
