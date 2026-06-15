#!/usr/bin/env python3
"""Host-side wrapper for MLflow/vLLM adapter registry operations."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shared.model_registry import _cmd_list, _cmd_sync  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Host env file to load before running the registry command.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser("sync")
    sync.add_argument("--adapters-dir")
    sync.add_argument("--vllm-url")
    sync.add_argument("--aliases")

    list_cmd = subparsers.add_parser("list")
    list_cmd.add_argument("--aliases")
    return parser


def _load_host_env(env_file: str) -> Path:
    path = Path(env_file).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Host env file not found: {path}")

    dotenv.load_dotenv(path, override=False)
    _derive_host_env()
    logging.getLogger(__name__).info("Loaded host env from %s", path)
    return path


def _derive_host_env() -> None:
    project_root = Path(os.environ.get("PROJECT_ROOT", REPO_ROOT)).expanduser()
    os.environ.setdefault("CONFIG__RUNTIME_PATH", str(project_root / "runtime.toml"))
    os.environ.setdefault("CONFIG__CATALOG_PATH", str(project_root / "catalog.toml"))

    public_base_url = os.environ.get("PUBLIC__BASE_URL", "").rstrip("/")
    callback_path = os.environ.get("PUBLIC__AUTH_CALLBACK_PATH", "")
    if public_base_url and callback_path:
        os.environ.setdefault("AUTH__GOOGLE_REDIRECT_URI", f"{public_base_url}{callback_path}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    args = _parser().parse_args(argv)
    _load_host_env(args.env_file)

    if args.command == "sync":
        _cmd_sync(
            adapters_dir=args.adapters_dir,
            vllm_url=args.vllm_url,
            aliases=args.aliases,
        )
        return

    if args.command == "list":
        _cmd_list(aliases=args.aliases)
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
