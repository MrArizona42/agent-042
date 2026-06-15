#!/usr/bin/env python3
"""Host-side wrapper for MLflow/vLLM adapter registry operations."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shared.local_env import load_local_env  # noqa: E402
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


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    args = _parser().parse_args(argv)
    load_local_env(args.env_file, override=False)

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
