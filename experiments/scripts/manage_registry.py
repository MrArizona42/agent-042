"""CLI for managing LoRA adapters in MLflow Model Registry.

This script does NOT use Hydra — it is an operational tool, not an experiment.
The MLflow tracking URI is read from environment variables or the experiments
.env file, following the same convention as the training scripts.

Usage
-----
List all registered adapters::

    python experiments/scripts/manage_registry.py list

Show versions of a specific adapter::

    python experiments/scripts/manage_registry.py versions lora-summarization

Promote a version to production (alias "champion")::

    python experiments/scripts/manage_registry.py promote lora-summarization 3

Promote a version to staging (alias "challenger")::

    python experiments/scripts/manage_registry.py promote lora-summarization 5 --alias challenger

Download the production adapter locally::

    python experiments/scripts/manage_registry.py download lora-summarization ./adapters

Environment
-----------
Reads ``MLFLOW_BACKEND_URI`` (tracking URI) and S3 credentials from
``experiments/.env`` or from the shell environment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import dotenv

# ---------------------------------------------------------------------------
# Ensure imports work regardless of working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # …/agent-042

sys.path.insert(0, str(_SCRIPT_DIR))  # for train_adapter package

from train_adapter.registry import (  # noqa: E402
    ALIAS_PRODUCTION,
    AdapterRegistry,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────
def _load_env() -> None:
    """Load MLflow connection settings from experiments/.env."""
    env_file = _PROJECT_ROOT / "experiments" / ".env"
    if env_file.exists():
        dotenv.load_dotenv(env_file)
        logger.debug("Loaded env from %s", env_file)

    tracking_uri = os.getenv("MLFLOW_BACKEND_URI")
    if tracking_uri:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI: %s", tracking_uri)
    else:
        logger.warning(
            "MLFLOW_BACKEND_URI is not set. Using default tracking URI (local ./mlruns)."
        )


def _build_registry() -> AdapterRegistry:
    _load_env()
    return AdapterRegistry()


# ── sub-commands ─────────────────────────────────────────────────────────────
def cmd_list(_args: argparse.Namespace) -> None:
    registry = _build_registry()
    models = registry.list_models()
    if not models:
        print("No registered models found.")
        return

    for m in models:
        print(f"\n{'=' * 60}")
        print(f"  Model: {m['name']}")
        if m.get("description"):
            print(f"  Description: {m['description']}")
        if m.get("tags"):
            print(f"  Tags: {json.dumps(m['tags'], indent=4)}")
        for v in m.get("latest_versions", []):
            aliases = ", ".join(v.get("aliases", [])) or "—"
            print(
                f"    version {v['version']}  "
                f"status={v['status']}  "
                f"aliases=[{aliases}]  "
                f"run={v['run_id'][:8]}…"
            )
    print()


def cmd_versions(args: argparse.Namespace) -> None:
    registry = _build_registry()
    versions = registry.list_versions(args.model)
    if not versions:
        print(f"No versions found for '{args.model}'.")
        return

    print(f"\nVersions of '{args.model}'  (newest first):")
    print("-" * 70)
    for v in versions:
        aliases = ", ".join(v.aliases) or "—"
        desc = (v.description or "")[:50]
        print(f"  v{v.version:<4}  run={(v.run_id or '?')[:8]}…  aliases=[{aliases}]  {desc}")
    print()


def cmd_promote(args: argparse.Namespace) -> None:
    registry = _build_registry()
    alias = args.alias
    registry.promote(args.model, args.version, alias=alias)
    print(f"✓ '{args.model}' version {args.version} now has alias '{alias}'.")


def cmd_demote(args: argparse.Namespace) -> None:
    registry = _build_registry()
    alias = args.alias
    registry.demote(args.model, alias=alias)
    print(f"✓ Removed alias '{alias}' from '{args.model}'.")


def cmd_download(args: argparse.Namespace) -> None:
    registry = _build_registry()
    alias = args.alias
    path = registry.download_adapter(
        model_name=args.model,
        dst_dir=args.dst,
        alias=alias,
    )
    print(f"✓ Downloaded '{args.model}' ({alias}) → {path}")


def cmd_production(_args: argparse.Namespace) -> None:
    """Show all adapters currently tagged as production (champion)."""
    registry = _build_registry()
    adapters = registry.get_production_adapters()
    if not adapters:
        print("No adapters with 'champion' alias found.")
        return

    print("\nProduction adapters (alias=champion):")
    print("-" * 60)
    for name, a in adapters.items():
        print(f"  {name:<30}  v{a.version}  run={(a.run_id or '?')[:8]}…")
    print()


# ── CLI entry point ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage LoRA adapters in MLflow Model Registry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="List all registered adapter models.")

    # versions
    p_ver = sub.add_parser("versions", help="List all versions of a model.")
    p_ver.add_argument("model", help="Registered model name.")

    # promote
    p_pro = sub.add_parser("promote", help="Assign an alias to a model version.")
    p_pro.add_argument("model", help="Registered model name.")
    p_pro.add_argument("version", type=int, help="Version number.")
    p_pro.add_argument(
        "--alias",
        default=ALIAS_PRODUCTION,
        help=f"Alias to set (default: '{ALIAS_PRODUCTION}').",
    )

    # demote
    p_dem = sub.add_parser("demote", help="Remove an alias from a model.")
    p_dem.add_argument("model", help="Registered model name.")
    p_dem.add_argument(
        "--alias",
        default=ALIAS_PRODUCTION,
        help=f"Alias to remove (default: '{ALIAS_PRODUCTION}').",
    )

    # download
    p_dl = sub.add_parser("download", help="Download adapter artifacts from the registry.")
    p_dl.add_argument("model", help="Registered model name.")
    p_dl.add_argument(
        "dst",
        nargs="?",
        default=str(_PROJECT_ROOT / "assets" / "adapters"),
        help="Destination directory (default: assets/adapters/).",
    )
    p_dl.add_argument(
        "--alias",
        default=ALIAS_PRODUCTION,
        help=f"Alias to download (default: '{ALIAS_PRODUCTION}').",
    )

    # production
    sub.add_parser(
        "production",
        help="Show all adapters currently carrying the 'champion' alias.",
    )

    args = parser.parse_args()

    dispatch = {
        "list": cmd_list,
        "versions": cmd_versions,
        "promote": cmd_promote,
        "demote": cmd_demote,
        "download": cmd_download,
        "production": cmd_production,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
