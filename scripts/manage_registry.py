"""CLI for managing LoRA adapters in MLflow Model Registry.

This script does NOT use Hydra — it is an operational tool, not an experiment.
The MLflow tracking URI is read from environment variables or the experiments
.env file, following the same convention as the training scripts.

Usage
-----
Register an adapter from a completed MLflow run::

    python scripts/manage_registry.py register lora-summarize --run-id <RUN_ID>

List all registered adapters::

    python scripts/manage_registry.py list

Show versions of a specific adapter::

    python scripts/manage_registry.py versions lora-summarize

Promote a version to production (uses alias from ``REGISTRY_PRODUCTION_ALIAS``)::

    python scripts/manage_registry.py promote lora-summarize 3

Promote a version with a custom alias::

    python scripts/manage_registry.py promote lora-summarize 5 --alias challenger

Download the production adapter locally::

    python scripts/manage_registry.py download lora-summarize ./adapters

Sync adapters to a running vLLM instance::

    python scripts/manage_registry.py sync --vllm-url http://localhost:8000

Environment
-----------
Reads ``MLFLOW_BACKEND_URI`` (tracking URI) and S3 credentials from
``experiments/.env`` or from the shell environment.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import dotenv
import fire

# ---------------------------------------------------------------------------
# Ensure imports work regardless of working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent  # …/agent-042

sys.path.insert(0, str(_PROJECT_ROOT / "src"))  # for shared package

from shared.config import get_registry_settings  # noqa: E402
from shared.model_registry import AdapterRegistry, AdapterSyncer  # noqa: E402

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


def cmd_list() -> None:
    """List all registered adapter models."""
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


def cmd_versions(model: str) -> None:
    """List all versions of a model."""
    registry = _build_registry()
    versions = registry.list_versions(model)
    if not versions:
        print(f"No versions found for '{model}'.")
        return

    print(f"\nVersions of '{model}'  (newest first):")
    print("-" * 70)
    for v in versions:
        aliases = ", ".join(v.aliases) or "—"
        desc = (v.description or "")[:50]
        print(f"  v{v.version:<4}  run={(v.run_id or '?')[:8]}…  aliases=[{aliases}]  {desc}")
    print()


def cmd_promote(model: str, version: int, alias: str | None = None) -> None:
    """Assign an alias to a model version."""
    if alias is None:
        alias = get_registry_settings().production_alias
    registry = _build_registry()
    registry.promote(model, version, alias=alias)
    print(f"✓ '{model}' version {version} now has alias '{alias}'.")


def cmd_demote(model: str, alias: str | None = None) -> None:
    """Remove an alias from a model."""
    if alias is None:
        alias = get_registry_settings().production_alias
    registry = _build_registry()
    registry.demote(model, alias=alias)
    print(f"✓ Removed alias '{alias}' from '{model}'.")


def cmd_download(
    model: str,
    dst: str | None = None,
    alias: str | None = None,
) -> None:
    """Download adapter artifacts from the registry."""
    if dst is None:
        dst = str(_PROJECT_ROOT / "assets" / "adapters")
    if alias is None:
        alias = get_registry_settings().production_alias
    registry = _build_registry()
    path = registry.download_adapter(
        model_name=model,
        dst_dir=dst,
        alias=alias,
    )
    print(f"✓ Downloaded '{model}' ({alias}) → {path}")


def cmd_register(
    model: str,
    run_id: str,
    artifact_path: str = "model",
    tag: list[str] | None = None,
    description: str | None = None,
) -> None:
    """Register a trained adapter from an MLflow run into the Model Registry."""
    registry = _build_registry()
    tags = dict(t.split("=", 1) for t in tag) if tag else None
    mv = registry.register_adapter(
        run_id=run_id,
        artifact_path=artifact_path,
        model_name=model,
        tags=tags,
        description=description,
    )
    print(f"✓ Registered '{model}' version {mv.version} from run {run_id[:8]}…")


def cmd_production(alias: str | None = None) -> None:
    """Show all adapters currently carrying the production alias."""
    if alias is None:
        alias = get_registry_settings().production_alias
    registry = _build_registry()
    adapters = registry.get_production_adapters(alias=alias)
    if not adapters:
        print(f"No adapters with '{alias}' alias found.")
        return

    print(f"\nProduction adapters (alias={alias}):")
    print("-" * 60)
    for name, a in adapters.items():
        print(f"  {name:<30}  v{a.version}  run={(a.run_id or '?')[:8]}…")
    print()


def cmd_sync(
    adapters_dir: str | None = None,
    vllm_url: str | None = None,
    aliases: str | None = None,
) -> None:
    """Download aliased adapters from MLflow and hot-load them into vLLM."""
    _load_env()
    cfg = get_registry_settings()
    resolved_aliases = (
        [a.strip() for a in aliases.split(",") if a.strip()] if aliases else cfg.sync_aliases
    )
    syncer = AdapterSyncer(
        adapters_dir=adapters_dir or cfg.adapters_dir,
        sync_aliases=resolved_aliases,
        vllm_base_url=vllm_url or cfg.vllm_base_url,
    )
    infos = syncer.sync()
    if infos:
        print(f"\n✓ Synced {len(infos)} adapter(s):")
        for info in infos:
            print(f"  {info.name} v{info.version} → {info.local_path}")
    else:
        print("No adapters to sync.")


# ── CLI entry point ──────────────────────────────────────────────────────────
def main() -> None:
    fire.Fire(
        {
            "list": cmd_list,
            "register": cmd_register,
            "versions": cmd_versions,
            "promote": cmd_promote,
            "demote": cmd_demote,
            "download": cmd_download,
            "production": cmd_production,
            "sync": cmd_sync,
        }
    )


if __name__ == "__main__":
    main()
