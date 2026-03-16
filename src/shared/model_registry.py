"""Inference-side adapter management: sync production LoRA adapters from MLflow.

This module connects to the MLflow Model Registry, discovers every adapter
that carries the ``"champion"`` alias, and downloads it to a local directory
so that vLLM can load the adapters at startup (``--enable-lora``).

It also generates a ``lora-modules.json`` manifest that can be fed to vLLM's
``--lora-modules`` flag or consumed by the gateway to know which adapters are
available.

Typical flow
------------
1. Researcher promotes an adapter::

       python scripts/manage_registry.py promote lora-summarization 3

2. Ops (or CI/CD) syncs adapters to the inference host::

       python -m shared.model_registry sync

   This writes files to ``REGISTRY_ADAPTERS_DIR`` (default ``./adapters``)
   and produces ``lora-modules.json``.

3. vLLM is (re)started with the adapter config::

       --enable-lora --lora-modules @/adapters/lora-modules.json

Environment variables
---------------------
``MLFLOW_BACKEND_URI``
    MLflow tracking server URL (e.g. ``http://mlflow:5000``).
``MLFLOW_S3_ENDPOINT_URL``, ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``
    Credentials for downloading artifacts from S3.
``REGISTRY_ADAPTERS_DIR``
    Where to store downloaded adapters (default ``./adapters``).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mlflow
from mlflow import artifacts as mlflow_artifacts
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

ALIAS_PRODUCTION = "champion"


# ── Data objects ─────────────────────────────────────────────────────────────
@dataclass
class AdapterInfo:
    """Describes one LoRA adapter ready for vLLM."""

    name: str
    local_path: str
    version: int
    run_id: str
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class VllmLoraModule:
    """Single entry for the vLLM ``--lora-modules`` JSON array."""

    name: str
    path: str
    base_model_name: str | None = None


# ── Sync logic ───────────────────────────────────────────────────────────────
class AdapterSyncer:
    """Downloads production adapters from MLflow Model Registry.

    Args:
        tracking_uri: MLflow tracking URI.  Falls back to ``MLFLOW_BACKEND_URI``
            env var, then to the MLflow default.
        adapters_dir: Local root for downloaded adapter files.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        adapters_dir: str | Path = "./adapters",
    ):
        uri = tracking_uri or os.getenv("MLFLOW_BACKEND_URI")
        if uri:
            mlflow.set_tracking_uri(uri)
        self.client = MlflowClient()
        self.adapters_dir = Path(adapters_dir)

    def discover_production_adapters(self) -> dict[str, Any]:
        """Return ``{model_name: ModelVersion}`` for every champion adapter.

        Raises:
            RuntimeError: If MLflow is unreachable or returns an unexpected error.
        """
        try:
            registered_models = list(self.client.search_registered_models())
        except Exception as exc:
            raise RuntimeError(f"MLflow service unhealthy: {exc}") from exc

        result = {}
        for rm in registered_models:
            try:
                mv = self.client.get_model_version_by_alias(rm.name, ALIAS_PRODUCTION)
                result[rm.name] = mv
            except MlflowException as exc:
                if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
                    continue
                raise RuntimeError(
                    f"MLflow error while querying alias for '{rm.name}': {exc}"
                ) from exc
        return result

    def sync(self, base_model_name: str | None = None) -> list[AdapterInfo]:
        """Download all production adapters and write the vLLM manifest.

        Args:
            base_model_name: Optional base model name written into the vLLM
                manifest (used by vLLM to validate LoRA compatibility).

        Returns:
            List of :class:`AdapterInfo` for every downloaded adapter.
        """
        adapters_map = self.discover_production_adapters()
        if not adapters_map:
            logger.warning("No adapters with '%s' alias found.", ALIAS_PRODUCTION)
            # Write an empty manifest so vLLM can start without LoRA modules.
            manifest_path = self.adapters_dir / "lora-modules.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text("[]")
            logger.info("Wrote empty vLLM manifest: %s", manifest_path)
            return []

        infos: list[AdapterInfo] = []
        vllm_modules: list[VllmLoraModule] = []

        for model_name, mv in adapters_map.items():
            version = int(mv.version)
            adapter_dir = self.adapters_dir / model_name / f"v{version}"
            adapter_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Downloading '%s' v%s → %s …",
                model_name,
                version,
                adapter_dir,
            )

            local_path = mlflow_artifacts.download_artifacts(
                artifact_uri=mv.source,
                dst_path=str(adapter_dir),
            )

            info = AdapterInfo(
                name=model_name,
                local_path=str(Path(local_path).resolve()),
                version=version,
                run_id=mv.run_id,
                tags=mv.tags or {},
            )
            infos.append(info)

            vllm_modules.append(
                VllmLoraModule(
                    name=model_name,
                    path=info.local_path,
                    base_model_name=base_model_name,
                )
            )

        # Write the vLLM manifest
        manifest_path = self.adapters_dir / "lora-modules.json"
        manifest_data = [asdict(m) for m in vllm_modules]
        manifest_path.write_text(json.dumps(manifest_data, indent=2))
        logger.info("Wrote vLLM manifest: %s (%d adapters)", manifest_path, len(infos))

        # Write a human-readable summary
        summary_path = self.adapters_dir / "adapters-summary.json"
        summary_data = [asdict(a) for a in infos]
        summary_path.write_text(json.dumps(summary_data, indent=2))

        return infos


# ── CLI entry point ──────────────────────────────────────────────────────────
def _cli() -> None:
    """Minimal CLI: ``python -m shared.model_registry sync``."""
    import argparse

    import dotenv

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Sync production LoRA adapters from MLflow Model Registry."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sync = sub.add_parser("sync", help="Download all champion adapters.")
    p_sync.add_argument(
        "--adapters-dir",
        default=os.getenv("REGISTRY_ADAPTERS_DIR", "./adapters"),
        help="Destination directory for adapter files.",
    )
    p_sync.add_argument(
        "--base-model",
        default=None,
        help="Base model name for vLLM manifest (optional).",
    )
    p_sync.add_argument(
        "--env-file",
        default=None,
        help="Path to .env file with MLflow/S3 credentials.",
    )

    p_list = sub.add_parser("list", help="List production adapters (no download).")
    p_list.add_argument(
        "--env-file",
        default=None,
        help="Path to .env file with MLflow/S3 credentials.",
    )

    args = parser.parse_args()

    # Load env file
    env_file = args.env_file
    if env_file and Path(env_file).exists():
        dotenv.load_dotenv(env_file)
    elif Path(".env").exists():
        dotenv.load_dotenv(".env")

    if args.command == "sync":
        syncer = AdapterSyncer(adapters_dir=args.adapters_dir)
        infos = syncer.sync(base_model_name=args.base_model)
        if infos:
            print(f"\n✓ Synced {len(infos)} adapter(s):")
            for info in infos:
                print(f"  {info.name} v{info.version} → {info.local_path}")
        else:
            print("No production adapters to sync.")

    elif args.command == "list":
        syncer = AdapterSyncer()
        adapters = syncer.discover_production_adapters()
        if adapters:
            print("\nProduction adapters (champion):")
            for name, mv in adapters.items():
                print(f"  {name:<30} v{mv.version}  run={mv.run_id[:8]}…")
        else:
            print("No production adapters found.")


if __name__ == "__main__":
    _cli()
