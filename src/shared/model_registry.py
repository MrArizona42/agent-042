"""Inference-side adapter management: sync production LoRA adapters from MLflow.

This module connects to the MLflow Model Registry, discovers every adapter
that carries the configured production alias (default ``"champion"``), and
downloads it to a local directory so that vLLM can load the adapters at
startup (``--enable-lora``).

It also generates a ``lora-modules.json`` manifest that can be fed to vLLM's
``--lora-modules`` flag or consumed by the gateway to know which adapters are
available.

The production alias is configurable via the ``REGISTRY_PRODUCTION_ALIAS``
environment variable (see :class:`shared.config.ModelRegistrySettings`).

Typical flow
------------
1. Researcher promotes an adapter::

       python scripts/manage_registry.py promote lora-summarize 3

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


# ── Data objects ─────────────────────────────────────────────────────────────
@dataclass
class RegisteredAdapter:
    """Read-only snapshot of a registered LoRA adapter version."""

    name: str
    version: int
    run_id: str | None
    source: str | None
    aliases: list[str] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    description: str | None = None


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


# ── Registry facade ─────────────────────────────────────────────────────────
class AdapterRegistry:
    """High-level interface to MLflow Model Registry for LoRA adapters.

    Wraps :class:`mlflow.tracking.MlflowClient` and provides task-oriented
    methods for the full adapter lifecycle:

        train  →  register  →  promote("champion")  →  download  →  serve
    """

    def __init__(self, tracking_uri: str | None = None):
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    # ── Register ─────────────────────────────────────────────────────────
    def register_adapter(
        self,
        run_id: str,
        artifact_path: str,
        model_name: str,
        tags: dict[str, str] | None = None,
        description: str | None = None,
    ) -> Any:
        """Register a trained LoRA adapter from an existing MLflow run."""
        model_uri = f"runs:/{run_id}/{artifact_path}"

        mv = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags=tags,
        )

        if description:
            self.client.update_model_version(
                name=model_name,
                version=mv.version,
                description=description,
            )

        logger.info(
            "Registered adapter '%s' version %s (run %s)",
            model_name,
            mv.version,
            run_id,
        )
        return mv

    # ── Promote / demote ─────────────────────────────────────────────────
    def promote(
        self,
        model_name: str,
        version: int,
        alias: str,
    ) -> None:
        """Assign an alias to a specific model version."""
        self.client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=str(version),
        )
        logger.info(
            "Set alias '%s' on '%s' version %s",
            alias,
            model_name,
            version,
        )

    def demote(self, model_name: str, alias: str) -> None:
        """Remove an alias from a registered model."""
        try:
            self.client.delete_registered_model_alias(
                name=model_name,
                alias=alias,
            )
            logger.info("Removed alias '%s' from '%s'", alias, model_name)
        except MlflowException as e:
            logger.warning("Could not remove alias '%s' from '%s': %s", alias, model_name, e)

    # ── Query ────────────────────────────────────────────────────────────
    def list_models(self) -> list[dict[str, Any]]:
        """Return metadata for every registered model."""
        results = []
        for rm in self.client.search_registered_models():
            results.append(
                {
                    "name": rm.name,
                    "description": rm.description,
                    "tags": rm.tags,
                    "latest_versions": [
                        {
                            "version": v.version,
                            "run_id": v.run_id,
                            "status": v.status,
                            "aliases": getattr(v, "aliases", []),
                        }
                        for v in (rm.latest_versions or [])
                    ],
                }
            )
        return results

    def list_versions(self, model_name: str) -> list[RegisteredAdapter]:
        """List every version of *model_name*, newest first."""
        versions = self.client.search_model_versions(f"name='{model_name}'")
        adapters = [
            RegisteredAdapter(
                name=v.name,
                version=int(v.version),
                run_id=v.run_id,
                source=v.source,
                aliases=getattr(v, "aliases", []),
                tags=v.tags or {},
                description=v.description,
            )
            for v in versions
        ]
        adapters.sort(key=lambda a: a.version, reverse=True)
        return adapters

    def get_production_adapters(
        self,
        alias: str | None = None,
    ) -> dict[str, RegisteredAdapter]:
        """Return every adapter that carries the given *alias*.

        Args:
            alias: MLflow alias to look up.  When *None*, reads the default
                from ``ModelRegistrySettings.production_alias``.  If the
                config value is also *None*, returns an empty dict (no
                production adapters).
        """
        if alias is None:
            from shared.config import get_registry_settings

            alias = get_registry_settings().production_alias
        if not alias:
            return {}
        try:
            registered_models = list(self.client.search_registered_models())
        except Exception as exc:
            raise RuntimeError(f"MLflow service unhealthy: {exc}") from exc

        result: dict[str, RegisteredAdapter] = {}
        for rm in registered_models:
            try:
                mv = self.client.get_model_version_by_alias(rm.name, alias)
                result[rm.name] = RegisteredAdapter(
                    name=mv.name,
                    version=int(mv.version),
                    run_id=mv.run_id,
                    source=mv.source,
                    aliases=[alias],
                    tags=mv.tags or {},
                    description=mv.description,
                )
            except MlflowException as exc:
                if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
                    continue
                raise RuntimeError(
                    f"MLflow error while querying alias for '{rm.name}': {exc}"
                ) from exc
        return result

    # ── Download ─────────────────────────────────────────────────────────
    def download_adapter(
        self,
        model_name: str,
        dst_dir: str | Path,
        alias: str,
    ) -> Path:
        """Download adapter artifacts for *model_name* at *alias*."""
        dst_dir = Path(dst_dir)
        mv = self.client.get_model_version_by_alias(model_name, alias)

        adapter_dir = dst_dir / model_name / f"v{mv.version}"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        local_path = mlflow_artifacts.download_artifacts(
            artifact_uri=mv.source,
            dst_path=str(adapter_dir),
        )

        logger.info(
            "Downloaded '%s' v%s (%s) → %s",
            model_name,
            mv.version,
            alias,
            local_path,
        )
        return Path(local_path)


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
        *,
        adapters_dir: str | Path,
        production_alias: str | None = None,
    ):
        # Resolve from config when caller doesn't specify explicitly.
        if production_alias is None:
            from shared.config import get_registry_settings

            production_alias = get_registry_settings().production_alias
        uri = tracking_uri or os.getenv("MLFLOW_BACKEND_URI")
        if uri:
            mlflow.set_tracking_uri(uri)
        self.client = MlflowClient()
        self.adapters_dir = Path(adapters_dir)
        self.production_alias = production_alias

    def discover_production_adapters(self) -> dict[str, Any]:
        """Return ``{model_name: ModelVersion}`` for every adapter with the production alias.

        Returns an empty dict when ``production_alias`` is *None* (no alias configured).

        Raises:
            RuntimeError: If MLflow is unreachable or returns an unexpected error.
        """
        if not self.production_alias:
            return {}
        try:
            registered_models = list(self.client.search_registered_models())
        except Exception as exc:
            raise RuntimeError(f"MLflow service unhealthy: {exc}") from exc

        result = {}
        for rm in registered_models:
            try:
                mv = self.client.get_model_version_by_alias(
                    rm.name,
                    self.production_alias,
                )
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
            logger.warning(
                "No production adapters to sync (alias=%r).",
                self.production_alias,
            )
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

    from shared.config import get_registry_settings

    registry_cfg = get_registry_settings()

    parser = argparse.ArgumentParser(
        description="Sync production LoRA adapters from MLflow Model Registry."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sync = sub.add_parser("sync", help="Download production adapters.")
    p_sync.add_argument(
        "--adapters-dir",
        default=registry_cfg.adapters_dir,
        help="Destination directory for adapter files.",
    )
    p_sync.add_argument(
        "--base-model",
        default=None,
        help="Base model name for vLLM manifest (optional).",
    )
    p_sync.add_argument(
        "--production-alias",
        default=registry_cfg.production_alias,
        help="MLflow alias that marks a production adapter "
        f"(from config: {registry_cfg.production_alias!r}).",
    )
    p_sync.add_argument(
        "--env-file",
        default=None,
        help="Path to .env file with MLflow/S3 credentials.",
    )

    p_list = sub.add_parser("list", help="List production adapters (no download).")
    p_list.add_argument(
        "--production-alias",
        default=registry_cfg.production_alias,
        help=f"MLflow alias to look up (from config: {registry_cfg.production_alias!r}).",
    )
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
        syncer = AdapterSyncer(
            adapters_dir=args.adapters_dir,
            production_alias=args.production_alias,
        )
        infos = syncer.sync(base_model_name=args.base_model)
        if infos:
            print(f"\n✓ Synced {len(infos)} adapter(s) (alias='{args.production_alias}'):")
            for info in infos:
                print(f"  {info.name} v{info.version} → {info.local_path}")
        else:
            print(f"No adapters with alias '{args.production_alias}' to sync.")

    elif args.command == "list":
        syncer = AdapterSyncer(
            adapters_dir=registry_cfg.adapters_dir,
            production_alias=args.production_alias,
        )
        adapters = syncer.discover_production_adapters()
        if adapters:
            print(f"\nProduction adapters (alias='{args.production_alias}'):")
            for name, mv in adapters.items():
                print(f"  {name:<30} v{mv.version}  run={mv.run_id[:8]}…")
        else:
            print(f"No adapters with alias '{args.production_alias}' found.")


if __name__ == "__main__":
    _cli()
