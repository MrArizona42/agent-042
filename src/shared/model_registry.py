"""Inference-side adapter management: sync LoRA adapters from MLflow to vLLM.

This module connects to the MLflow Model Registry, discovers every adapter
that carries one of the configured sync aliases (default ``champion``,
``challenger``), downloads adapter files to a local directory, and hot-loads
them into a running vLLM instance via its ``/v1/load_lora_adapter`` API.

No manifest files are written; the vLLM runtime state is the live truth,
and MLflow Model Registry is the source of truth for alias → version mapping.

Disk layout
-----------
Adapters are stored by version (immutable, cacheable)::

    /adapters/{model_name}/v{version}/model/adapter_config.json

vLLM adapter naming
-------------------
Each (model, alias) pair is loaded into vLLM as ``{model}-{alias}``::

    lora-summarize-champion   →  /adapters/lora-summarize/v3/model
    lora-summarize-challenger →  /adapters/lora-summarize/v5/model

Typical flow
------------
1. Researcher promotes an adapter (operations notebook)::

       registry.promote("lora-summarize", version=3, alias="challenger")

2. Adapter-sync container syncs into the running vLLM::

       python -m shared.model_registry sync

   This downloads missing versions and hot-loads/unloads adapters via the
   vLLM API.  No restart required.

Environment variables
---------------------
``MLFLOW_TRACKING_URI``
    MLflow tracking server URL (e.g. ``http://mlflow:5000``).
``MLFLOW_S3_ENDPOINT_URL``, ``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``
    Credentials for downloading artifacts from S3.
``REGISTRY_ADAPTERS_DIR``
    Where to store downloaded adapters (default ``./adapters``).
``REGISTRY_SYNC_ALIASES``
    Comma-separated list of MLflow aliases to sync (default ``champion,challenger``).
``REGISTRY_VLLM_BASE_URL``
    vLLM server URL for hot-load API (default ``http://localhost:8000``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import requests
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
        """Register a trained LoRA adapter from an existing MLflow run.

        Uses ``MlflowClient.create_model_version`` directly so that adapters
        logged via ``log_artifacts`` (rather than ``log_model``) are supported.
        Newer MLflow versions (≥2.17) require a "logged model" record when
        using ``mlflow.register_model``; ``create_model_version`` bypasses that
        check and works with plain artifact uploads.
        """
        run = self.client.get_run(run_id)
        source = f"{run.info.artifact_uri}/{artifact_path}"

        tag_entities = (
            [mlflow.entities.ModelVersionTag(k, v) for k, v in tags.items()] if tags else None
        )

        self._ensure_registered_model(model_name)

        mv = self.client.create_model_version(
            name=model_name,
            source=source,
            run_id=run_id,
            tags=tag_entities,
            description=description,
        )

        logger.info(
            "Registered adapter '%s' version %s (run %s)",
            model_name,
            mv.version,
            run_id,
        )
        return mv

    def _ensure_registered_model(self, model_name: str) -> None:
        """Create the registered model entry if it does not already exist."""
        try:
            self.client.create_registered_model(model_name)
        except MlflowException:
            pass  # already exists

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
    """Syncs LoRA adapters from MLflow Model Registry into a running vLLM.

    For each registered model × each configured alias, the syncer:
    1. Queries MLflow for the version the alias points to.
    2. Downloads adapter files to ``{adapters_dir}/{model}/v{version}/``.
    3. Unloads stale adapters from vLLM.
    4. Loads desired adapters via the vLLM hot-load API.

    Args:
        tracking_uri: MLflow tracking URI. Falls back to ``MLFLOW_TRACKING_URI``
            from settings/env, then to the MLflow default.
        adapters_dir: Local root for downloaded adapter files.
        sync_aliases: List of MLflow aliases to iterate over.
        vllm_base_url: vLLM OpenAI-compatible server base URL.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        *,
        adapters_dir: str | Path,
        sync_aliases: list[str] | None = None,
        vllm_base_url: str | None = None,
    ):
        from shared.config import get_registry_settings

        cfg = get_registry_settings()

        if sync_aliases is None:
            sync_aliases = cfg.sync_aliases
        if vllm_base_url is None:
            vllm_base_url = cfg.vllm_base_url

        uri = tracking_uri or cfg.mlflow_tracking_uri
        if uri:
            mlflow.set_tracking_uri(uri)

        self.client = MlflowClient()
        self.adapters_dir = Path(adapters_dir)
        self.sync_aliases = sync_aliases
        self.vllm_base_url = vllm_base_url.rstrip("/")

    # ── MLflow discovery ─────────────────────────────────────────────────
    def discover_aliased_adapters(
        self,
    ) -> dict[tuple[str, str], Any]:
        """Return ``{(model_name, alias): ModelVersion}`` for every adapter with a sync alias.

        Returns an empty dict when no aliases match.

        Raises:
            RuntimeError: If MLflow is unreachable or returns an unexpected error.
        """
        if not self.sync_aliases:
            return {}

        try:
            registered_models = list(self.client.search_registered_models())
        except Exception as exc:
            raise RuntimeError(f"MLflow service unhealthy: {exc}") from exc

        result: dict[tuple[str, str], Any] = {}
        for rm in registered_models:
            for alias in self.sync_aliases:
                try:
                    mv = self.client.get_model_version_by_alias(rm.name, alias)
                    result[(rm.name, alias)] = mv
                except MlflowException as exc:
                    if (
                        "not found" in str(exc).lower()
                        or exc.error_code == "RESOURCE_DOES_NOT_EXIST"
                    ):
                        continue
                    raise RuntimeError(
                        f"MLflow error while querying alias '{alias}' for '{rm.name}': {exc}"
                    ) from exc
        return result

    # ── vLLM API helpers ─────────────────────────────────────────────────
    def _vllm_get_loaded_adapters(self) -> set[str]:
        """Query vLLM for currently loaded LoRA adapter names."""
        resp = requests.get(f"{self.vllm_base_url}/v1/models", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # vLLM returns {"data": [{"id": "base-model"}, {"id": "lora-name"}, ...]}
        # The base model is also listed; adapter names are those loaded via LoRA.
        return {m["id"] for m in data.get("data", [])}

    def _vllm_load_adapter(self, lora_name: str, lora_path: str) -> None:
        """Hot-load a LoRA adapter into running vLLM."""
        resp = requests.post(
            f"{self.vllm_base_url}/v1/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": lora_path},
            timeout=60,
        )
        resp.raise_for_status()
        logger.info("Loaded adapter '%s' from %s", lora_name, lora_path)

    def _vllm_unload_adapter(self, lora_name: str) -> None:
        """Unload a LoRA adapter from running vLLM."""
        resp = requests.post(
            f"{self.vllm_base_url}/v1/unload_lora_adapter",
            json={"lora_name": lora_name},
            timeout=60,
        )
        resp.raise_for_status()
        logger.info("Unloaded adapter '%s'", lora_name)

    # ── Main sync ────────────────────────────────────────────────────────
    @staticmethod
    def adapter_vllm_name(model_name: str, alias: str) -> str:
        """Build the vLLM adapter name from model name and alias."""
        return f"{model_name}-{alias}"

    def sync(self) -> list[AdapterInfo]:
        """Sync all aliased adapters from MLflow into the running vLLM.

        1. Query MLflow for desired (model, alias) → version mapping.
        2. Download any missing adapter versions to disk.
        3. Unload all currently loaded adapters that we manage.
        4. Load the desired adapters.

        Returns:
            List of :class:`AdapterInfo` for every loaded adapter.
        """
        # 1. Discover desired state from MLflow
        desired = self.discover_aliased_adapters()
        if not desired:
            logger.info("No aliased adapters found in MLflow — nothing to load.")

        # Build the set of adapter names we manage (model-alias pattern)
        managed_names = {self.adapter_vllm_name(model, alias) for model, alias in desired}

        # 2. Download missing versions
        for (model_name, alias), mv in desired.items():
            version = int(mv.version)
            adapter_dir = self.adapters_dir / model_name / f"v{version}"
            if adapter_dir.exists():
                logger.info(
                    "Adapter '%s' v%s already on disk — skipping download.",
                    model_name,
                    version,
                )
                continue

            adapter_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading '%s' v%s → %s …", model_name, version, adapter_dir)
            mlflow_artifacts.download_artifacts(
                artifact_uri=mv.source,
                dst_path=str(adapter_dir),
            )

        # 3. Unload stale adapters from vLLM
        loaded = self._vllm_get_loaded_adapters()
        # Also include managed names we're about to reload (unconditional reload)
        to_unload = loaded & (
            managed_names
            | {
                name
                for name in loaded
                if any(name == self.adapter_vllm_name(m, a) for m, a in desired)
            }
        )
        for name in to_unload:
            try:
                self._vllm_unload_adapter(name)
            except requests.HTTPError as exc:
                logger.warning("Failed to unload '%s': %s", name, exc)

        # 4. Load desired adapters
        infos: list[AdapterInfo] = []
        for (model_name, alias), mv in desired.items():
            version = int(mv.version)
            adapter_dir = self.adapters_dir / model_name / f"v{version}"

            # Find the actual model subdirectory
            # MLflow artifacts download into a subdirectory (usually 'model/')
            model_subdir = adapter_dir / "model"
            lora_path = str(model_subdir) if model_subdir.is_dir() else str(adapter_dir)

            vllm_name = self.adapter_vllm_name(model_name, alias)
            try:
                self._vllm_load_adapter(vllm_name, lora_path)
            except requests.HTTPError as exc:
                logger.error("Failed to load '%s' from %s: %s", vllm_name, lora_path, exc)
                continue

            infos.append(
                AdapterInfo(
                    name=vllm_name,
                    local_path=lora_path,
                    version=version,
                    run_id=mv.run_id or "",
                    tags=mv.tags or {},
                )
            )

        logger.info(
            "Sync complete: %d adapter(s) loaded, %d unloaded.",
            len(infos),
            len(to_unload),
        )
        return infos


# ── CLI entry point ──────────────────────────────────────────────────────────


def _load_env(env_file: str | None) -> None:
    """Load dotenv from *env_file* or fall back to .env."""
    import dotenv

    if env_file and Path(env_file).exists():
        dotenv.load_dotenv(env_file)
    elif Path(".env").exists():
        dotenv.load_dotenv(".env")


def _resolve_aliases(aliases: str | list | tuple | None):
    """Parse comma-separated aliases or fall back to config default."""
    from shared.config import get_registry_settings

    cfg = get_registry_settings()
    if aliases:
        if isinstance(aliases, (list, tuple)):
            return [a.strip() for a in aliases if a.strip()]
        return [a.strip() for a in aliases.split(",") if a.strip()]
    return cfg.sync_aliases


def _cmd_sync(
    adapters_dir: str | None = None,
    vllm_url: str | None = None,
    aliases: str | None = None,
    env_file: str | None = None,
) -> None:
    """Download and hot-load aliased adapters."""
    from shared.config import get_registry_settings

    cfg = get_registry_settings()
    _load_env(env_file)

    syncer = AdapterSyncer(
        adapters_dir=adapters_dir or cfg.adapters_dir,
        sync_aliases=_resolve_aliases(aliases),
        vllm_base_url=vllm_url or cfg.vllm_base_url,
    )
    infos = syncer.sync()
    if infos:
        print(f"\nSynced {len(infos)} adapter(s):")
        for info in infos:
            print(f"  {info.name} v{info.version} -> {info.local_path}")
    else:
        print("No adapters to sync.")


def _cmd_list(
    aliases: str | None = None,
    env_file: str | None = None,
) -> None:
    """List aliased adapters in MLflow (no download)."""
    from shared.config import get_registry_settings

    cfg = get_registry_settings()
    _load_env(env_file)

    syncer = AdapterSyncer(
        adapters_dir=cfg.adapters_dir,
        sync_aliases=_resolve_aliases(aliases),
        vllm_base_url=cfg.vllm_base_url,
    )
    adapters = syncer.discover_aliased_adapters()
    if adapters:
        print("\nAliased adapters:")
        for (name, alias), mv in adapters.items():
            vllm_name = AdapterSyncer.adapter_vllm_name(name, alias)
            print(f"  {vllm_name:<35} v{mv.version}  run={mv.run_id[:8]}")
    else:
        resolved = _resolve_aliases(aliases)
        print(f"No adapters with aliases {resolved} found.")


def _cli() -> None:
    """Minimal CLI: ``python -m shared.model_registry sync``."""
    import fire

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    fire.Fire(
        {
            "sync": _cmd_sync,
            "list": _cmd_list,
        }
    )


if __name__ == "__main__":
    _cli()
