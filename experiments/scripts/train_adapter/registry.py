"""MLflow Model Registry utilities for LoRA adapter management.

Provides registration, promotion, listing, and download of LoRA adapters
using MLflow Model Registry with alias-based lifecycle management.

Aliases (lifecycle stages):
    - "champion"   : Current production adapter (served by vLLM).
    - "challenger"  : Candidate adapter under evaluation.

Naming convention for registered models:
    - "lora-<task>"  e.g. "lora-summarization", "lora-code", "lora-chat"
    - Matches the task names used by the TaskRouter in the inference service.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
from mlflow import artifacts as mlflow_artifacts
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# ── Alias constants ──────────────────────────────────────────────────────────
ALIAS_PRODUCTION = "champion"
ALIAS_STAGING = "challenger"


# ── Data transfer objects ────────────────────────────────────────────────────
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
        artifact_path: str = "model",
        model_name: str = "lora-default",
        tags: dict[str, str] | None = None,
        description: str | None = None,
    ) -> Any:
        """Register a trained LoRA adapter from an existing MLflow run.

        Creates (or appends a new version to) a registered model named
        *model_name*.  The adapter files are referenced via
        ``runs:/<run_id>/<artifact_path>``.

        Args:
            run_id: MLflow run that contains the adapter artifacts.
            artifact_path: Sub-path inside the run's artifact store
                (default ``"model"`` — the path used by ``pipeline.py``).
            model_name: Registered model name, e.g. ``"lora-summarization"``.
            tags: Optional key-value tags stored on the model *version*.
            description: Free-text description for the version.

        Returns:
            ``mlflow.entities.model_registry.ModelVersion``
        """
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
        alias: str = ALIAS_PRODUCTION,
    ) -> None:
        """Assign an alias to a specific model version.

        Typical usage::

            registry.promote("lora-summarization", version=3)
            # → alias "champion" now points to v3

        Args:
            model_name: Registered model name.
            version: Version number to promote.
            alias: Alias to assign (default ``"champion"``).
        """
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

    def demote(self, model_name: str, alias: str = ALIAS_PRODUCTION) -> None:
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

    def get_production_adapters(self) -> dict[str, RegisteredAdapter]:
        """Return every adapter that carries the *champion* alias.

        This is the entry point used by the inference service to discover
        which LoRA adapters should be loaded into vLLM.

        Returns:
            ``{model_name: RegisteredAdapter}`` for each model with a
            ``"champion"`` alias.
        """
        result: dict[str, RegisteredAdapter] = {}
        for rm in self.client.search_registered_models():
            try:
                mv = self.client.get_model_version_by_alias(rm.name, ALIAS_PRODUCTION)
                result[rm.name] = RegisteredAdapter(
                    name=mv.name,
                    version=int(mv.version),
                    run_id=mv.run_id,
                    source=mv.source,
                    aliases=[ALIAS_PRODUCTION],
                    tags=mv.tags or {},
                    description=mv.description,
                )
            except MlflowException:
                # Model exists but has no champion alias — skip.
                continue
        return result

    # ── Download ─────────────────────────────────────────────────────────
    def download_adapter(
        self,
        model_name: str,
        dst_dir: str | Path,
        alias: str = ALIAS_PRODUCTION,
    ) -> Path:
        """Download adapter artifacts for *model_name* at *alias*.

        Files are written to ``<dst_dir>/<model_name>/v<version>/``.

        Args:
            model_name: Registered model name.
            dst_dir: Root directory for downloaded adapters.
            alias: Which alias to resolve (default ``"champion"``).

        Returns:
            Path to the directory containing the adapter files.
        """
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
