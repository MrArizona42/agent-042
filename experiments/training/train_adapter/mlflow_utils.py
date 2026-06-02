from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import MLFlowLogger

from shared.local_env import load_local_env, resolve_local_env_path

from .config import AppConfig

logger = logging.getLogger(__name__)


def configure_mlflow_tracking(cfg: AppConfig) -> str:
    """Load MLflow environment settings and return the active tracking URI."""
    tracking_cfg = cfg.tracking
    project_root = Path(cfg.paths.project_root)

    env_path = tracking_cfg.env_path
    if env_path:
        env_file = resolve_local_env_path(env_path, repo_root=project_root)
        loaded_env = load_local_env(
            env_file,
            repo_root=project_root,
        )
        if loaded_env is None:
            logger.warning("MLflow env file missing: %s", env_file)
    else:
        loaded_env = load_local_env(
            repo_root=project_root,
        )
        if loaded_env is None:
            logger.info("No local env file loaded; using process environment only")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI: %s", tracking_uri)

    return mlflow.get_tracking_uri()


def setup_mlflow(cfg: AppConfig) -> MLFlowLogger:
    """Prepare environment and return a Lightning MLFlowLogger."""
    configure_mlflow_tracking(cfg)

    configured_run_name = cfg.logger.run_name
    configured_tags = _to_plain_dict(cfg.logger.tags)

    return instantiate(
        cfg.logger,
        tracking_uri=mlflow.get_tracking_uri(),
        run_name=configured_run_name or build_default_run_name(cfg),
        tags={**build_default_tags(cfg), **configured_tags},
    )


def build_default_run_name(cfg: AppConfig) -> str:
    """Build a non-static MLflow run name from the training config."""
    dataset_name = _dataset_name(cfg)
    lr = format(float(cfg.training.lr), ".0e")
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{cfg.task.run_name_prefix}-{dataset_name}-r{cfg.lora.r}-lr{lr}-{timestamp}"


def build_default_tags(cfg: AppConfig) -> dict[str, str]:
    """Return stable low-cardinality tags for every training run."""
    return {
        "pipeline": cfg.tracking.pipeline_name,
        "training.task": cfg.task.name,
        "training.dataset": _dataset_name(cfg),
        "training.base_model": _model_name(cfg),
        **dict(cfg.task.tags),
    }


def log_training_lineage(
    mlf_logger: MLFlowLogger,
    cfg: AppConfig,
    config_snapshot: DictConfig | dict[str, Any],
    *,
    dataset_path: Path,
    run_artifacts_dir: Path,
    trainable_param_count: int,
) -> dict[str, Any]:
    """Log resolved config and runtime lineage for later audit/debugging."""
    project_root = Path(cfg.paths.project_root)
    tracking_cfg = cfg.tracking
    if OmegaConf.is_config(config_snapshot):
        resolved_cfg = OmegaConf.to_container(config_snapshot, resolve=True)
    else:
        resolved_cfg = config_snapshot
    if not isinstance(resolved_cfg, dict):
        raise TypeError(
            f"Expected resolved config snapshot to be a mapping, got {type(resolved_cfg)!r}"
        )
    effective_batch_size = _effective_batch_size(cfg)
    dataset_dvc_hash = _find_dataset_dvc_hash(dataset_path, project_root)
    git_sha, git_dirty = _git_context(project_root)
    airflow_context = _airflow_context()
    hardware_info = _hardware_context()

    lineage = {
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "dataset_path": str(dataset_path),
        "dataset_dvc_hash": dataset_dvc_hash,
        "run_artifacts_dir": str(run_artifacts_dir),
        "effective_batch_size": effective_batch_size,
        "trainable_param_count": int(trainable_param_count),
        "airflow_context": airflow_context,
        "hardware": hardware_info,
    }

    metadata_dir = run_artifacts_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "resolved_config.json").write_text(
        json.dumps(resolved_cfg, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (metadata_dir / "lineage.json").write_text(
        json.dumps(lineage, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    client = mlf_logger.experiment
    run_id = mlf_logger.run_id

    tags = {
        "git.sha": git_sha,
        "git.dirty": str(git_dirty).lower(),
        "dataset.dvc_hash": dataset_dvc_hash or "",
        "run.artifacts_dir": str(run_artifacts_dir),
        "run.orchestrator": airflow_context.get("orchestrator", "cli"),
    }
    for key, value in airflow_context.items():
        if key == "orchestrator" or not value:
            continue
        tags[f"airflow.{key}"] = value
    for key, value in tags.items():
        if value:
            client.set_tag(run_id, key, value)

    if tracking_cfg.log_params:
        params = {
            "dataset_dvc_hash": dataset_dvc_hash or "",
            "effective_batch_size": effective_batch_size,
            "trainable_param_count": int(trainable_param_count),
            "model_path": str(cfg.model.local_path),
            "dataset_path": str(dataset_path),
            "git_sha": git_sha or "",
            "hardware.cuda_available": hardware_info["cuda_available"],
            "hardware.cuda_device_count": hardware_info["cuda_device_count"],
            "hardware.cuda_primary_name": hardware_info.get("cuda_primary_name", ""),
        }
        params.update({f"cfg.{k}": v for k, v in _flatten_mapping(resolved_cfg).items()})
        _log_params(client, run_id, params)

    if tracking_cfg.log_artifacts:
        try:
            client.log_artifacts(run_id, str(metadata_dir), artifact_path="metadata")
        except Exception as e:
            logger.warning("Failed to log metadata artifacts: %s", e)

    return lineage


def log_hydra_artifacts_via_logger(mlf_logger: MLFlowLogger) -> None:
    """Upload Hydra output directory to MLflow via the Lightning logger."""
    try:
        runtime = HydraConfig.get().runtime
        out_dir = Path(runtime.output_dir) if runtime and runtime.output_dir else None
        if out_dir and out_dir.exists():
            mlf_logger.experiment.log_artifacts(
                mlf_logger.run_id, str(out_dir), artifact_path="hydra"
            )
            logger.info("Uploaded Hydra artifacts from %s", out_dir)
    except Exception as e:
        logger.warning("Hydra artifact upload failed: %s", e)


def teardown_mlflow() -> None:
    """No-op: Lightning MLFlowLogger manages run lifecycle."""
    return None


def _effective_batch_size(cfg: AppConfig) -> int:
    devices = cfg.trainer.devices
    if isinstance(devices, int):
        num_devices = max(1, devices)
    elif isinstance(devices, (list, tuple)):
        num_devices = max(1, len(devices))
    else:
        num_devices = 1
    return (
        int(cfg.data.batch_size)
        * int(cfg.trainer.accumulate_grad_batches)
        * num_devices
    )


def _dataset_name(cfg: AppConfig) -> str:
    return cfg.dataset.name or Path(cfg.dataset.local_path).name


def _model_name(cfg: AppConfig) -> str:
    return cfg.model.name or Path(cfg.model.local_path).name


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        materialized = OmegaConf.to_container(value, resolve=True)
        return dict(materialized or {})
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"Expected MLflow logger tags to be a mapping, got {type(value)!r}")


def _flatten_mapping(data: Any, prefix: str = "") -> dict[str, str]:
    if isinstance(data, dict):
        flat: dict[str, str] = {}
        for key, value in data.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_mapping(value, child_prefix))
        return flat

    if isinstance(data, (list, tuple)):
        value = json.dumps(data)
    else:
        value = data

    value_str = str(value)
    if len(value_str) > 500:
        return {}
    return {prefix: value_str}


def _log_params(client: Any, run_id: str, params: dict[str, Any]) -> None:
    for key, value in params.items():
        if value in (None, ""):
            continue
        try:
            client.log_param(run_id, key, value)
        except Exception as e:
            logger.debug("Skipping MLflow param %s=%r: %s", key, value, e)


def _git_context(project_root: Path) -> tuple[str | None, bool]:
    git_cmd = ["git", "-c", f"safe.directory={project_root.as_posix()}"]

    try:
        sha = subprocess.check_output(
            [*git_cmd, "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
        ).strip()
    except Exception:
        sha = None

    try:
        dirty_output = subprocess.check_output(
            [*git_cmd, "status", "--porcelain", "--untracked-files=no"],
            cwd=project_root,
            text=True,
        )
        dirty = bool(dirty_output.strip())
    except Exception:
        dirty = False

    return sha, dirty


def _find_dataset_dvc_hash(dataset_path: Path, project_root: Path) -> str | None:
    candidates = [
        dataset_path.parent / f"{dataset_path.name}.dvc",
        project_root / "assets" / "datasets" / f"{dataset_path.name}.dvc",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            normalized = stripped.lstrip("-").strip()
            if normalized.startswith("md5:"):
                return normalized.split(":", 1)[1].strip()
    return None


def _airflow_context() -> dict[str, str]:
    dag_id = os.getenv("AIRFLOW_CTX_DAG_ID", "")
    return {
        "orchestrator": "airflow" if dag_id else "cli",
        "dag_id": dag_id,
        "task_id": os.getenv("AIRFLOW_CTX_TASK_ID", ""),
        "dag_run_id": os.getenv("AIRFLOW_CTX_DAG_RUN_ID", ""),
        "execution_date": os.getenv("AIRFLOW_CTX_EXECUTION_DATE", ""),
    }


def _hardware_context() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    device_names = [torch.cuda.get_device_name(i) for i in range(cuda_device_count)]
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_primary_name": device_names[0] if device_names else "",
        "cuda_device_names": device_names,
    }
