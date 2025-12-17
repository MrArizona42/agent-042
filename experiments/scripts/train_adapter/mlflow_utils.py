from __future__ import annotations

import logging
import os
from pathlib import Path

import dotenv
import mlflow
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.loggers import MLFlowLogger

from .config import AppConfig

logger = logging.getLogger(__name__)


def setup_mlflow(cfg: AppConfig) -> MLFlowLogger:
    """Prepare environment and return a Lightning MLFlowLogger.

    - Loads .env with MLFLOW_* variables
    - Sets tracking URI if provided
    - Creates MLFlowLogger configured with experiment/run names
    """
    mlflow_cfg = cfg.experiment.mlflow
    project_root = Path(cfg.paths.project_root)

    # Load env
    env_path = mlflow_cfg.env_path
    if env_path:
        env_file = project_root / env_path if not Path(env_path).is_absolute() else Path(env_path)
        if env_file.exists():
            dotenv.load_dotenv(env_file)
            logger.info("Loaded MLflow env from %s", env_file)
        else:
            logger.warning("MLflow env file missing: %s", env_file)
    else:
        dotenv.load_dotenv(project_root / "experiments" / ".env")

    # Tracking URI
    tracking_uri = os.getenv("MLFLOW_BACKEND_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI: %s", tracking_uri)

    # Create Lightning MLflow logger
    mlf_logger = MLFlowLogger(
        experiment_name=mlflow_cfg.experiment_name or "default",
        run_name=mlflow_cfg.run_name or None,
        tracking_uri=mlflow.get_tracking_uri(),
        log_model=False,
        tags=None,
    )

    # Optionally log basic params at start via logger
    if mlflow_cfg.log_params:
        params = {
            "lr": cfg.experiment.training.lr,
            "weight_decay": cfg.experiment.training.weight_decay,
            "batch_size": cfg.experiment.data.batch_size,
            "max_seq_length": cfg.experiment.data.max_seq_length,
            "model_path": cfg.experiment.model.local_path,
            "load_in_4bit": cfg.experiment.model.load_in_4bit,
        }
        for k, v in params.items():
            mlf_logger.log_hyperparams({k: v})

    return mlf_logger


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
