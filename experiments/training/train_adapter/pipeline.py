from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .config import load_app_config
from .data_module import ArxivDataModule
from .lit_module import PeftCausalLMModule
from .mlflow_utils import log_hydra_artifacts_via_logger, setup_mlflow, teardown_mlflow
from .modeling import build_model_and_tokenizer

logger = logging.getLogger(__name__)


def run_training(cfg: DictConfig) -> Tuple[str, str, str]:
    """Run a full training loop.

    Accepts a raw Hydra ``DictConfig`` so that ``hydra.utils.instantiate``
    can build the trainer and callbacks from ``_target_`` entries.  Domain
    configs (model, data, lora …) are still accessed through the typed
    ``AppConfig`` produced by :func:`load_app_config`.
    """
    app_cfg = load_app_config(cfg)
    project_root = Path(app_cfg.paths.project_root)

    if app_cfg.experiment.seed is not None:
        pl.seed_everything(app_cfg.experiment.seed, workers=True)

    # Create MLflow logger for Lightning
    mlf_logger = setup_mlflow(app_cfg)

    try:
        # Upload Hydra config as artifacts early
        log_hydra_artifacts_via_logger(mlf_logger)

        model, tokenizer = build_model_and_tokenizer(app_cfg)

        data_cfg = app_cfg.experiment.data
        dataset_path = Path(data_cfg.local_path)
        if not dataset_path.is_absolute():
            dataset_path = project_root / dataset_path
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        data_cfg.local_path = str(dataset_path)

        datamodule = ArxivDataModule(tokenizer=tokenizer, data_cfg=data_cfg, shuffle=True)

        scheduler_cfg = app_cfg.experiment.scheduler
        lightning_module = PeftCausalLMModule(
            model=model,
            lr=app_cfg.experiment.training.lr,
            weight_decay=app_cfg.experiment.training.weight_decay,
            scheduler_cfg=scheduler_cfg.__dict__ if scheduler_cfg else None,
        )

        artifacts_dir = project_root / "artifacts" / "training"
        run_tag = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        run_artifacts_dir = artifacts_dir / "runs" / run_tag
        run_artifacts_dir.mkdir(parents=True, exist_ok=True)

        # ── Instantiate callbacks from config ──────────────────────────
        checkpoint_cb = instantiate(
            cfg.experiment.callbacks.checkpoint,
            dirpath=str(run_artifacts_dir / "checkpoints"),
        )
        callbacks = [checkpoint_cb]

        es_cfg = OmegaConf.select(cfg, "experiment.callbacks.early_stopping")
        if es_cfg is not None:
            callbacks.append(instantiate(es_cfg))

        # ── Instantiate trainer from config ────────────────────────────
        trainer = instantiate(
            cfg.experiment.trainer,
            callbacks=callbacks,
            logger=mlf_logger,
            default_root_dir=str(run_artifacts_dir),
        )

        trainer.fit(lightning_module, datamodule=datamodule)

        # Reload best checkpoint weights before export
        best_ckpt_path = checkpoint_cb.best_model_path
        if best_ckpt_path:
            logger.info("Reloading best checkpoint for export: %s", best_ckpt_path)
            best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            lightning_module.load_state_dict(best_ckpt["state_dict"])

        save_dir = run_artifacts_dir / "export"
        save_dir.mkdir(parents=True, exist_ok=True)
        lightning_module.model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        # Upload saved adapter/tokenizer as MLflow artifacts
        try:
            mlf_logger.experiment.log_artifacts(
                mlf_logger.run_id, str(save_dir), artifact_path="model"
            )
        except Exception as e:
            logger.warning("Failed to log model artifacts: %s", e)

        logger.info(
            "Training complete. Run ID: %s. Register via the "
            "lora_ops notebook or AdapterRegistry API.",
            mlf_logger.run_id,
        )

        return mlf_logger.run_id, str(save_dir), str(run_artifacts_dir)
    finally:
        teardown_mlflow()
