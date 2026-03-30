from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from .config import AppConfig
from .data_module import ArxivDataModule
from .lit_module import PeftCausalLMModule
from .mlflow_utils import log_hydra_artifacts_via_logger, setup_mlflow, teardown_mlflow
from .modeling import build_model_and_tokenizer

logger = logging.getLogger(__name__)


def run_training(cfg: AppConfig) -> Tuple[str, str, str]:
    project_root = Path(cfg.paths.project_root)
    if cfg.experiment.seed is not None:
        pl.seed_everything(cfg.experiment.seed, workers=True)

    # Create MLflow logger for Lightning
    mlf_logger = setup_mlflow(cfg)

    try:
        # Upload Hydra config as artifacts early
        log_hydra_artifacts_via_logger(mlf_logger)

        model, tokenizer = build_model_and_tokenizer(cfg)

        data_cfg = cfg.experiment.data
        dataset_path = Path(data_cfg.local_path)
        if not dataset_path.is_absolute():
            dataset_path = project_root / dataset_path
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        data_cfg.local_path = str(dataset_path)

        datamodule = ArxivDataModule(tokenizer=tokenizer, data_cfg=data_cfg, shuffle=True)

        scheduler_cfg = cfg.experiment.scheduler
        lightning_module = PeftCausalLMModule(
            model=model,
            lr=cfg.experiment.training.lr,
            weight_decay=cfg.experiment.training.weight_decay,
            scheduler_cfg=scheduler_cfg.__dict__ if scheduler_cfg else None,
        )

        artifacts_dir = project_root / "artifacts" / "training"
        run_tag = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        run_artifacts_dir = artifacts_dir / "runs" / run_tag
        run_artifacts_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=run_artifacts_dir / "checkpoints",
            filename="adapter-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True,
        )

        trainer_cfg = cfg.experiment.trainer
        trainer_kwargs: Dict[str, Any] = dict(
            max_epochs=trainer_cfg.max_epochs,
            devices=trainer_cfg.devices,
            accelerator=trainer_cfg.accelerator,
            gradient_clip_val=trainer_cfg.gradient_clip_val,
            accumulate_grad_batches=trainer_cfg.accumulate_grad_batches,
            log_every_n_steps=trainer_cfg.log_every_n_steps,
            val_check_interval=trainer_cfg.val_check_interval,
            num_sanity_val_steps=0,
            default_root_dir=str(run_artifacts_dir),
            callbacks=[checkpoint_callback],
            logger=mlf_logger,
        )
        # Pass precision as provided (string like "32-true") without type checker complaint
        trainer_kwargs["precision"] = trainer_cfg.precision
        trainer = pl.Trainer(**trainer_kwargs)

        trainer.fit(lightning_module, datamodule=datamodule)

        # Reload best checkpoint weights before export
        best_ckpt_path = checkpoint_callback.best_model_path
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
