from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Any, Dict

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from .config import AppConfig
from .data_module import ArxivDataModule
from .lit_module import PeftCausalLMModule
from .mlflow_utils import setup_mlflow, teardown_mlflow, log_hydra_artifacts_via_logger
from .modeling import build_model_and_tokenizer

logger = logging.getLogger(__name__)


def run_training(cfg: AppConfig) -> Tuple[str, str]:
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
            mlflow_cfg=cfg.experiment.mlflow.__dict__,
            scheduler_cfg=scheduler_cfg.__dict__ if scheduler_cfg else None,
        )

        lightning_logs_dir = project_root / "experiments" / "logs" / "lightning_logs"
        lightning_logs_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=lightning_logs_dir / "checkpoints",
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
            default_root_dir=str(lightning_logs_dir),
            callbacks=[checkpoint_callback],
            logger=mlf_logger,
        )
        # Pass precision as provided (string like "32-true") without type checker complaint
        trainer_kwargs["precision"] = trainer_cfg.precision
        trainer = pl.Trainer(**trainer_kwargs)

        trainer.fit(lightning_module, datamodule=datamodule)

        save_dir = Path(cfg.experiment.output.save_dir)
        if not save_dir.is_absolute():
            save_dir = project_root / save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        lightning_module.model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        # Upload saved adapter/tokenizer as MLflow artifacts
        try:
            mlf_logger.experiment.log_artifacts(mlf_logger.run_id, str(save_dir), artifact_path="model")
        except Exception as e:
            logger.warning("Failed to log model artifacts: %s", e)

        return str(save_dir), str(lightning_logs_dir)
    finally:
        teardown_mlflow()
