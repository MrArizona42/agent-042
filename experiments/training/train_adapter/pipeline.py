from __future__ import annotations

import dataclasses
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from .config import load_app_config
from .lit_module import PeftCausalLMModule
from .mlflow_utils import (
    log_hydra_artifacts_via_logger,
    log_training_lineage,
    setup_mlflow,
    teardown_mlflow,
)
from .modeling import build_model_and_tokenizer


class TrainingResult(NamedTuple):
    run_id: str
    save_dir: str
    run_artifacts_dir: str
    summary_path: str


logger = logging.getLogger(__name__)


def _checkpoint_score_to_float(score: Any) -> float | None:
    if score is None:
        return None
    if isinstance(score, torch.Tensor):
        return float(score.detach().cpu().item())
    return float(score)


def _write_training_summary(
    run_id: str,
    save_dir: Path,
    run_artifacts_dir: Path,
    *,
    best_checkpoint_path: str | None,
    best_model_score: float | None,
    monitor_name: str | None,
) -> Path:
    run_artifacts_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = run_artifacts_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_id": run_id,
        "save_dir": str(save_dir),
        "run_artifacts_dir": str(run_artifacts_dir),
        "metadata_dir": str(metadata_dir),
        "best_checkpoint_path": best_checkpoint_path,
        "best_model_score": best_model_score,
        "monitor_name": monitor_name,
    }
    if monitor_name == "val_loss":
        summary["best_val_loss"] = best_model_score

    summary_path = run_artifacts_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary_path


def _restore_best_checkpoint_for_export(
    lightning_module: PeftCausalLMModule,
    checkpoint_path: str,
) -> None:
    """Restore the best checkpoint into the live module before adapter export.

    Lightning checkpoints created from 4-bit bitsandbytes models can contain
    serialized quantization metadata keys that do not appear on the already
    instantiated live module.  Export only needs the overlapping trainable
    weights, so filter the checkpoint down to keys that the current module
    actually exposes.
    """

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    live_keys = set(lightning_module.state_dict().keys())
    filtered_state_dict = {key: value for key, value in state_dict.items() if key in live_keys}
    skipped_keys = sorted(set(state_dict.keys()) - live_keys)

    if skipped_keys:
        logger.info(
            "Skipping %d non-restorable checkpoint entries during export restore; sample keys: %s",
            len(skipped_keys),
            skipped_keys[:5],
        )

    incompatible = lightning_module.load_state_dict(filtered_state_dict, strict=False)
    if incompatible.missing_keys:
        logger.warning(
            "Missing %d keys while restoring best checkpoint for export; sample keys: %s",
            len(incompatible.missing_keys),
            incompatible.missing_keys[:5],
        )
    if incompatible.unexpected_keys:
        logger.warning(
            "Unexpected %d keys while restoring best checkpoint for export; sample keys: %s",
            len(incompatible.unexpected_keys),
            incompatible.unexpected_keys[:5],
        )


def run_training(
    cfg: DictConfig,
) -> TrainingResult:
    """Run a full training loop.

    Accepts the raw Hydra ``DictConfig`` at the composition boundary, then
    converts it into a typed ``AppConfig`` for domain logic and factory
    instantiation.  The raw config is retained only to capture an exact
    resolved snapshot for lineage metadata.
    """
    app_cfg = load_app_config(cfg)
    project_root = Path(app_cfg.paths.project_root)

    if app_cfg.training.seed is not None:
        pl.seed_everything(app_cfg.training.seed, workers=True)

    # Create MLflow logger for Lightning
    mlf_logger = setup_mlflow(app_cfg)

    try:
        # Upload Hydra config as artifacts early
        if app_cfg.tracking.log_artifacts:
            log_hydra_artifacts_via_logger(mlf_logger)

        model, tokenizer = build_model_and_tokenizer(app_cfg)

        data_cfg = app_cfg.data
        dataset_path = Path(app_cfg.dataset.local_path)
        if not dataset_path.is_absolute():
            dataset_path = project_root / dataset_path
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        dataset_cfg = dataclasses.replace(app_cfg.dataset, local_path=str(dataset_path))

        datamodule = instantiate(
            app_cfg.data_module,
            tokenizer=tokenizer,
            task_cfg=app_cfg.task,
            dataset_cfg=dataset_cfg,
            data_cfg=data_cfg,
        )

        lightning_module = PeftCausalLMModule(
            model=model,
            lr=app_cfg.training.lr,
            weight_decay=app_cfg.training.weight_decay,
            scheduler_cfg=app_cfg.scheduler,
        )

        artifacts_dir = project_root / "artifacts" / "training"
        run_tag = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        run_artifacts_dir = artifacts_dir / "runs" / run_tag
        run_artifacts_dir.mkdir(parents=True, exist_ok=True)

        trainable_param_count = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        log_training_lineage(
            mlf_logger,
            app_cfg,
            cfg,
            dataset_path=dataset_path,
            run_artifacts_dir=run_artifacts_dir,
            trainable_param_count=trainable_param_count,
        )

        # ── Instantiate callbacks from typed factory config ────────────
        checkpoint_cb = instantiate(
            app_cfg.callbacks.checkpoint,
            dirpath=str(run_artifacts_dir / "checkpoints"),
        )
        callbacks = [checkpoint_cb]

        if app_cfg.callbacks.early_stopping is not None:
            callbacks.append(instantiate(app_cfg.callbacks.early_stopping))

        # ── Instantiate trainer from typed factory config ──────────────
        trainer = instantiate(
            app_cfg.trainer,
            callbacks=callbacks,
            logger=mlf_logger,
            default_root_dir=str(run_artifacts_dir),
        )

        trainer.fit(lightning_module, datamodule=datamodule)

        # Reload best checkpoint weights before export
        best_ckpt_path = checkpoint_cb.best_model_path or None
        best_model_score = _checkpoint_score_to_float(checkpoint_cb.best_model_score)
        if best_ckpt_path:
            logger.info("Reloading best checkpoint for export: %s", best_ckpt_path)
            _restore_best_checkpoint_for_export(lightning_module, best_ckpt_path)

        save_dir = run_artifacts_dir / "export"
        save_dir.mkdir(parents=True, exist_ok=True)
        lightning_module.model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        summary_path = _write_training_summary(
            mlf_logger.run_id,
            save_dir,
            run_artifacts_dir,
            best_checkpoint_path=best_ckpt_path,
            best_model_score=best_model_score,
            monitor_name=getattr(checkpoint_cb, "monitor", None),
        )

        # Upload saved adapter/tokenizer as MLflow artifacts
        if app_cfg.tracking.log_artifacts:
            try:
                mlf_logger.experiment.log_artifacts(
                    mlf_logger.run_id, str(save_dir), artifact_path="model"
                )
            except Exception as e:
                logger.warning("Failed to log model artifacts: %s", e)

        logger.info(
            "Training complete. Run ID: %s. Best %s=%s. Register via the "
            "lora_ops notebook or AdapterRegistry API.",
            mlf_logger.run_id,
            getattr(checkpoint_cb, "monitor", None) or "model_score",
            best_model_score if best_model_score is not None else "n/a",
        )

        return TrainingResult(
            run_id=mlf_logger.run_id,
            save_dir=str(save_dir),
            run_artifacts_dir=str(run_artifacts_dir),
            summary_path=str(summary_path),
        )
    finally:
        teardown_mlflow()
