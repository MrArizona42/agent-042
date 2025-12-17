from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
from pytorch_lightning import LightningModule


class PeftCausalLMModule(LightningModule):
    """Lightning wrapper for PEFT causal LM training with MLflow logging via Lightning logger."""

    def __init__(
        self,
        model: Any,
        lr: float,
        weight_decay: float,
        mlflow_cfg: Dict[str, Any],
        scheduler_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.save_hyperparameters({"lr": lr, "weight_decay": weight_decay})
        self.mlflow_cfg = mlflow_cfg
        self.scheduler_cfg = scheduler_cfg or {}
        self._last_step_t: Optional[float] = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None):
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if attention_mask.dtype not in (torch.int32, torch.int64, torch.long):
            attention_mask = attention_mask.long()
        if labels is not None and labels.ndim == 1:
            labels = labels.unsqueeze(0)
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def on_train_start(self) -> None:
        self._last_step_t = time.perf_counter()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        start = time.perf_counter()
        outputs = self.forward(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Additional metrics routed through Lightning's logger
        step = int(self.global_step)
        opt = self.trainer.optimizers[0] if self.trainer and self.trainer.optimizers else None
        lr = float(opt.param_groups[0]["lr"]) if opt and opt.param_groups else float(self.hparams.lr)
        self.log("learning_rate", lr, on_step=True, prog_bar=False, logger=True)

        tokens = float(batch["attention_mask"].to(torch.float32).sum().item())
        elapsed = max(1e-9, time.perf_counter() - (self._last_step_t or start))
        self.log("tokens_per_second", tokens / elapsed, on_step=True, prog_bar=False, logger=True)
        if torch.cuda.is_available():
            mem_mb = torch.cuda.memory_allocated(device=self.device) / 1e6
            self.log("gpu_memory_allocated_mb", float(mem_mb), on_step=True, prog_bar=False, logger=True)
        self._last_step_t = time.perf_counter()

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self.forward(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
        val_loss = outputs.loss
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if not self.scheduler_cfg or not self.scheduler_cfg.get("enabled", False):
            return optimizer

        sched_type = self.scheduler_cfg.get("type", "linear_warmup")
        if sched_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR

            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_cfg.get("T_max", 100),
                eta_min=self.scheduler_cfg.get("eta_min", 0.0),
            )
        else:
            from torch.optim.lr_scheduler import LinearLR

            scheduler = LinearLR(
                optimizer,
                start_factor=self.scheduler_cfg.get("start_factor", 1.0),
                total_iters=self.scheduler_cfg.get("warmup_steps", 100),
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.scheduler_cfg.get("interval", "step"),
                "frequency": self.scheduler_cfg.get("frequency", 1),
            },
        }
