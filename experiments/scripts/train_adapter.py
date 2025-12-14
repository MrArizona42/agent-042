import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import pytorch_lightning as pl
import torch
from datasets import load_from_disk
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)

import time
import mlflow
import dotenv


class ArxivDataModule(pl.LightningDataModule):
    """
    Minimal LightningDataModule:
    - Loads local HF dataset (train + validation)
    - Builds prompt+target token ids per example
    - Pads and masks in a simple collate_fn
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            local_path: str,
            max_seq_len: int,
            batch_size: int,
            shuffle: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.local_path = str(local_path)
        self.max_seq_len = int(max_seq_len)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.ds_train = None
        self.ds_val = None
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def setup(self, stage: Optional[str] = None) -> None:
        ds_dict = load_from_disk(self.local_path)
        self.ds_train = self._with_transform(ds_dict["train"])
        self.ds_val = self._with_transform(ds_dict["validation"])

    def _with_transform(self, ds):
        max_len = self.max_seq_len

        def transform(example: Dict[str, Any]) -> Dict[str, Any]:
            # Detect batched vs single example
            is_batched = isinstance(example.get("article"), list) or isinstance(example.get("abstract"), list)

            def build(prompt_text: str, target_text: str) -> Tuple[List[int], int]:
                prompt_enc = self.tokenizer(
                    prompt_text,
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=False,
                )
                target_enc = self.tokenizer(
                    target_text,
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=False,
                )
                prompt_ids = prompt_enc["input_ids"]
                target_ids = target_enc["input_ids"]
                input_ids = (prompt_ids + target_ids)[:max_len]
                prompt_len = min(len(prompt_ids), max_len)
                return input_ids, prompt_len

            if is_batched:
                arts = example.get("article") or []
                abss = example.get("abstract") or []
                n = max(len(arts), len(abss))
                # pad to equal length
                if len(arts) < n:
                    arts = arts + [""] * (n - len(arts))
                if len(abss) < n:
                    abss = abss + [""] * (n - len(abss))
                input_ids_list: List[List[int]] = []
                prompt_len_list: List[int] = []
                for art, abs_ in zip(arts, abss):
                    prompt = (
                        "Summarize the following article into an abstract:\n\n"
                        f"Article:\n{art}\n\nAbstract:\n"
                    )
                    ids, plen = build(prompt, abs_)
                    input_ids_list.append(ids)
                    prompt_len_list.append(plen)
                return {"input_ids": input_ids_list, "prompt_len": prompt_len_list}
            else:
                art = example.get("article") or ""
                abs_ = example.get("abstract") or ""
                prompt = (
                    "Summarize the following article into an abstract:\n\n"
                    f"Article:\n{art}\n\nAbstract:\n"
                )
                ids, plen = build(prompt, abs_)
                return {"input_ids": ids, "prompt_len": plen}

        ds.set_transform(transform)
        return ds

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        padded = self.tokenizer.pad(
            {"input_ids": [ex["input_ids"] for ex in batch]},
            padding=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        input_ids = padded["input_ids"]
        attention_mask = padded["attention_mask"]
        labels = input_ids.clone().masked_fill(attention_mask.eq(0), -100)
        for i, ex in enumerate(batch):
            upto = min(int(ex.get("prompt_len", 0)), labels.shape[1])
            if upto > 0:
                labels[i, :upto] = -100
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)


class PeftCausalLMModule(pl.LightningModule):
    """PEFT LoRA adapter training wrapper for causal LM with MLflow metric logging."""

    def __init__(self, model: Any, lr: float = 2e-4, weight_decay: float = 0.0):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])  # logs lr, weight_decay
        self._last_step_t: Optional[float] = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Any:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.ne(0)
        if labels is not None and labels.ndim == 1:
            labels = labels.unsqueeze(0)
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def on_train_start(self) -> None:
        # Initialize MLflow run, load env for creds, set tracking URI and experiment explicitly
        dotenv.load_dotenv()
        # Prefer explicit config value, fallback to env var
        tracking_uri = os.getenv("MLFLOW_BACKEND_URI") or getattr(self.trainer.logger, "_tracking_uri", None)
        if hasattr(self, "hparams") and hasattr(self.hparams, "mlflow_tracking_uri"):
            tracking_uri = self.hparams.mlflow_tracking_uri or tracking_uri
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        # Set experiment to keep runs grouped
        exp_name = None
        if hasattr(self, "hparams") and hasattr(self.hparams, "mlflow_experiment_name"):
            exp_name = self.hparams.mlflow_experiment_name
        if exp_name:
            mlflow.set_experiment(exp_name)

        if mlflow.active_run() is None:
            mlflow.start_run(run_name="train_adapter")
        self._last_step_t = time.perf_counter()

    def on_train_end(self) -> None:
        # Upload Hydra logs/configs as artifacts, then end MLflow run
        try:
            out_dir = HydraConfig.get().runtime.output_dir
            if out_dir:
                mlflow.log_artifacts(out_dir, artifact_path="hydra")
        except Exception as e:
            # Print once for visibility; continue without failing training
            print(f"MLflow artifact logging failed: {e}")

        if mlflow.active_run() is not None:
            mlflow.end_run()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        t0 = time.perf_counter()
        out = self.forward(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])  # has .loss
        loss = out.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log metrics to MLflow
        step = int(self.global_step)
        # train_loss
        mlflow.log_metric("train_loss", float(loss.detach().cpu().item()), step=step)
        # learning_rate from optimizer
        opt = self.trainer.optimizers[0] if self.trainer.optimizers else None
        if opt is not None and len(opt.param_groups) > 0:
            lr = float(opt.param_groups[0].get("lr", self.hparams.lr))
        else:
            lr = float(self.hparams.lr)
        mlflow.log_metric("learning_rate", lr, step=step)
        # tokens_per_second
        with torch.no_grad():
            mask = batch["attention_mask"].to(torch.float32)
            tokens = float(mask.sum().item())
        t1 = time.perf_counter()
        elapsed = max(1e-9, (t1 - (self._last_step_t or t0)))
        tokens_per_sec = tokens / elapsed
        mlflow.log_metric("tokens_per_second", tokens_per_sec, step=step)
        self._last_step_t = t1
        # gpu_memory_allocated_mb
        if torch.cuda.is_available():
            mem_mb = torch.cuda.memory_allocated(device=self.device) / 1e6
            mlflow.log_metric("gpu_memory_allocated_mb", float(mem_mb), step=step)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        val_out = self.forward(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])  # has .loss
        val_loss = val_out.loss
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        # Log val_loss with epoch-based step to avoid collision with training steps
        val_step = int(self.current_epoch)
        mlflow.log_metric("val_loss", float(val_loss.detach().cpu().item()), step=val_step)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)


def build_model_and_tokenizer(cfg: DictConfig) -> Tuple[Any, PreTrainedTokenizerBase]:
    src = cfg.experiment.model.local_path
    tokenizer = AutoTokenizer.from_pretrained(src, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_in_4bit = bool(cfg.experiment.model.get("load_in_4bit", False))
    dtype_str = cfg.experiment.model.get("dtype", None)
    torch_dtype = getattr(torch, dtype_str) if dtype_str else None

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=cfg.experiment.model.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_quant_type=cfg.experiment.model.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch_dtype or torch.bfloat16,
    ) if load_in_4bit else None

    # Ensure offload folder exists if provided
    offload_folder = cfg.experiment.model.get("offload_folder", None)
    if offload_folder:
        Path(offload_folder).mkdir(parents=True, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        src,
        device_map=cfg.experiment.model.device_map,
        dtype=torch_dtype,
        quantization_config=quant_cfg,
        local_files_only=True,
        offload_folder=offload_folder,
    )

    if bool(cfg.experiment.model.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=cfg.experiment.lora.r,
        lora_alpha=cfg.experiment.lora.lora_alpha,
        lora_dropout=cfg.experiment.lora.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.experiment.lora.target_modules),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("=== Training adapter with config ===")
    print(OmegaConf.to_yaml(cfg))
    print("#" * 50)
    try:
        print(f"Resolved Hydra output_dir: {HydraConfig.get().runtime.output_dir}")
    except Exception as e:
        print(f"Could not access Hydra output_dir: {e}")

    if cfg.experiment.seed is not None:
        pl.seed_everything(int(cfg.experiment.seed), workers=True)

    model, tokenizer = build_model_and_tokenizer(cfg)

    dm = ArxivDataModule(
        tokenizer=tokenizer,
        local_path=cfg.experiment.data.local_path,
        max_seq_len=cfg.experiment.data.max_seq_length,
        batch_size=cfg.experiment.data.batch_size,
        shuffle=True,
    )

    lit_module = PeftCausalLMModule(
        model=model,
        lr=cfg.experiment.training.lr,
        weight_decay=cfg.experiment.training.weight_decay,
    )

    # Inject MLflow tracking config into hparams for visibility in hooks
    setattr(lit_module.hparams, "mlflow_tracking_uri", cfg.paths.get("mlflow_tracking_uri", None))
    setattr(lit_module.hparams, "mlflow_experiment_name", cfg.experiment.get("mlflow", {}).get("experiment_name", None))

    # Direct Lightning logs to the project logs folder
    project_root = Path(cfg.paths.project_root)
    lightning_logs_dir = project_root / "experiments" / "logs" / "lighning_logs"

    trainer = pl.Trainer(
        max_epochs=cfg.experiment.trainer.max_epochs,
        devices=cfg.experiment.trainer.devices,
        accelerator=cfg.experiment.trainer.accelerator,
        precision=cfg.experiment.trainer.precision,
        gradient_clip_val=cfg.experiment.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.experiment.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.experiment.trainer.log_every_n_steps,
        val_check_interval=cfg.experiment.trainer.val_check_interval,
        num_sanity_val_steps=0,
        default_root_dir=str(lightning_logs_dir),
    )

    trainer.fit(lit_module, datamodule=dm)

    try:
        # Prefer explicit output path from config, fallback to Hydra output_dir
        cfg_save_dir = cfg.experiment.get("output", {}).get("save_dir", None)
        if cfg_save_dir is not None:
            save_dir = Path(cfg_save_dir)
        else:
            out_dir = Path(HydraConfig.get().runtime.output_dir)
            save_dir = out_dir / "adapter"
        save_dir.mkdir(parents=True, exist_ok=True)
        lit_module.model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Saved adapter and tokenizer to: {save_dir}")
    except Exception as e:
        print(f"Failed to save adapter: {e}")

    return {"status": "successful"}


if __name__ == "__main__":
    main()
