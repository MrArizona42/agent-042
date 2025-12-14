import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import pytorch_lightning as pl
import torch
from datasets import load_from_disk
from datasets.utils.logging import (
    set_verbosity as ds_set_verbosity,
)
from datasets.utils.logging import (
    set_verbosity_info as ds_set_verbosity_info,
)
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
from transformers.utils import logging as tf_logging

# Hard-disable any HF Hub/network usage for datasets and prefer local files only
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("DISABLE_TELEMETRY", "1")
# Only use local cache directories if ever touched; avoid accidental remote downloads
os.environ.setdefault("HF_DATASETS_CACHE", str(Path.home() / ".cache" / "huggingface" / "datasets"))
os.environ.setdefault(
    "TRANSFORMERS_CACHE", str(Path.home() / ".cache" / "huggingface" / "transformers")
)

# Enable concise HF logging & progress bars
os.environ.setdefault("HF_HUB_ENABLE_PROGRESS_BARS", "1")
try:
    ds_set_verbosity_info()
except Exception:
    ds_set_verbosity(20)
try:
    tf_logging.set_verbosity_info()
except Exception:
    tf_logging.set_verbosity(tf_logging.INFO)

# Optimize matmul for Tensor Cores on consumer GPUs
try:
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass


class ArxivDataModule(pl.LightningDataModule):
    """
    Minimal, local-only LightningDataModule.

    - Loads local HF dataset (train + validation)
    - Prepares token ids per example with a simple prompt
    - Uses tokenizer.pad at collate time for clean padding & attention masks
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        local_path: str,
        max_seq_len: int,
        batch_size: int,
        shuffle: bool,
    ):
        super().__init__()
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.local_path = str(local_path)
        self.max_seq_len = int(max_seq_len)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.ds_train = None
        self.ds_val = None
        # Ensure pad token id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pad_id: int = int(self.tokenizer.pad_token_id)

    def setup(self, stage: Optional[str] = None) -> None:
        ds_dict = load_from_disk(self.local_path)

        self.ds_train = self._attach_transform(ds_dict["train"])
        self.ds_val = self._attach_transform(ds_dict["validation"])

    def _attach_transform(self, ds):
        max_len = int(self.max_seq_len)

        def as_text(x: Any) -> str:
            if x is None:
                return ""
            if isinstance(x, (str, bytes)):
                return x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else x
            return str(x)

        def transform(example: Dict[str, Any]) -> Dict[str, Any]:
            art = as_text(example.get("article"))
            abs_ = as_text(example.get("abstract"))
            prompt = (
                "Summarize the following article into an abstract:\n\n"
                f"Article:\n{art}\n\nAbstract:\n"
            )
            # Encode prompt and target separately (no padding yet)
            prompt_ids = self.tokenizer.encode(
                prompt, truncation=True, max_length=max_len, add_special_tokens=False
            )
            target_ids = self.tokenizer.encode(
                abs_, truncation=True, max_length=max_len, add_special_tokens=False
            )
            input_ids = (prompt_ids + target_ids)[:max_len]
            prompt_len = min(len(prompt_ids), max_len)
            # Defer padding to collate_fn via tokenizer.pad for clean attention_mask
            return {"input_ids": input_ids, "prompt_len": prompt_len}

        ds.set_transform(transform)
        return ds

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Use tokenizer.pad to build nicely padded input_ids and attention_mask
        features = {"input_ids": [ex["input_ids"] for ex in batch]}
        padded = self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        input_ids: torch.Tensor = padded["input_ids"]  # (B, T)
        attention_mask: torch.Tensor = padded["attention_mask"]  # (B, T)

        # Build labels: copy input_ids, mask out prompt and pad positions with -100
        labels: torch.Tensor = input_ids.clone()
        # Mask pads
        labels = labels.masked_fill(attention_mask.eq(0), -100)
        # Mask prompt tokens per sample
        for i, ex in enumerate(batch):
            p_len = int(ex.get("prompt_len", 0))
            if p_len > 0:
                upto = min(p_len, labels.shape[1])
                labels[i, :upto] = -100

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )


class PeftCausalLMModule(pl.LightningModule):
    """LightningModule wrapping a PEFT LoRA adapter on top of a base causal LM (Mistral-7B)."""

    def __init__(self, model: Any, lr: float = 1e-4, weight_decay: float = 0.0):
        super().__init__()
        self.model: Any = model
        self.save_hyperparameters(ignore=["model"])  # logs lr, weight_decay
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Any:
        # Normalize shapes and dtypes to avoid 1D attention_mask issues inside HF masking utils
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        # HF models expect boolean or float attention masks; prefer boolean
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.ne(0)
        if labels is not None and labels.ndim == 1:
            labels = labels.unsqueeze(0)
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.forward(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss  # type: ignore[attr-defined]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self.forward(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
        val_loss = outputs.loss  # type: ignore[attr-defined]
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


def build_model_and_tokenizer(cfg: DictConfig) -> Tuple[Any, PreTrainedTokenizerBase]:
    model_source = cfg.experiment.model.local_path

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_source, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_in_4bit = bool(cfg.experiment.model.get("load_in_4bit", False))
    dtype_str = cfg.experiment.model.get("dtype", None)
    torch_dtype = getattr(torch, dtype_str) if dtype_str else None

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=cfg.experiment.model.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=cfg.experiment.model.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=torch_dtype or torch.bfloat16,
        )

    model: Any = AutoModelForCausalLM.from_pretrained(
        model_source,
        device_map=cfg.experiment.model.device_map,
        dtype=torch_dtype,
        quantization_config=quantization_config,
        local_files_only=True,
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
        out_dir = HydraConfig.get().runtime.output_dir
        print(f"Resolved Hydra output_dir: {out_dir}")
    except Exception as e:
        print(f"Could not access Hydra output_dir: {e}")

    # Set random seed
    if cfg.experiment.seed is not None:
        print(f"Setting seed: {cfg.experiment.seed}")
        pl.seed_everything(int(cfg.experiment.seed), workers=True)

    # Build model and tokenizer
    model, tokenizer = build_model_and_tokenizer(cfg)

    # Data module: strictly local
    ds_path = cfg.experiment.data.local_path
    print(f"Loading dataset from local path: {ds_path}")
    dm = ArxivDataModule(
        tokenizer=tokenizer,
        local_path=ds_path,
        max_seq_len=cfg.experiment.data.max_seq_length,
        batch_size=cfg.experiment.data.batch_size,
        shuffle=True,
    )

    # Lightning module
    lit_module = PeftCausalLMModule(
        model=model,
        lr=cfg.experiment.training.lr,
        weight_decay=cfg.experiment.training.weight_decay,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.experiment.trainer.max_epochs,
        devices=cfg.experiment.trainer.devices,
        accelerator=cfg.experiment.trainer.accelerator,
        precision=cfg.experiment.trainer.precision,
        gradient_clip_val=cfg.experiment.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.experiment.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.experiment.trainer.log_every_n_steps,
        val_check_interval=cfg.experiment.trainer.val_check_interval,
        num_sanity_val_steps=0,  # disable sanity check to reduce initial memory spikes
    )

    # Fit
    trainer.fit(lit_module, datamodule=dm)

    # Save adapter
    try:
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
