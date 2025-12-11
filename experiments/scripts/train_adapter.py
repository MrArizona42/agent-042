import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import pytorch_lightning as pl
import torch
from datasets import load_from_disk
from datasets.utils.logging import (
    set_verbosity as ds_set_verbosity,
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
# Optional: make sure we don't accidentally reuse cached remote datasets
# Users can also override via HF_DATASETS_CACHE env var if needed
os.environ.setdefault("HF_DATASETS_CACHE", str(Path.home() / ".cache" / "huggingface" / "datasets"))

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
    Lightning DataModule that strictly loads pre-fetched local Hugging Face datasets from a path.

    Expects a dataset saved via `datasets.load_from_disk(path)` that contains 'train' and 'validation' splits
    with columns 'article' and 'abstract'. No network calls are made.
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            local_path: str,
            train_split: str,
            val_split: str,
            max_seq_len: int,
            batch_size: int,
            num_workers: int,
            shuffle: bool,
    ):
        super().__init__()
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.local_path = str(local_path)
        self.train_split = train_split
        self.val_split = val_split
        self.max_seq_len = int(max_seq_len)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.shuffle = bool(shuffle)
        self.ds_train = None
        self.ds_val = None
        # Cache pad id for collate
        self.pad_id: int = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

    def setup(self, stage: Optional[str] = None) -> None:
        # Validate local dataset path
        assert Path(self.local_path).exists(), f"Local dataset path not found: {self.local_path}"
        # Load dataset strictly from local disk
        ds_dict = load_from_disk(self.local_path)
        assert "train" in ds_dict and "validation" in ds_dict, (
            "Local dataset must contain 'train' and 'validation' splits"
        )
        ds_train = ds_dict["train"]
        ds_val = ds_dict["validation"]
        ds_train = _apply_simple_slice(ds_train, self.train_split)
        ds_val = _apply_simple_slice(ds_val, self.val_split)
        # Attach lazy transform
        self.ds_train = self._attach_transform(ds_train)
        self.ds_val = self._attach_transform(ds_val)

    def _attach_transform(self, ds):
        max_len = int(self.max_seq_len)
        pad_id = self.pad_id

        def transform(example: Dict[str, Any]) -> Dict[str, Any]:
            art = example["article"]
            abs_ = example["abstract"]
            prompt = (
                "Summarize the following article into an abstract:\n\n"
                f"Article:\n{art}\n\nAbstract:\n"
            )
            # Tokenize prompt and target
            prompt_ids = self.tokenizer.encode(prompt, truncation=True, max_length=max_len)
            target_ids = self.tokenizer.encode(abs_, truncation=True, max_length=max_len)
            input_ids = (prompt_ids + target_ids)[:max_len]
            labels_ids = ([-100] * len(prompt_ids) + target_ids)[:max_len]
            if len(input_ids) < max_len:
                pad_len = max_len - len(input_ids)
                input_ids = input_ids + [pad_id] * pad_len
                labels_ids = labels_ids + [-100] * pad_len
            return {"input_ids": input_ids, "labels": labels_ids}

        ds.set_transform(transform)
        return ds

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            return torch.tensor(x, dtype=torch.long)

        input_ids = torch.stack([to_tensor(x["input_ids"]) for x in batch])
        labels = torch.stack([to_tensor(x["labels"]) for x in batch])
        attention_mask = (input_ids != self.pad_id).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
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
    # Always load model/tokenizer from local path when use_local is true
    model_source = cfg.experiment.model.local_path if bool(
        cfg.experiment.model.use_local) else cfg.experiment.model.name_or_path
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


def _apply_simple_slice(ds, slice_spec: str):
    """Apply a very small subset parser for patterns like 'train[:1%]' or 'validation[:1000]'."""
    try:
        if not slice_spec or "[:]" in slice_spec:
            return ds
        start = slice_spec.find("[:")
        end = slice_spec.find("]", start)
        if start == -1 or end == -1:
            return ds
        token = slice_spec[start + 2: end]
        n = len(ds)
        if token.endswith("%"):
            pct = float(token[:-1])
            k = max(1, int(n * pct / 100.0))
            return ds.select(range(k))
        else:
            k = int(token)
            return ds.select(range(min(k, n)))
    except Exception:
        return ds


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
        train_split=cfg.experiment.data.train_split,
        val_split=cfg.experiment.data.val_split,
        max_seq_len=cfg.experiment.data.max_seq_length,
        batch_size=cfg.experiment.data.batch_size,
        num_workers=cfg.experiment.data.num_workers,
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
