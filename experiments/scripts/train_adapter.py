import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import pytorch_lightning as pl
import torch
from datasets import load_dataset, load_from_disk
from datasets.utils.logging import set_verbosity as ds_set_verbosity
from datasets.utils.logging import set_verbosity_info as ds_set_verbosity_info
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import logging as tf_logging

# Enable detailed HF logging & progress bars
os.environ.setdefault("HF_HUB_ENABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "0")
try:
    ds_set_verbosity_info()
except Exception:
    ds_set_verbosity("INFO")
try:
    tf_logging.set_verbosity_info()
except Exception:
    tf_logging.set_verbosity(tf_logging.INFO)


class ArxivDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for the Hugging Face 'scientific_papers' dataset (config 'arxiv').

    It constructs causal LM inputs to learn to generate the abstract conditioned on the article.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        dataset_name: str = "scientific_papers",
        dataset_config: str = "arxiv",
        train_split: str = "train",
        val_split: str = "validation",
        max_seq_len: int = 2048,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.train_split = train_split
        self.val_split = val_split
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.ds_train = None
        self.ds_val = None

    def prepare_data(self) -> None:
        # Downloads the dataset if not present.
        load_dataset(self.dataset_name, self.dataset_config, split=self.train_split)
        load_dataset(self.dataset_name, self.dataset_config, split=self.val_split)

    def setup(self, stage: Optional[str] = None) -> None:
        self.ds_train = load_dataset(self.dataset_name, self.dataset_config, split=self.train_split)
        self.ds_val = load_dataset(self.dataset_name, self.dataset_config, split=self.val_split)

        def preprocess(batch: Dict[str, List[str]]) -> Dict[str, Any]:
            # Each batch contains lists of 'article' and 'abstract'
            articles = batch["article"]
            abstracts = batch["abstract"]
            inputs: List[List[int]] = []
            labels: List[List[int]] = []
            for art, abs_ in zip(articles, abstracts):
                prompt = (
                    "Summarize the following article into an abstract:\n\n"
                    f"Article:\n{art}\n\nAbstract:\n"
                )
                # Tokenize
                prompt_ids = self.tokenizer(
                    prompt, truncation=True, max_length=self.max_seq_len
                ).input_ids
                target_ids = self.tokenizer(
                    abs_, truncation=True, max_length=self.max_seq_len
                ).input_ids
                # Concatenate, limit to max_seq_len
                input_ids = (prompt_ids + target_ids)[: self.max_seq_len]
                # Labels: -100 for prompt tokens, predict only abstract tokens
                labels_ids = [-100] * len(prompt_ids) + target_ids
                labels_ids = labels_ids[: self.max_seq_len]
                # Pad input and labels to same length using tokenizer pad token
                pad_id = (
                    self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                )
                if len(input_ids) < self.max_seq_len:
                    pad_len = self.max_seq_len - len(input_ids)
                    input_ids = input_ids + [pad_id] * pad_len
                    labels_ids = labels_ids + [-100] * pad_len
                inputs.append(input_ids)
                labels.append(labels_ids)
            return {"input_ids": inputs, "labels": labels}

        self.ds_train = self.ds_train.map(
            preprocess, batched=True, remove_columns=self.ds_train.column_names
        )
        self.ds_val = self.ds_val.map(
            preprocess, batched=True, remove_columns=self.ds_val.column_names
        )
        # Set format to torch tensors for DataLoader
        self.ds_train.set_format(type="torch", columns=["input_ids", "labels"])
        self.ds_val.set_format(type="torch", columns=["input_ids", "labels"])

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Inputs are already padded to max_seq_len
        input_ids = torch.stack([x["input_ids"] for x in batch])
        labels = torch.stack([x["labels"] for x in batch])
        attention_mask = (
            (input_ids != self.tokenizer.pad_token_id).long()
            if self.tokenizer.pad_token_id is not None
            else torch.ones_like(input_ids)
        )
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

    def __init__(self, model: AutoModelForCausalLM, lr: float = 1e-4, weight_decay: float = 0.0):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])  # logs lr, weight_decay
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.forward(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self.forward(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
        val_loss = outputs.loss
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


def build_model_and_tokenizer(cfg: DictConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    # Tokenizer
    model_source = (
        cfg.experiment.model.local_path
        if bool(cfg.experiment.model.use_local)
        else cfg.experiment.model.name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set up 4-bit (QLoRA) if requested
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

    # Base model: prefer dtype over deprecated torch_dtype; use device_map for placement
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        device_map=cfg.experiment.model.device_map,
        dtype=torch_dtype,  # prefer new 'dtype' arg
        quantization_config=quantization_config,
    )

    # Optional gradient checkpointing to save memory
    if bool(cfg.experiment.model.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()

    # LoRA config
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
    # Show resolved output directory

    print("#" * 50)
    # print(OmegaConf.to_yaml(HydraConfig.get()))
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

    # Data module
    if bool(cfg.experiment.data.use_local):
        # Load preprocessed dataset from disk
        ds_path = cfg.experiment.data.local_path
        print(f"Loading dataset from local path: {ds_path}")
        ds_dict = load_from_disk(ds_path)
        dm = ArxivDataModule(
            tokenizer=tokenizer,
            dataset_name=cfg.experiment.data.dataset_name,
            dataset_config=cfg.experiment.data.dataset_config,
            train_split=cfg.experiment.data.train_split,
            val_split=cfg.experiment.data.val_split,
            max_seq_len=cfg.experiment.data.max_seq_length,
            batch_size=cfg.experiment.data.batch_size,
            num_workers=cfg.experiment.data.num_workers,
            shuffle=True,
        )
        # Override internal datasets
        dm.ds_train = ds_dict["train"]
        dm.ds_val = ds_dict["validation"]
        # Re-run formatting since loaded dataset holds raw columns
        dm.ds_train.set_format(
            type="torch", columns=["input_ids", "labels"]
        ) if "input_ids" in dm.ds_train.column_names else None
        dm.ds_val.set_format(
            type="torch", columns=["input_ids", "labels"]
        ) if "input_ids" in dm.ds_val.column_names else None
    else:
        dm = ArxivDataModule(
            tokenizer=tokenizer,
            dataset_name=cfg.experiment.data.dataset_name,
            dataset_config=cfg.experiment.data.dataset_config,
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
