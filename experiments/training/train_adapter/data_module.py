from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from .config import DataConfig


class ArxivDataModule(pl.LightningDataModule):
    """LightningDataModule that tokenizes ArXiv article/abstract pairs on the fly."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        data_cfg: DataConfig,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = data_cfg
        self.shuffle = shuffle
        self.ds_train = None
        self.ds_val = None

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = load_from_disk(self.cfg.local_path)
        self.ds_train = self._with_transform(dataset["train"])
        self.ds_val = self._with_transform(dataset["validation"])

    def _with_transform(self, dataset):
        max_len = self.cfg.max_seq_length
        prompt_template = self.cfg.prompt_template

        def transform(example: Dict[str, Any]) -> Dict[str, Any]:
            is_batched = isinstance(example.get("article"), list)

            def build(prompt_text: str, target_text: str) -> Tuple[List[int], int]:
                prompt_ids = self.tokenizer(
                    prompt_text,
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=False,
                )["input_ids"]
                target_ids = self.tokenizer(
                    target_text,
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=False,
                )["input_ids"]
                input_ids = (prompt_ids + target_ids + [self.tokenizer.eos_token_id])[:max_len]
                prompt_len = min(len(prompt_ids), max_len)
                return input_ids, prompt_len

            if is_batched:
                arts = example.get("article") or []
                abs_ = example.get("abstract") or []
                n = max(len(arts), len(abs_))
                if len(arts) < n:
                    arts += [""] * (n - len(arts))
                if len(abs_) < n:
                    abs_ += [""] * (n - len(abs_))
                ids_list: List[List[int]] = []
                prompt_lens: List[int] = []
                for art, summ in zip(arts, abs_):
                    prompt = prompt_template.format(article=art)
                    ids, plen = build(prompt, summ)
                    ids_list.append(ids)
                    prompt_lens.append(plen)
                return {"input_ids": ids_list, "prompt_len": prompt_lens}

            article = example.get("article", "")
            summary = example.get("abstract", "")
            prompt = prompt_template.format(article=article)
            ids, plen = build(prompt, summary)
            return {"input_ids": ids, "prompt_len": plen}

        dataset.set_transform(transform)
        return dataset

    def _collate(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        padded = self.tokenizer.pad(
            {"input_ids": [ex["input_ids"] for ex in batch]},
            padding=True,
            max_length=self.cfg.max_seq_length,
            return_tensors="pt",
        )
        input_ids = padded["input_ids"]
        attention_mask = padded["attention_mask"]
        labels = input_ids.clone().masked_fill(attention_mask.eq(0), -100)
        for idx, ex in enumerate(batch):
            prompt_tokens = min(int(ex.get("prompt_len", 0)), labels.shape[1])
            if prompt_tokens > 0:
                labels[idx, :prompt_tokens] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            batch_size=self.cfg.batch_size,
            shuffle=self.shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=self._collate,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=self._collate,
            pin_memory=torch.cuda.is_available(),
        )
