from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from .config import DataConfig, DatasetConfig, TaskConfig

logger = logging.getLogger(__name__)


class PromptTargetDataModule(pl.LightningDataModule):
    """LightningDataModule for prompt-target CausalLM supervision."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        task_cfg: TaskConfig,
        dataset_cfg: DatasetConfig,
        data_cfg: DataConfig,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.task_cfg = task_cfg
        self.dataset_cfg = dataset_cfg
        self.data_cfg = data_cfg
        self.shuffle = shuffle
        self.ds_train = None
        self.ds_val = None

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        budget = data_cfg.source_max_length + data_cfg.target_max_length + 1  # +1 for EOS
        if budget > data_cfg.max_seq_length:
            raise ValueError(
                f"source_max_length ({data_cfg.source_max_length}) + "
                f"target_max_length ({data_cfg.target_max_length}) + 1 (EOS) = {budget} "
                f"exceeds max_seq_length ({data_cfg.max_seq_length}). "
                f"Reduce source or target budget so they fit within the sequence limit."
            )

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = load_from_disk(self.dataset_cfg.local_path)
        train_split = self.dataset_cfg.train_split
        validation_split = self.dataset_cfg.validation_split

        if train_split not in dataset:
            available_splits = sorted(dataset.keys())
            raise KeyError(
                f"Dataset {self.dataset_cfg.local_path} is missing train split {train_split!r}; "
                f"available splits: {available_splits}"
            )

        train_dataset = dataset[train_split]
        if validation_split:
            if validation_split not in dataset:
                available_splits = sorted(dataset.keys())
                raise KeyError(
                    f"Dataset {self.dataset_cfg.local_path} is missing validation split "
                    f"{validation_split!r}; available splits: {available_splits}"
                )
            val_dataset = dataset[validation_split]
        else:
            validation_fraction = float(self.dataset_cfg.validation_fraction)
            if not 0 < validation_fraction < 1:
                raise ValueError(
                    "Dataset config must provide validation_split or set "
                    "validation_fraction to a value between 0 and 1."
                )
            split_dataset = train_dataset.train_test_split(
                test_size=validation_fraction,
                seed=int(self.dataset_cfg.split_seed),
                shuffle=True,
            )
            train_dataset = split_dataset["train"]
            val_dataset = split_dataset["test"]
            logger.info(
                "Created validation split from %s using validation_fraction=%s",
                train_split,
                validation_fraction,
            )

        self.ds_train = self._with_transform(train_dataset)
        self.ds_val = self._with_transform(val_dataset)

    def _with_transform(self, dataset):
        max_len = self.data_cfg.max_seq_length
        source_max = self.data_cfg.source_max_length
        target_max = self.data_cfg.target_max_length
        prompt_template = self.task_cfg.prompt_template
        prompt_field_map = dict(self.dataset_cfg.prompt_field_map)
        target_field = self.dataset_cfg.target_field

        def transform(example: Dict[str, Any]) -> Dict[str, Any]:
            is_batched = any(
                isinstance(example.get(field_name), list)
                for field_name in (*prompt_field_map.values(), target_field)
            )

            def build(prompt_text: str, target_text: str) -> Tuple[List[int], int]:
                prompt_ids = self.tokenizer(
                    prompt_text,
                    truncation=True,
                    max_length=source_max,
                    add_special_tokens=False,
                )["input_ids"]
                target_ids = self.tokenizer(
                    target_text,
                    truncation=True,
                    max_length=target_max,
                    add_special_tokens=False,
                )["input_ids"]
                input_ids = (prompt_ids + target_ids + [self.tokenizer.eos_token_id])[:max_len]
                prompt_len = len(prompt_ids)
                return input_ids, prompt_len

            def normalize_text(value: Any) -> str:
                if value is None:
                    return ""
                if isinstance(value, str):
                    return value
                return str(value)

            def build_prompt(record: Dict[str, Any]) -> str:
                return prompt_template.format(
                    **{
                        template_var: normalize_text(record.get(dataset_field))
                        for template_var, dataset_field in prompt_field_map.items()
                    }
                )

            def iter_records(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
                field_names = {target_field, *prompt_field_map.values()}
                columns: Dict[str, List[Any]] = {}
                record_count = 0
                for field_name in field_names:
                    value = batch.get(field_name)
                    if isinstance(value, list):
                        columns[field_name] = value
                        record_count = max(record_count, len(value))
                    elif value is None:
                        columns[field_name] = []
                    else:
                        columns[field_name] = [value]
                        record_count = max(record_count, 1)

                records: List[Dict[str, Any]] = []
                for index in range(record_count):
                    record = {}
                    for field_name, values in columns.items():
                        record[field_name] = values[index] if index < len(values) else ""
                    records.append(record)
                return records

            if is_batched:
                ids_list: List[List[int]] = []
                prompt_lens: List[int] = []
                for record in iter_records(example):
                    prompt = build_prompt(record)
                    target_text = normalize_text(record.get(target_field))
                    ids, plen = build(prompt, target_text)
                    ids_list.append(ids)
                    prompt_lens.append(plen)
                return {"input_ids": ids_list, "prompt_len": prompt_lens}

            prompt = build_prompt(example)
            target_text = normalize_text(example.get(target_field))
            ids, plen = build(prompt, target_text)
            return {"input_ids": ids, "prompt_len": plen}

        dataset.set_transform(transform)
        return dataset

    def _collate(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        padded = self.tokenizer.pad(
            {"input_ids": [ex["input_ids"] for ex in batch]},
            padding=True,
            max_length=self.data_cfg.max_seq_length,
            return_tensors="pt",
        )
        input_ids = padded["input_ids"]
        attention_mask = padded["attention_mask"]
        labels = input_ids.clone().masked_fill(attention_mask.eq(0), -100)
        zero_target_count = 0
        for idx, ex in enumerate(batch):
            prompt_tokens = min(int(ex.get("prompt_len", 0)), labels.shape[1])
            non_pad_target = int(attention_mask[idx, prompt_tokens:].sum().item())
            if non_pad_target <= 1:
                zero_target_count += 1
            if prompt_tokens > 0 and not self.data_cfg.train_on_inputs:
                labels[idx, :prompt_tokens] = -100
        if zero_target_count > 0:
            logger.warning(
                "%d / %d samples in batch have zero supervised target tokens",
                zero_target_count,
                len(batch),
            )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "zero_target_count": torch.tensor(zero_target_count, dtype=torch.long),
            "batch_size": torch.tensor(len(batch), dtype=torch.long),
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            batch_size=self.data_cfg.batch_size,
            shuffle=self.shuffle,
            num_workers=self.data_cfg.num_workers,
            collate_fn=self._collate,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            batch_size=self.data_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
            collate_fn=self._collate,
            pin_memory=torch.cuda.is_available(),
        )


ArxivDataModule = PromptTargetDataModule
