from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerBase

from .config import AppConfig

logger = logging.getLogger(__name__)


def build_model_and_tokenizer(cfg: AppConfig) -> Tuple[Any, PreTrainedTokenizerBase]:
    model_cfg = cfg.experiment.model
    project_root = Path(cfg.paths.project_root)
    model_path = Path(model_cfg.local_path)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_in_4bit = bool(model_cfg.load_in_4bit)
    dtype_str = model_cfg.dtype
    torch_dtype = getattr(torch, dtype_str)

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=model_cfg.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, model_cfg.bnb_4bit_compute_dtype) if model_cfg.bnb_4bit_compute_dtype else torch_dtype,
    ) if load_in_4bit else None

    offload_folder = model_cfg.offload_folder
    if offload_folder:
        offload_path = Path(offload_folder)
        if not offload_path.is_absolute():
            offload_path = project_root / offload_path
        offload_path.mkdir(parents=True, exist_ok=True)
    else:
        offload_path = None

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=model_cfg.device_map,
        torch_dtype=torch_dtype,
        quantization_config=quant_cfg,
        local_files_only=True,
        offload_folder=str(offload_path) if offload_path else None,
    )

    if model_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_cfg = cfg.experiment.lora
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora_cfg.target_modules),
    )
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    return model, tokenizer

