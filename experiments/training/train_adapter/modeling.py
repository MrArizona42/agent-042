from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
)

from .config import AppConfig

logger = logging.getLogger(__name__)


def _load_tokenizer(model_path: Path) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def _resolve_model_path(cfg: AppConfig, candidate: str | Path) -> Path:
    project_root = Path(cfg.paths.project_root)
    path = Path(candidate)
    if not path.is_absolute():
        path = project_root / path
    return path


def load_base_model_and_tokenizer(
    cfg: AppConfig,
    *,
    for_training: bool,
) -> Tuple[Any, PreTrainedTokenizerBase]:
    model_cfg = cfg.experiment.model
    project_root = Path(cfg.paths.project_root)
    model_path = _resolve_model_path(cfg, model_cfg.local_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    tokenizer = _load_tokenizer(model_path)

    load_in_4bit = bool(model_cfg.load_in_4bit)
    dtype_str = model_cfg.dtype
    torch_dtype = getattr(torch, dtype_str)

    device_map = model_cfg.device_map
    if load_in_4bit and device_map == "auto":
        # Avoid device_map="auto" with quantized models under Lightning:
        # Lightning manages device placement, so pin to the current CUDA device.
        # This respects CUDA_VISIBLE_DEVICES and avoids conflicts with Trainer.
        if torch.cuda.is_available():
            device_map = {"": torch.cuda.current_device()}
        else:
            device_map = {"": "cpu"}

    quant_cfg = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=model_cfg.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, model_cfg.bnb_4bit_compute_dtype)
            if model_cfg.bnb_4bit_compute_dtype
            else torch_dtype,
        )
        if load_in_4bit
        else None
    )

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
        device_map=device_map,
        torch_dtype=torch_dtype,
        quantization_config=quant_cfg,
        local_files_only=True,
        offload_folder=str(offload_path) if offload_path else None,
    )

    model.config.use_cache = False

    if for_training and load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=model_cfg.gradient_checkpointing,
        )
    elif for_training and model_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model, tokenizer


def _attach_lora_adapter(model: Any, cfg: AppConfig) -> Any:
    lora_cfg = cfg.experiment.lora
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora_cfg.target_modules),
    )
    return get_peft_model(model, peft_config)


def build_model_and_tokenizer(cfg: AppConfig) -> Tuple[Any, PreTrainedTokenizerBase]:
    model, tokenizer = load_base_model_and_tokenizer(cfg, for_training=True)
    model = _attach_lora_adapter(model, cfg)

    model.print_trainable_parameters()
    return model, tokenizer
