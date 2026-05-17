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

from .config import AppConfig, ModelConfig

logger = logging.getLogger(__name__)


def _load_tokenizer(model_source: str, model_cfg: ModelConfig) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        use_fast=model_cfg.tokenizer_use_fast,
        local_files_only=model_cfg.local_files_only,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def _resolve_model_source(cfg: AppConfig, candidate: str | Path) -> str:
    project_root = Path(cfg.paths.project_root)
    path = Path(str(candidate)).expanduser()
    if not path.is_absolute():
        project_relative = project_root / path
        if cfg.model.local_files_only or project_relative.exists():
            return str(project_relative)
        return str(candidate)
    return str(path)


def load_base_model_and_tokenizer(
    cfg: AppConfig,
    *,
    for_training: bool,
) -> Tuple[Any, PreTrainedTokenizerBase]:
    model_cfg = cfg.model
    project_root = Path(cfg.paths.project_root)
    model_source = _resolve_model_source(cfg, model_cfg.local_path)
    resolved_model_path = Path(model_source)
    if (model_cfg.local_files_only or resolved_model_path.is_absolute()) and not resolved_model_path.exists():
        raise FileNotFoundError(f"Model path not found: {resolved_model_path}")

    tokenizer = _load_tokenizer(model_source, model_cfg)

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
        model_source,
        device_map=device_map,
        torch_dtype=torch_dtype,
        quantization_config=quant_cfg,
        local_files_only=model_cfg.local_files_only,
        trust_remote_code=model_cfg.trust_remote_code,
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
    lora_cfg = cfg.lora
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
