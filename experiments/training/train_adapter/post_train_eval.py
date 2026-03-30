from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import MLFlowLogger
from transformers import PreTrainedTokenizerBase

from experiments.eval.eval_scripts.datasets import load_dataset_samples
from experiments.eval.eval_scripts.runner import calculate_metrics

from .config import AppConfig
from .mlflow_utils import log_evaluation_summary

logger = logging.getLogger(__name__)


def run_post_train_evaluation(
    *,
    cfg: AppConfig,
    raw_cfg: DictConfig,
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    mlf_logger: MLFlowLogger,
    run_artifacts_dir: Path,
    lineage: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate the freshly trained adapter before any manual registry action."""
    del raw_cfg

    eval_cfg = cfg.experiment.evaluation
    if not eval_cfg.enabled:
        return []
    if eval_cfg.task != "summarize":
        raise ValueError(
            f"Post-train evaluation currently supports summarize only, got {eval_cfg.task!r}"
        )

    samples = load_dataset_samples(
        eval_cfg.task, eval_cfg.dataset_name, limit=eval_cfg.sample_limit
    )
    if not samples:
        raise RuntimeError(
            f"No evaluation samples loaded for {eval_cfg.task}/{eval_cfg.dataset_name}"
        )

    bundle = _generate_local_generation_bundle(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        batch_size=eval_cfg.batch_size,
        max_new_tokens=eval_cfg.max_new_tokens,
        temperature=eval_cfg.temperature,
        do_sample=eval_cfg.do_sample,
        run_id=mlf_logger.run_id,
    )

    prediction_data = {
        "task": eval_cfg.task,
        "dataset_name": eval_cfg.dataset_name,
        "kb_name": None,
        "base_model": str(cfg.experiment.model.local_path),
        "eval_context": {
            "temperature": eval_cfg.temperature,
            "max_tokens": eval_cfg.max_new_tokens,
            "extra": {
                "evaluation_backend": "local_peft_generation",
                "training_run_id": mlf_logger.run_id,
                "dataset_dvc_hash": lineage.get("dataset_dvc_hash"),
                "git_sha": lineage.get("git_sha"),
                "run_artifacts_dir": str(run_artifacts_dir),
            },
        },
        "bundles": [bundle],
    }

    all_rows: list[dict[str, Any]] = []
    for metric_name in eval_cfg.metrics:
        rows = calculate_metrics(metric=metric_name, prediction_data=prediction_data)
        all_rows.extend(rows)

    evaluation_dir = run_artifacts_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    summary = {row["metric_name"]: row["metric_value"] for row in all_rows}
    payload = {
        "task": eval_cfg.task,
        "dataset_name": eval_cfg.dataset_name,
        "metrics": summary,
        "rows": [_json_ready_row(row) for row in all_rows],
    }
    (evaluation_dir / "post_train_eval.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    log_evaluation_summary(
        mlf_logger,
        cfg,
        summary_metrics=summary,
        evaluation_dir=evaluation_dir,
    )
    logger.info(
        "Post-train evaluation complete for run %s: %s",
        mlf_logger.run_id,
        ", ".join(f"{k}={v:.4f}" for k, v in sorted(summary.items())),
    )
    return all_rows


def _generate_local_generation_bundle(
    *,
    cfg: AppConfig,
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    samples: list[dict[str, str]],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    run_id: str,
) -> dict[str, Any]:
    prompts = [cfg.experiment.data.prompt_template.format(article=s["question"]) for s in samples]
    predictions: list[str] = []
    references: list[str] = []
    judge_samples: list[dict[str, str]] = []
    sample_details: list[dict[str, Any]] = []

    device = _resolve_model_device(model)
    original_padding_side = tokenizer.padding_side
    original_use_cache = getattr(model.config, "use_cache", None)

    model.eval()
    tokenizer.padding_side = "left"
    if original_use_cache is not None:
        model.config.use_cache = True

    try:
        with torch.inference_mode():
            for start in range(0, len(prompts), max(1, batch_size)):
                batch_prompts = prompts[start : start + max(1, batch_size)]
                batch_samples = samples[start : start + max(1, batch_size)]
                encoded = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=cfg.experiment.data.source_max_length,
                    add_special_tokens=False,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                generate_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                if do_sample:
                    generate_kwargs["temperature"] = temperature

                outputs = model.generate(
                    **encoded,
                    **generate_kwargs,
                )

                prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
                for offset, sample in enumerate(batch_samples):
                    generated_tokens = outputs[offset][int(prompt_lengths[offset]) :]
                    prediction = tokenizer.decode(
                        generated_tokens, skip_special_tokens=True
                    ).strip()
                    reference = sample.get("answer", "")

                    predictions.append(prediction)
                    references.append(reference)
                    judge_samples.append(
                        {
                            "question": sample["question"],
                            "answer": prediction,
                            "reference": reference,
                            "context": "",
                        }
                    )
                    sample_details.append(
                        {
                            "sample_idx": start + offset,
                            "sample_id": sample.get("id"),
                            "input": sample["question"],
                            "output": prediction,
                            "reference": reference,
                            "detail": {"prompt": batch_prompts[offset]},
                        }
                    )
    finally:
        tokenizer.padding_side = original_padding_side
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache

    return {
        "rag_alias": "none",
        "lora_alias": "local",
        "lora_info": {
            "adapter_name": "lora-summarize-local",
            "adapter_version": None,
            "adapter_mlflow_run_id": run_id,
        },
        "rag_enabled": False,
        "predictions": predictions,
        "references": references,
        "judge_samples": judge_samples,
        "sample_details": sample_details,
    }


def _resolve_model_device(model: Any) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _json_ready_row(row: dict[str, Any]) -> dict[str, Any]:
    ready: dict[str, Any] = {}
    for key, value in row.items():
        if hasattr(value, "isoformat"):
            ready[key] = value.isoformat()
        else:
            ready[key] = value
    return ready
