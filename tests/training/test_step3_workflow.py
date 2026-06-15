from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("hydra")
pytest.importorskip("mlflow")
pytest.importorskip("pytorch_lightning")
pytest.importorskip("datasets")
torch = pytest.importorskip("torch")

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from datasets import Dataset, DatasetDict

from experiments.training.train_adapter.config import (
    DataConfig,
    DatasetConfig,
    TaskConfig,
    load_app_config,
    register_configs,
)
from experiments.training.train_adapter.data_module import PromptTargetDataModule
from experiments.training.train_adapter.mlflow_utils import (
    _find_dataset_dvc_hash,
    _git_context,
    log_training_lineage,
    setup_mlflow,
)
from experiments.training.train_adapter.pipeline import (
    _restore_best_checkpoint_for_export,
    _write_training_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, truncation, max_length, add_special_tokens):
        token_count = min(max_length, max(1, len(str(text).split())))
        return {"input_ids": list(range(2, 2 + token_count))}

    def pad(self, encoded_inputs, padding, max_length, return_tensors):
        sequences = [list(sequence[:max_length]) for sequence in encoded_inputs["input_ids"]]
        padded_length = max(len(sequence) for sequence in sequences)
        padded_inputs = []
        attention_masks = []
        for sequence in sequences:
            pad_length = padded_length - len(sequence)
            padded_inputs.append(sequence + [self.pad_token_id] * pad_length)
            attention_masks.append([1] * len(sequence) + [0] * pad_length)
        return {
            "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }


def _compose_training_cfg(overrides: list[str] | None = None):
    register_configs()
    GlobalHydra.instance().clear()
    config_dir = PROJECT_ROOT / "experiments" / "training" / "conf"
    cli_overrides = [f"paths.project_root={PROJECT_ROOT.as_posix()}"]
    if overrides:
        cli_overrides.extend(overrides)

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="config", overrides=cli_overrides)


def test_compose_training_cfg_uses_structured_defaults_and_selected_presets():
    raw_cfg = _compose_training_cfg()

    assert raw_cfg.task.name == "summarization"
    assert raw_cfg.task.run_name_prefix == "summarize"
    assert raw_cfg.model.dtype == "float16"
    assert raw_cfg.model.local_path.endswith("assets/models/Qwen/Qwen3-0.6B")
    assert raw_cfg.dataset.local_path.endswith("assets/datasets/arxiv-summarization")
    assert raw_cfg.dataset.prompt_field_map.article == "article"
    assert raw_cfg.data.max_seq_length == 768
    assert raw_cfg.data.batch_size == 1
    assert raw_cfg.training.seed == 42
    assert raw_cfg.training.lr == 1e-5
    assert raw_cfg.training.weight_decay == 0.01
    assert "env_path" not in raw_cfg.tracking
    assert raw_cfg.trainer.precision == "16-mixed"
    assert raw_cfg.callbacks.checkpoint.monitor == "val_loss"
    assert raw_cfg.callbacks.checkpoint.save_top_k == 3
    assert raw_cfg.logger.experiment_name == "train_adapter"


def test_compose_training_cfg_supports_code_generation_preset():
    raw_cfg = _compose_training_cfg(["+experiment=open_code_instruct_qwen"])

    assert raw_cfg.task.name == "code_generation"
    assert raw_cfg.task.run_name_prefix == "codegen"
    assert raw_cfg.dataset.local_path.endswith("assets/datasets/open-code-instruct")
    assert raw_cfg.dataset.validation_split is None
    assert raw_cfg.dataset.validation_fraction == pytest.approx(0.01)
    assert raw_cfg.dataset.prompt_field_map.instruction == "input"
    assert raw_cfg.dataset.target_field == "output"


def test_compose_training_cfg_supports_long_context_data_preset():
    raw_cfg = _compose_training_cfg(
        [
            "+experiment=open_code_instruct_qwen",
            "data=sft_1536_tokens",
        ]
    )

    assert raw_cfg.task.name == "code_generation"
    assert raw_cfg.data.max_seq_length == 1536
    assert raw_cfg.data.source_max_length == 1023
    assert raw_cfg.data.target_max_length == 512
    assert raw_cfg.data.batch_size == 1


def test_setup_mlflow_instantiates_from_typed_logger_config():
    raw_cfg = _compose_training_cfg()
    app_cfg = load_app_config(raw_cfg)
    fake_logger = MagicMock()

    with (
        patch("experiments.training.train_adapter.mlflow_utils.configure_mlflow_tracking"),
        patch(
            "experiments.training.train_adapter.mlflow_utils.mlflow.get_tracking_uri",
            return_value="http://mlflow.local",
        ),
        patch(
            "experiments.training.train_adapter.mlflow_utils.instantiate",
            return_value=fake_logger,
        ) as mock_instantiate,
    ):
        result = setup_mlflow(app_cfg)
    assert result is fake_logger
    args, kwargs = mock_instantiate.call_args
    assert args[0] == app_cfg.logger
    assert kwargs["tracking_uri"] == "http://mlflow.local"
    assert kwargs["run_name"].startswith("summarize-arxiv-summarization-r8-lr")
    assert kwargs["tags"]["pipeline"] == "train_adapter"
    assert kwargs["tags"]["training.task"] == "summarization"
    assert kwargs["tags"]["training.dataset"] == "arxiv-summarization"
    assert kwargs["tags"]["task.family"] == "summarization"


def test_hydra_instantiate_accepts_typed_factory_configs():
    raw_cfg = _compose_training_cfg()
    app_cfg = load_app_config(raw_cfg)

    assert instantiate(app_cfg.logger, _partial_=True) is not None
    assert instantiate(app_cfg.data_module, _partial_=True) is not None
    assert instantiate(app_cfg.trainer, _partial_=True) is not None


def test_log_training_lineage_writes_metadata_and_logs_key_fields(tmp_path):
    raw_cfg = _compose_training_cfg(
        [
            "data.batch_size=2",
            "trainer.accumulate_grad_batches=4",
            "trainer.devices=2",
        ]
    )
    app_cfg = load_app_config(raw_cfg)
    dataset_path = PROJECT_ROOT / "assets" / "datasets" / "arxiv-summarization"
    run_artifacts_dir = tmp_path / "run"
    expected_dataset_dvc_hash = _find_dataset_dvc_hash(dataset_path, PROJECT_ROOT)

    client = MagicMock()
    mlf_logger = SimpleNamespace(run_id="run-123", experiment=client)

    lineage = log_training_lineage(
        mlf_logger,
        app_cfg,
        raw_cfg,
        dataset_path=dataset_path,
        run_artifacts_dir=run_artifacts_dir,
        trainable_param_count=42,
    )

    metadata_dir = run_artifacts_dir / "metadata"
    assert (metadata_dir / "resolved_config.json").exists()
    assert (metadata_dir / "lineage.json").exists()
    assert lineage["dataset_dvc_hash"] == expected_dataset_dvc_hash
    assert lineage["effective_batch_size"] == 16

    saved_lineage = json.loads((metadata_dir / "lineage.json").read_text(encoding="utf-8"))
    assert saved_lineage["trainable_param_count"] == 42

    logged_params = {call.args[1]: call.args[2] for call in client.log_param.call_args_list}
    assert logged_params["effective_batch_size"] == 16
    assert logged_params["trainable_param_count"] == 42
    assert logged_params["dataset_dvc_hash"] == expected_dataset_dvc_hash

    set_tags = {(call.args[1], call.args[2]) for call in client.set_tag.call_args_list}
    assert ("run.orchestrator", "cli") in set_tags
    assert client.log_artifacts.called


def test_prompt_target_datamodule_supports_custom_fields_and_train_split_validation(tmp_path):
    dataset_path = tmp_path / "coding-dataset"
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "input": [
                        "Write a function that doubles a number.",
                        "Return the square of n.",
                        "Count vowels in a string.",
                        "Reverse a list in place.",
                    ],
                    "output": [
                        "def double(n):\n    return n * 2",
                        "def square(n):\n    return n ** 2",
                        (
                            "def count_vowels(text):\n    "
                            "return sum(ch in 'aeiou' for ch in text.lower())"
                        ),
                        "def reverse_in_place(items):\n    items.reverse()",
                    ],
                }
            )
        }
    )
    dataset.save_to_disk(str(dataset_path))

    datamodule = PromptTargetDataModule(
        tokenizer=FakeTokenizer(),
        task_cfg=TaskConfig(
            name="code_generation",
            run_name_prefix="codegen",
            prompt_template="Solve the coding task.\n\nTask:\n{instruction}\n\nSolution:\n",
        ),
        dataset_cfg=DatasetConfig(
            local_path=str(dataset_path),
            train_split="train",
            validation_split=None,
            validation_fraction=0.5,
            split_seed=7,
            prompt_field_map={"instruction": "input"},
            target_field="output",
            name="toy-code",
        ),
        data_cfg=DataConfig(
            max_seq_length=32,
            source_max_length=16,
            target_max_length=8,
            train_on_inputs=False,
            batch_size=2,
            num_workers=0,
        ),
    )

    datamodule.setup()

    assert len(datamodule.ds_train) == 2
    assert len(datamodule.ds_val) == 2

    sample = datamodule.ds_train[0]
    assert sample["prompt_len"] > 0
    assert sample["input_ids"]

    batch = datamodule._collate([datamodule.ds_train[0], datamodule.ds_train[1]])
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape == batch["input_ids"].shape


def test_restore_best_checkpoint_for_export_filters_non_live_keys(tmp_path):
    checkpoint_path = tmp_path / "best.ckpt"
    torch_state = {
        "state_dict": {
            "model.adapter.weight": 123,
            "model.base_layer.weight.absmax": 456,
        }
    }
    module = MagicMock()
    module.state_dict.return_value = {"model.adapter.weight": 0}
    module.load_state_dict.return_value = SimpleNamespace(missing_keys=[], unexpected_keys=[])

    with patch("experiments.training.train_adapter.pipeline.torch.load", return_value=torch_state):
        _restore_best_checkpoint_for_export(module, str(checkpoint_path))

    module.load_state_dict.assert_called_once_with(
        {"model.adapter.weight": 123},
        strict=False,
    )


def test_write_training_summary_records_paths_and_best_val_loss(tmp_path):
    run_artifacts_dir = tmp_path / "run"
    save_dir = run_artifacts_dir / "export"
    summary_path = _write_training_summary(
        "run-123",
        save_dir,
        run_artifacts_dir,
        best_checkpoint_path="C:/tmp/best.ckpt",
        best_model_score=0.123,
        monitor_name="val_loss",
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["run_id"] == "run-123"
    assert summary["save_dir"] == str(save_dir)
    assert summary["run_artifacts_dir"] == str(run_artifacts_dir)
    assert summary["metadata_dir"] == str(run_artifacts_dir / "metadata")
    assert summary["best_checkpoint_path"] == "C:/tmp/best.ckpt"
    assert summary["best_model_score"] == 0.123
    assert summary["best_val_loss"] == 0.123


def test_git_context_marks_project_root_as_safe_directory(tmp_path):
    expected_path = tmp_path.as_posix()

    with patch(
        "experiments.training.train_adapter.mlflow_utils.subprocess.check_output",
        side_effect=["abc123\n", " M experiments/training/train_adapter/pipeline.py\n"],
    ) as mock_check_output:
        sha, dirty = _git_context(tmp_path)

    assert sha == "abc123"
    assert dirty is True
    first_call = mock_check_output.call_args_list[0]
    second_call = mock_check_output.call_args_list[1]
    assert first_call.args[0][:3] == ["git", "-c", f"safe.directory={expected_path}"]
    assert second_call.args[0][:3] == ["git", "-c", f"safe.directory={expected_path}"]


def test_find_dataset_dvc_hash_parses_outs_md5_entry(tmp_path):
    datasets_dir = tmp_path / "assets" / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = datasets_dir / "arxiv-summarization"
    dataset_path.mkdir()
    (datasets_dir / "arxiv-summarization.dvc").write_text(
        "outs:\n- md5: hash-123.dir\n  path: arxiv-summarization\n",
        encoding="utf-8",
    )

    assert _find_dataset_dvc_hash(dataset_path, tmp_path) == "hash-123.dir"
