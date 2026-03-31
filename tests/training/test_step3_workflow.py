from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

pytest.importorskip("mlflow")
pytest.importorskip("pytorch_lightning")

from experiments.training.train_adapter.config import load_app_config, register_configs
from experiments.training.train_adapter.mlflow_utils import (
    _find_dataset_dvc_hash,
    _git_context,
    log_training_lineage,
)
from experiments.training.train_adapter.pipeline import (
    _restore_best_checkpoint_for_export,
    _write_training_manifest,
    run_post_train_evaluation_from_manifest,
)
from experiments.training.train_adapter.post_train_eval import (
    _resolve_model_device,
    run_post_train_evaluation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _compose_training_cfg(overrides: list[str] | None = None):
    register_configs()
    GlobalHydra.instance().clear()
    config_dir = PROJECT_ROOT / "experiments" / "training" / "conf"
    cli_overrides = [f"paths.project_root={PROJECT_ROOT.as_posix()}"]
    if overrides:
        cli_overrides.extend(overrides)

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="config", overrides=cli_overrides)


def test_log_training_lineage_writes_metadata_and_logs_key_fields(tmp_path):
    raw_cfg = _compose_training_cfg(
        [
            "experiment.data.batch_size=2",
            "experiment.trainer.accumulate_grad_batches=4",
            "experiment.trainer.devices=2",
        ]
    )
    app_cfg = load_app_config(raw_cfg)
    dataset_path = PROJECT_ROOT / "assets" / "datasets" / "arxiv-summarization"
    run_artifacts_dir = tmp_path / "run"

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
    assert lineage["dataset_dvc_hash"] == "7acaed09289faae03d0ed1ecb8affcc0.dir"
    assert lineage["effective_batch_size"] == 16

    saved_lineage = json.loads((metadata_dir / "lineage.json").read_text(encoding="utf-8"))
    assert saved_lineage["trainable_param_count"] == 42

    logged_params = {call.args[1]: call.args[2] for call in client.log_param.call_args_list}
    assert logged_params["effective_batch_size"] == 16
    assert logged_params["trainable_param_count"] == 42
    assert logged_params["dataset_dvc_hash"] == "7acaed09289faae03d0ed1ecb8affcc0.dir"

    set_tags = {(call.args[1], call.args[2]) for call in client.set_tag.call_args_list}
    assert ("run.orchestrator", "cli") in set_tags
    assert client.log_artifacts.called


def test_run_post_train_evaluation_writes_summary_and_links_to_run_id(tmp_path):
    raw_cfg = _compose_training_cfg(["experiment.evaluation.sample_limit=1"])
    app_cfg = load_app_config(raw_cfg)
    run_artifacts_dir = tmp_path / "run"
    run_artifacts_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "rag_alias": "none",
        "lora_alias": "local",
        "lora_info": {
            "adapter_name": "lora-summarize-local",
            "adapter_version": None,
            "adapter_mlflow_run_id": "run-123",
        },
        "rag_enabled": False,
        "predictions": ["summary"],
        "references": ["summary"],
        "judge_samples": [
            {
                "question": "article",
                "answer": "summary",
                "reference": "summary",
                "context": "",
            }
        ],
        "sample_details": [
            {
                "sample_idx": 0,
                "input": "article",
                "output": "summary",
                "reference": "summary",
                "detail": {},
            }
        ],
    }
    row = {
        "metric_name": "rouge_l",
        "metric_value": 0.9,
        "adapter_mlflow_run_id": "run-123",
    }

    mlf_logger = SimpleNamespace(run_id="run-123", experiment=MagicMock())

    with (
        patch(
            "experiments.training.train_adapter.post_train_eval.load_dataset_samples",
            return_value=[{"question": "article", "answer": "summary"}],
        ),
        patch(
            "experiments.training.train_adapter.post_train_eval._generate_local_generation_bundle",
            return_value=bundle,
        ) as mock_bundle,
        patch(
            "experiments.training.train_adapter.post_train_eval.calculate_metrics",
            return_value=[row],
        ) as mock_calculate,
        patch(
            "experiments.training.train_adapter.post_train_eval.log_evaluation_summary"
        ) as mock_log_summary,
    ):
        rows = run_post_train_evaluation(
            cfg=app_cfg,
            raw_cfg=raw_cfg,
            model=object(),
            tokenizer=object(),
            mlf_logger=mlf_logger,
            run_artifacts_dir=run_artifacts_dir,
            lineage={"dataset_dvc_hash": "hash-123", "git_sha": "abc123"},
        )

    assert rows == [row]
    assert mock_bundle.call_args.kwargs["run_id"] == "run-123"

    prediction_data = mock_calculate.call_args.kwargs["prediction_data"]
    assert prediction_data["eval_context"]["extra"]["training_run_id"] == "run-123"
    assert prediction_data["eval_context"]["extra"]["dataset_dvc_hash"] == "hash-123"

    summary_path = run_artifacts_dir / "evaluation" / "post_train_eval.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["metrics"] == {"rouge_l": 0.9}
    mock_log_summary.assert_called_once()


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


def test_write_training_manifest_records_paths(tmp_path):
    run_artifacts_dir = tmp_path / "run"
    save_dir = run_artifacts_dir / "export"
    manifest_path = _write_training_manifest("run-123", save_dir, run_artifacts_dir)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["run_id"] == "run-123"
    assert manifest["save_dir"] == str(save_dir)
    assert manifest["run_artifacts_dir"] == str(run_artifacts_dir)
    assert manifest["resolved_config_path"] == str(
        run_artifacts_dir / "metadata" / "resolved_config.json"
    )


def test_run_post_train_evaluation_from_manifest_reloads_saved_export(tmp_path):
    raw_cfg = _compose_training_cfg(["experiment.evaluation.sample_limit=1"])
    app_cfg = load_app_config(raw_cfg)
    run_artifacts_dir = tmp_path / "run"
    metadata_dir = run_artifacts_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    save_dir = run_artifacts_dir / "export"
    save_dir.mkdir(parents=True, exist_ok=True)

    (metadata_dir / "resolved_config.json").write_text(
        json.dumps(OmegaConf.to_container(raw_cfg, resolve=True), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (metadata_dir / "lineage.json").write_text(
        json.dumps({"dataset_dvc_hash": "hash-123", "git_sha": "abc123"}),
        encoding="utf-8",
    )
    manifest_path = _write_training_manifest("run-123", save_dir, run_artifacts_dir)
    fake_model = object()
    fake_tokenizer = object()
    fake_logger = SimpleNamespace(run_id="run-123", experiment=MagicMock())
    rows = [{"metric_name": "rouge_l", "metric_value": 0.9}]

    with (
        patch(
            "experiments.training.train_adapter.pipeline.configure_mlflow_tracking"
        ) as mock_tracking,
        patch(
            "experiments.training.train_adapter.pipeline.build_mlflow_run_logger",
            return_value=fake_logger,
        ) as mock_run_logger,
        patch(
            "experiments.training.train_adapter.pipeline.load_exported_model_and_tokenizer",
            return_value=(fake_model, fake_tokenizer),
        ) as mock_load_exported,
        patch(
            "experiments.training.train_adapter.pipeline.run_post_train_evaluation",
            return_value=rows,
        ) as mock_run_eval,
    ):
        result = run_post_train_evaluation_from_manifest(manifest_path)

    assert result == rows
    mock_tracking.assert_called_once()
    mock_run_logger.assert_called_once_with("run-123")
    mock_load_exported.assert_called_once()
    assert mock_load_exported.call_args.args[1] == save_dir

    eval_kwargs = mock_run_eval.call_args.kwargs
    assert (
        eval_kwargs["cfg"].experiment.evaluation.sample_limit
        == app_cfg.experiment.evaluation.sample_limit
    )
    assert eval_kwargs["mlf_logger"] is fake_logger
    assert eval_kwargs["run_artifacts_dir"] == run_artifacts_dir
    assert eval_kwargs["lineage"]["dataset_dvc_hash"] == "hash-123"


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


def test_resolve_model_device_prefers_cuda_from_hf_device_map():
    model = MagicMock()
    model.hf_device_map = {"": 0, "lm_head": "cpu"}

    assert _resolve_model_device(model) == torch.device("cuda:0")


def test_resolve_model_device_prefers_trainable_cuda_parameter():
    cpu_param = MagicMock(requires_grad=False)
    cpu_param.device = torch.device("cpu")
    cuda_param = MagicMock(requires_grad=True)
    cuda_param.device = torch.device("cuda:0")
    model = MagicMock()
    model.hf_device_map = {}
    model.parameters.return_value = iter([cpu_param, cuda_param])
    model.buffers.return_value = iter([])

    assert _resolve_model_device(model) == torch.device("cuda:0")
