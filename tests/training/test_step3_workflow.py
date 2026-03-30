from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

pytest.importorskip("mlflow")
pytest.importorskip("pytorch_lightning")

from experiments.training.train_adapter.config import load_app_config, register_configs
from experiments.training.train_adapter.mlflow_utils import _git_context, log_training_lineage
from experiments.training.train_adapter.pipeline import _restore_best_checkpoint_for_export
from experiments.training.train_adapter.post_train_eval import run_post_train_evaluation

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
