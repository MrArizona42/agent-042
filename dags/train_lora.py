"""DAG: LoRA adapter training and post-train evaluation.

Runs LoRA fine-tuning on the Airflow GPU worker. The ``train_adapter``
task invokes ``start_train.py`` as a subprocess to avoid Hydra global-state
collisions across concurrent runs and returns a small manifest path via XCom.

The ``eval_adapter`` task consumes that manifest and runs post-train
evaluation as a separate Airflow step using the exported adapter artifacts.

**Does NOT** register, promote, or sync. Those are manual decisions made
in ``experiments/training/lora_ops.ipynb`` after inspecting the training run.

Params (Airflow UI):
    experiment_config : Hydra experiment config name (default: train_adapter)
    hydra_overrides   : JSON list of Hydra override strings
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk import Param

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
}


def _train_adapter(**context) -> str:
    """Run training as subprocess; return the training manifest path."""
    params = context["params"]
    experiment_config = params["experiment_config"]
    overrides_raw = params.get("hydra_overrides", "[]")
    overrides: list[str] = json.loads(overrides_raw) if overrides_raw else []
    task_instance = context["ti"]
    dag_run = context.get("dag_run")

    cmd = [
        "python",
        "-m",
        "experiments.training.train_adapter.start_train",
        f"experiment={experiment_config}",
        *overrides,
    ]

    env = {
        **os.environ,
        "PYTHONPATH": f"{PROJECT_ROOT}:{PROJECT_ROOT / 'src'}",
        "AIRFLOW_CTX_DAG_ID": context["dag"].dag_id,
        "AIRFLOW_CTX_TASK_ID": task_instance.task_id,
        "AIRFLOW_CTX_DAG_RUN_ID": dag_run.run_id if dag_run else "",
        "AIRFLOW_CTX_EXECUTION_DATE": str(context.get("logical_date", "")),
        "TRAIN_ADAPTER_SKIP_POST_TRAIN_EVAL": "1",
    }

    print(
        f"Starting training DAG task: \n"
        f"  dag={context['dag'].dag_id} \n"
        f"  run_id={env['AIRFLOW_CTX_DAG_RUN_ID']}"
    )
    print(f"Experiment config: {experiment_config}")
    if overrides:
        print("Hydra overrides:")
        for item in overrides:
            print(f"  {item}")

    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    run_id = ""
    manifest_path = ""
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        if line.startswith("run_id="):
            run_id = line.split("=", 1)[1].strip()
        if line.startswith("training_manifest="):
            manifest_path = line.split("=", 1)[1].strip()

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    if not manifest_path:
        raise RuntimeError("Training completed but no training_manifest was emitted.")
    if not run_id:
        print("Training completed but no run_id was emitted.")
    return manifest_path


def _eval_adapter(**context) -> None:
    """Run post-train evaluation as subprocess using the training manifest."""
    task_instance = context["ti"]
    dag_run = context.get("dag_run")
    manifest_path = context["ti"].xcom_pull(task_ids="train_adapter")
    if not manifest_path:
        raise RuntimeError("No training manifest received from train_adapter task")

    cmd = [
        "python",
        "-m",
        "experiments.training.train_adapter.start_post_train_eval",
        "--manifest",
        manifest_path,
    ]
    env = {
        **os.environ,
        "PYTHONPATH": f"{PROJECT_ROOT}:{PROJECT_ROOT / 'src'}",
        "AIRFLOW_CTX_DAG_ID": context["dag"].dag_id,
        "AIRFLOW_CTX_TASK_ID": task_instance.task_id,
        "AIRFLOW_CTX_DAG_RUN_ID": dag_run.run_id if dag_run else "",
        "AIRFLOW_CTX_EXECUTION_DATE": str(context.get("logical_date", "")),
    }

    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


with DAG(
    dag_id="train_lora",
    default_args=default_args,
    description="LoRA adapter training via Hydra + PyTorch Lightning",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["training", "lora"],
    params={
        "experiment_config": Param(
            default="train_adapter",
            type="string",
            description=(
                "Hydra experiment config name under experiments/training/conf/experiment/"
            ),
        ),
        "hydra_overrides": Param(
            default="[]",
            type="string",
            description=(
                "JSON list of Hydra override strings in key=value format. "
                "Example: "
                '["experiment.training.lr=2e-5", '
                '"experiment.lora.r=16", '
                '"experiment.trainer.max_epochs=3"]'
            ),
        ),
    },
) as dag:
    train = PythonOperator(
        task_id="train_adapter",
        python_callable=_train_adapter,
        queue="gpu",
    )

    evaluate = PythonOperator(
        task_id="eval_adapter",
        python_callable=_eval_adapter,
        queue="gpu",
    )

    train >> evaluate
