"""DAG: LoRA adapter training.

Runs LoRA fine-tuning on the Airflow GPU worker. The ``train_adapter``
task invokes ``start_train.py`` as a subprocess to avoid Hydra global-state
collisions across concurrent runs and returns a small training summary path via XCom.

The main output of the DAG is the exported adapter plus the best monitored
validation score recorded in ``training_summary.json``.

**Does NOT** register, promote, or sync. Those are manual decisions made
in ``experiments/training/lora_ops.ipynb`` after inspecting the training run.

Params (Airflow UI):
    experiment_config : optional Hydra experiment preset name (default: arxiv_summarization)
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
    """Run training as subprocess; return the training summary path."""
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
    ]
    if experiment_config:
        cmd.append(f"+experiment={experiment_config}")
    cmd.extend(overrides)

    env = {
        **os.environ,
        "PYTHONPATH": f"{PROJECT_ROOT}:{PROJECT_ROOT / 'src'}",
        "AIRFLOW_CTX_DAG_ID": context["dag"].dag_id,
        "AIRFLOW_CTX_TASK_ID": task_instance.task_id,
        "AIRFLOW_CTX_DAG_RUN_ID": dag_run.run_id if dag_run else "",
        "AIRFLOW_CTX_EXECUTION_DATE": str(context.get("logical_date", "")),
    }

    print(
        f"Starting training DAG task: \n"
        f"  dag={context['dag'].dag_id} \n"
        f"  run_id={env['AIRFLOW_CTX_DAG_RUN_ID']}"
    )
    print(f"Experiment preset: {experiment_config or '<root defaults>'}")
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
    summary_path = ""
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        if line.startswith("run_id="):
            run_id = line.split("=", 1)[1].strip()
        if line.startswith("training_summary="):
            summary_path = line.split("=", 1)[1].strip()

    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    if not summary_path:
        raise RuntimeError("Training completed but no training_summary was emitted.")
    if not run_id:
        print("Training completed but no run_id was emitted.")
    return summary_path


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
            default="arxiv_summarization",
            type="string",
            description=(
                "Optional Hydra experiment preset under experiments/training/conf/experiment/. "
                "It overrides the top-level task/dataset/model/lora/data/training groups."
            ),
        ),
        "hydra_overrides": Param(
            default="[]",
            type="string",
            description=(
                "JSON list of Hydra override strings in key=value format. "
                "Example: "
                '["training.lr=2e-5", '
                '"lora.r=16", '
                '"trainer.max_epochs=3"]'
            ),
        ),
    },
) as dag:
    train = PythonOperator(
        task_id="train_adapter",
        python_callable=_train_adapter,
        queue="gpu",
    )
