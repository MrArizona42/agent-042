"""DAG: LoRA adapter training.

Runs LoRA fine-tuning on the Airflow GPU worker.  The ``train_adapter``
task is a ``PythonOperator`` that invokes ``start_train.py`` as a subprocess
to avoid Hydra global-state collisions across concurrent runs.

The training result (``run_id``) is printed on the last output line and
captured as XCom for downstream use.

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
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
}


def _train_adapter(**context) -> str:
    """Run training as subprocess; return the MLflow run_id."""
    params = context["params"]
    experiment_config = params["experiment_config"]
    overrides_raw = params.get("hydra_overrides", "[]")
    overrides: list[str] = json.loads(overrides_raw) if overrides_raw else []

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
    }

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    # Print stdout/stderr for Airflow log visibility
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Extract run_id from last output line (format: "run_id=<id>")
    for line in reversed(result.stdout.strip().splitlines()):
        if line.startswith("run_id="):
            return line.split("=", 1)[1]

    return ""


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
