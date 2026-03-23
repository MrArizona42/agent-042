"""Airflow DAGs for evaluation workflows.

Each DAG represents a unique ``(task, dataset)`` evaluation suite.
The **metric** to compute and the **alias** configuration are selected
at trigger time via Airflow ``Params`` (rendered as dropdowns in the UI).

Two-step execution:
    1. ``fetch_predictions`` — calls the gateway / retrieval system.
    2. ``calculate_metrics`` — computes the selected metric on the
       pre-fetched predictions and logs results to the database.

Predictions are handed between tasks via a temporary JSON file to avoid
XCom size limits.

For custom parameter values that are not in the dropdown lists, put
a JSON string into the ``custom_params`` field when triggering the DAG.
Example::

    {"metric": "my_custom_metric", "rag_aliases": ["my_alias"]}

Zero retries.  No silent fallback to default values — if any required
parameter is missing or invalid the DAG fails immediately.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
}

# ---------------------------------------------------------------------------
# Eval suite definitions — one DAG per (task, dataset) combo
# ---------------------------------------------------------------------------

_EVAL_SUITES: list[dict] = [
    {
        "dag_id": "eval_chat_hotpotqa",
        "task": "chat",
        "dataset": "hotpotqa",
        "metrics": ["relevance", "correctness", "bertscore_f1", "rouge_l"],
        "kb": None,
        "description": "Eval: chat on HotpotQA",
        "tags": ["eval", "chat"],
    },
    {
        "dag_id": "eval_chat_nq",
        "task": "chat",
        "dataset": "nq",
        "metrics": ["relevance", "correctness", "bertscore_f1", "rouge_l"],
        "kb": None,
        "description": "Eval: chat on Natural Questions",
        "tags": ["eval", "chat"],
    },
    {
        "dag_id": "eval_summarization_arxiv",
        "task": "summarize",
        "dataset": "arxiv_summarization",
        "metrics": ["faithfulness", "coverage", "bertscore_f1", "rouge_l"],
        "kb": None,
        "description": "Eval: summarization on ArXiv",
        "tags": ["eval", "summarize"],
    },
    {
        "dag_id": "eval_code_humaneval",
        "task": "code",
        "dataset": "humaneval",
        "metrics": ["pass_at_1", "executable_rate"],
        "kb": None,
        "description": "Eval: code generation on HumanEval",
        "tags": ["eval", "code"],
    },
    {
        "dag_id": "eval_retrieval_arxiv_beir_scifact",
        "task": "retrieval",
        "dataset": "beir_scifact",
        "metrics": ["recall_at_10", "ndcg_at_10"],
        "kb": "arxiv",
        "description": "Eval: retrieval on BEIR-SciFact (arxiv KB)",
        "tags": ["eval", "retrieval"],
    },
    {
        "dag_id": "eval_retrieval_arxiv_beir_nfcorpus",
        "task": "retrieval",
        "dataset": "beir_nfcorpus",
        "metrics": ["recall_at_10", "ndcg_at_10"],
        "kb": "arxiv",
        "description": "Eval: retrieval on BEIR-NFCorpus (arxiv KB)",
        "tags": ["eval", "retrieval"],
    },
    {
        "dag_id": "eval_retrieval_pytorch_msmarco",
        "task": "retrieval",
        "dataset": "msmarco",
        "metrics": ["recall_at_10", "ndcg_at_10"],
        "kb": "pytorch_docs",
        "description": "Eval: retrieval on MS MARCO (pytorch_docs KB)",
        "tags": ["eval", "retrieval"],
    },
]


# ---------------------------------------------------------------------------
# Parameter resolution — no silent fallbacks
# ---------------------------------------------------------------------------


def _resolve_params(context: dict) -> dict:
    """Extract trigger params, applying ``custom_params`` overrides.

    Raises immediately if any required parameter is missing or empty.
    Never falls back to implicit defaults.
    """
    params = context["params"]

    # Parse custom_params — fail hard on malformed JSON
    custom_raw = params["custom_params"]
    custom: dict = json.loads(custom_raw) if custom_raw else {}

    metric = custom["metric"] if "metric" in custom else params["metric"]
    rag_aliases = custom["rag_aliases"] if "rag_aliases" in custom else params["rag_aliases"]
    lora_aliases = custom["lora_aliases"] if "lora_aliases" in custom else params["lora_aliases"]

    # Normalise aliases to list[str]
    if isinstance(rag_aliases, str):
        rag_aliases = [a.strip() for a in rag_aliases.split(",") if a.strip()]
    if isinstance(lora_aliases, str):
        lora_aliases = [a.strip() for a in lora_aliases.split(",") if a.strip()]

    # Strict validation — fail if anything is absent
    if not metric:
        raise ValueError("Required parameter 'metric' is empty")
    if not rag_aliases:
        raise ValueError("Required parameter 'rag_aliases' is empty")
    if not lora_aliases:
        raise ValueError("Required parameter 'lora_aliases' is empty")

    return {
        "metric": metric,
        "rag_aliases": rag_aliases,
        "lora_aliases": lora_aliases,
    }


# ---------------------------------------------------------------------------
# Airflow task callables
# ---------------------------------------------------------------------------


def _fetch_predictions_task(
    eval_task: str,
    dataset: str,
    kb: str | None,
    **context: object,
) -> str:
    """Airflow task: call the model/retrieval system and save predictions."""
    log = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    resolved = _resolve_params(context)

    log.info(
        "fetch_predictions: task=%s dataset=%s rag=%s lora=%s",
        eval_task,
        dataset,
        resolved["rag_aliases"],
        resolved["lora_aliases"],
    )

    from experiments.scripts.eval.runner import fetch_predictions

    prediction_data = fetch_predictions(
        task=eval_task,
        dataset_name=dataset,
        kb_name=kb,
        rag_aliases=resolved["rag_aliases"],
        lora_aliases=resolved["lora_aliases"],
    )

    # Persist to file (avoids XCom size limits)
    run_id = str(context["run_id"])
    safe_run_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_id)
    output_path = f"/tmp/eval_predictions_{safe_run_id}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prediction_data, f, ensure_ascii=False)

    log.info(
        "Predictions saved to %s (%d bundles)",
        output_path,
        len(prediction_data["bundles"]),
    )
    return output_path  # auto-pushed to XCom


def _calculate_metrics_task(
    **context: object,
) -> None:
    """Airflow task: dispatch metric computation to the eval-worker via Celery.

    The actual ``bert-score`` / ``rouge`` calculation runs inside the
    eval-worker container (which carries ``torch`` CPU), keeping the
    Airflow image lightweight.
    """
    log = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    predictions_path = context["ti"].xcom_pull(task_ids="fetch_predictions")
    if not predictions_path:
        raise RuntimeError("No predictions file path received from fetch_predictions task")

    with open(predictions_path, encoding="utf-8") as f:
        prediction_data = json.load(f)

    resolved = _resolve_params(context)
    metric = resolved["metric"]

    log.info("calculate_metrics: dispatching metric=%s to eval-worker queue", metric)

    from celery import Celery

    broker_url = os.environ.get(
        "CELERY_BROKER_URL",
        "amqp://guest:guest@rabbitmq:5672//",
    )
    _celery = Celery(broker=broker_url, backend="rpc://")

    async_result = _celery.send_task(
        "eval_worker.tasks.calculate_metrics_task",
        kwargs={"metric": metric, "prediction_data": prediction_data},
        queue="eval",
    )

    # Block until the eval-worker finishes (up to 30 min)
    rows = async_result.get(timeout=1800)

    log.info("Metrics complete: %d rows", len(rows))
    for row in rows:
        log.info(
            "  %s/%s metric=%s rag=%s lora=%s → %.4f",
            row["task"],
            row["dataset_name"],
            row["metric_name"],
            row.get("rag_alias") or "none",
            row.get("lora_alias") or "none",
            row["metric_value"],
        )

    # Clean up the temporary predictions file
    Path(predictions_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# DAG generation
# ---------------------------------------------------------------------------

for _suite in _EVAL_SUITES:
    _dag_id = _suite["dag_id"]
    _task = _suite["task"]
    _dataset = _suite["dataset"]
    _metrics = _suite["metrics"]
    _kb = _suite["kb"]

    _dag = DAG(
        dag_id=_dag_id,
        default_args=default_args,
        description=_suite["description"],
        schedule=None,
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=_suite["tags"],
        params={
            "metric": Param(
                type="string",
                enum=_metrics,
                description=(
                    f"Metric to compute. Valid: {', '.join(_metrics)}. "
                    "For values outside this list, use custom_params."
                ),
            ),
            "rag_aliases": Param(
                type="string",
                enum=["none", "champion", "challenger", "champion,challenger"],
                description=(
                    "RAG aliases to evaluate (comma-separated). "
                    "For custom aliases, use custom_params."
                ),
            ),
            "lora_aliases": Param(
                type="string",
                enum=["none", "champion", "challenger", "champion,challenger"],
                description=(
                    "LoRA aliases to evaluate (comma-separated). "
                    "For custom aliases, use custom_params."
                ),
            ),
            "custom_params": Param(
                default="",
                type=["string", "null"],
                description=(
                    'Optional JSON overrides. Example: '
                    '{"metric": "my_metric", "rag_aliases": ["a1", "a2"]}'
                ),
            ),
        },
    )

    with _dag:
        _fetch = PythonOperator(
            task_id="fetch_predictions",
            python_callable=_fetch_predictions_task,
            op_kwargs={"eval_task": _task, "dataset": _dataset, "kb": _kb},
        )

        _calc = PythonOperator(
            task_id="calculate_metrics",
            python_callable=_calculate_metrics_task,
        )

        _fetch >> _calc

    globals()[_dag_id] = _dag
