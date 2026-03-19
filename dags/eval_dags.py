"""Airflow DAGs for evaluation workflows (Stages 1–3).

Stage 1: Base LLM evaluation (no RAG, no LoRA).
Stage 2: Base LLM + RAG evaluation.
Stage 3: Base LLM + RAG + LoRA evaluation (full Cartesian product).

Each DAG is one eval-suite = one ``(task, dataset, metric)`` triple.
DAG naming: ``eval_{task}_{dataset}_{metric}``,
or ``eval_retrieval_{kb}_{dataset}_{metric}`` for retrieval-only evals.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/airflow/project"))

# Ensure project sources are importable inside the Airflow process.
for _p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _run_eval(
    task: str,
    dataset: str,
    metric: str,
    kb: str | None = None,
    **context: object,
) -> None:
    """PythonOperator callable — delegates to ``runner.run_eval``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    from experiments.scripts.eval.runner import run_eval

    params = context.get("params", {})
    rag_aliases = [a.strip() for a in str(params.get("rag_aliases", "none")).split(",")]
    lora_aliases = [a.strip() for a in str(params.get("lora_aliases", "none")).split(",")]

    rows = run_eval(
        task=task,
        dataset_name=dataset,
        metric=metric,
        kb_name=kb,
        rag_aliases=rag_aliases,
        lora_aliases=lora_aliases,
    )

    log = logging.getLogger(__name__)
    log.info("Eval complete: %d metric rows", len(rows))
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


# =========================================================================
# Stage 1: Base LLM (no RAG, no LoRA) — Chat
# =========================================================================

# --- Chat / HotpotQA ---
for _metric in ("relevance", "correctness", "bertscore_f1", "rouge_l"):
    _dag_id = f"eval_chat_hotpotqa_{_metric}"
    with DAG(
        dag_id=_dag_id,
        default_args=default_args,
        description=f"Eval: chat on HotpotQA — {_metric}",
        schedule=None,
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=["eval", "chat", _metric, "stage1"],
        params={
            "rag_aliases": "none",
            "lora_aliases": "none",
        },
    ):
        PythonOperator(
            task_id="run_eval",
            python_callable=_run_eval,
            op_kwargs={"task": "chat", "dataset": "hotpotqa", "metric": _metric},
        )

# --- Chat / Natural Questions ---
for _metric in ("relevance", "correctness", "bertscore_f1", "rouge_l"):
    _dag_id = f"eval_chat_nq_{_metric}"
    with DAG(
        dag_id=_dag_id,
        default_args=default_args,
        description=f"Eval: chat on Natural Questions — {_metric}",
        schedule=None,
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=["eval", "chat", _metric, "stage1"],
        params={
            "rag_aliases": "none",
            "lora_aliases": "none",
        },
    ):
        PythonOperator(
            task_id="run_eval",
            python_callable=_run_eval,
            op_kwargs={"task": "chat", "dataset": "nq", "metric": _metric},
        )

# =========================================================================
# Stage 1: Base LLM (no RAG, no LoRA) — Summarization
# =========================================================================

for _metric in ("faithfulness", "coverage", "bertscore_f1", "rouge_l"):
    _dag_id = f"eval_summarization_arxiv_{_metric}"
    with DAG(
        dag_id=_dag_id,
        default_args=default_args,
        description=f"Eval: summarization on ArXiv — {_metric}",
        schedule=None,
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=["eval", "summarize", _metric, "stage1"],
        params={
            "lora_aliases": "none",
        },
    ):
        PythonOperator(
            task_id="run_eval",
            python_callable=_run_eval,
            op_kwargs={"task": "summarize", "dataset": "arxiv_summarization", "metric": _metric},
        )

# =========================================================================
# Stage 1: Base LLM (no RAG, no LoRA) — Code
# =========================================================================

for _metric in ("pass_at_1", "executable_rate"):
    _dag_id = f"eval_code_humaneval_{_metric}"
    with DAG(
        dag_id=_dag_id,
        default_args=default_args,
        description=f"Eval: code generation on HumanEval — {_metric}",
        schedule=None,
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=["eval", "code", _metric, "stage1"],
        params={
            "rag_aliases": "none",
            "lora_aliases": "none",
        },
    ):
        PythonOperator(
            task_id="run_eval",
            python_callable=_run_eval,
            op_kwargs={"task": "code", "dataset": "humaneval", "metric": _metric},
        )

# =========================================================================
# Stage 2: Base LLM + RAG — Retrieval-only
# =========================================================================

for _metric in ("recall_at_10", "ndcg_at_10"):
    # --- arxiv / BEIR-SciFact ---
    with DAG(
        dag_id=f"eval_retrieval_arxiv_beir_scifact_{_metric}",
        default_args=default_args,
        description=f"Eval: retrieval on BEIR-SciFact (arxiv KB) — {_metric}",
        schedule=None,
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=["eval", "retrieval", _metric, "stage2"],
        params={
            "rag_aliases": "champion",
        },
    ):
        PythonOperator(
            task_id="run_eval",
            python_callable=_run_eval,
            op_kwargs={
                "task": "retrieval",
                "dataset": "beir_scifact",
                "metric": _metric,
                "kb": "arxiv",
            },
        )

    # --- arxiv / BEIR-NFCorpus ---
    with DAG(
        dag_id=f"eval_retrieval_arxiv_beir_nfcorpus_{_metric}",
        default_args=default_args,
        description=f"Eval: retrieval on BEIR-NFCorpus (arxiv KB) — {_metric}",
        schedule=None,
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=["eval", "retrieval", _metric, "stage2"],
        params={
            "rag_aliases": "champion",
        },
    ):
        PythonOperator(
            task_id="run_eval",
            python_callable=_run_eval,
            op_kwargs={
                "task": "retrieval",
                "dataset": "beir_nfcorpus",
                "metric": _metric,
                "kb": "arxiv",
            },
        )

    # --- pytorch_docs / MS MARCO ---
    with DAG(
        dag_id=f"eval_retrieval_pytorch_msmarco_{_metric}",
        default_args=default_args,
        description=f"Eval: retrieval on MS MARCO (pytorch_docs KB) — {_metric}",
        schedule=None,
        start_date=datetime(2025, 1, 1),
        catchup=False,
        tags=["eval", "retrieval", _metric, "stage2"],
        params={
            "rag_aliases": "champion",
        },
    ):
        PythonOperator(
            task_id="run_eval",
            python_callable=_run_eval,
            op_kwargs={
                "task": "retrieval",
                "dataset": "msmarco",
                "metric": _metric,
                "kb": "pytorch_docs",
            },
        )

# =========================================================================
# Stage 3: Base LLM + RAG + LoRA (full matrix via DAG params)
# =========================================================================
# The same Stage 1 DAGs are reused with different params.  Stage 3 adds
# no new DAG definitions — operators already accept --rag-aliases and
# --lora-aliases via Airflow params so they can run full Cartesian product
# comparisons.  This keeps the number of DAGs minimal while supporting all
# three stages through parameterised triggering.
