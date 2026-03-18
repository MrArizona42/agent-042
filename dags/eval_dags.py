"""Airflow DAGs for evaluation workflows (Stages 1–3).

Stage 1: Base LLM evaluation (no RAG, no LoRA).
Stage 2: Base LLM + RAG evaluation.
Stage 3: Base LLM + RAG + LoRA evaluation (full Cartesian product).

Each DAG is one eval-suite = one ``(task, dataset, metric)`` triple.
DAG naming: ``eval_{task}_{dataset}_{metric}``,
or ``eval_retrieval_{kb}_{dataset}_{metric}`` for retrieval-only evals.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/airflow/project"))
_project_root = str(PROJECT_ROOT)
_runner = str(PROJECT_ROOT / "experiments" / "scripts" / "eval" / "runner.py")

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _eval_bash(
    task: str, dataset: str, metric: str,
    rag_aliases: str, lora_aliases: str, kb: str = "",
) -> str:
    """Build bash command string for the eval runner."""
    kb_flag = f"--kb {kb} " if kb else ""
    return (
        f"cd {_project_root} && "
        f"PYTHONPATH={_project_root}/src:{_project_root}:$PYTHONPATH "
        f"python {_runner} "
        f"--task {task} --dataset {dataset} --metric {metric} "
        f"{kb_flag}"
        f"--rag-aliases {rag_aliases} "
        f"--lora-aliases {lora_aliases} "
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
        BashOperator(
            task_id="run_eval",
            bash_command=_eval_bash(
                "chat", "hotpotqa", _metric,
                "{{ params.rag_aliases }}",
                "{{ params.lora_aliases }}",
            ),
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
        BashOperator(
            task_id="run_eval",
            bash_command=_eval_bash(
                "chat", "nq", _metric,
                "{{ params.rag_aliases }}",
                "{{ params.lora_aliases }}",
            ),
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
        BashOperator(
            task_id="run_eval",
            bash_command=_eval_bash(
                "summarize", "arxiv_summarization", _metric,
                "none",
                "{{ params.lora_aliases }}",
            ),
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
        BashOperator(
            task_id="run_eval",
            bash_command=_eval_bash(
                "code", "humaneval", _metric,
                "{{ params.rag_aliases }}",
                "{{ params.lora_aliases }}",
            ),
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
        BashOperator(
            task_id="run_eval",
            bash_command=_eval_bash(
                "retrieval", "beir_scifact", _metric,
                "{{ params.rag_aliases }}",
                "none",
                kb="arxiv",
            ),
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
        BashOperator(
            task_id="run_eval",
            bash_command=_eval_bash(
                "retrieval", "beir_nfcorpus", _metric,
                "{{ params.rag_aliases }}",
                "none",
                kb="arxiv",
            ),
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
        BashOperator(
            task_id="run_eval",
            bash_command=_eval_bash(
                "retrieval", "msmarco", _metric,
                "{{ params.rag_aliases }}",
                "none",
                kb="pytorch_docs",
            ),
        )

# =========================================================================
# Stage 3: Base LLM + RAG + LoRA (full matrix via DAG params)
# =========================================================================
# The same Stage 1 DAGs are reused with different params.  Stage 3 adds
# no new DAG definitions — operators already accept --rag-aliases and
# --lora-aliases via Airflow params so they can run full Cartesian product
# comparisons.  This keeps the number of DAGs minimal while supporting all
# three stages through parameterised triggering.
