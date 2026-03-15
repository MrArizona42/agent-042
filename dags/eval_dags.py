"""Airflow DAGs for evaluation workflows (Stages 1–3).

Stage 1: Base LLM evaluation (no RAG, no LoRA).
Stage 2: Base LLM + RAG evaluation.
Stage 3: Base LLM + RAG + LoRA evaluation (full Cartesian product).

DAG naming: ``eval_{task}_{dataset}`` for generation evals,
``eval_retrieval_{kb}_{dataset}`` for retrieval-only evals.
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


def _eval_bash(task: str, dataset: str, rag_aliases: str, lora_aliases: str, kb: str = "") -> str:
    """Build bash command string for the eval runner."""
    kb_flag = f"--kb {kb} " if kb else ""
    return (
        f"cd {_project_root} && "
        f"PYTHONPATH={_project_root}/src:{_project_root}:$PYTHONPATH "
        f"python {_runner} "
        f"--task {task} --dataset {dataset} "
        f"{kb_flag}"
        f"--rag-aliases {rag_aliases} "
        f"--lora-aliases {lora_aliases} "
    )


# =========================================================================
# Stage 1: Base LLM (no RAG, no LoRA)
# =========================================================================

with DAG(
    dag_id="eval_chat_hotpotqa",
    default_args=default_args,
    description="Eval: chat on HotpotQA (base model)",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["eval", "chat", "stage1"],
    params={
        "rag_aliases": "none",
        "lora_aliases": "none",
    },
) as dag_chat_hotpotqa:
    BashOperator(
        task_id="run_eval",
        bash_command=_eval_bash(
            "chat", "hotpotqa",
            "{{ params.rag_aliases }}",
            "{{ params.lora_aliases }}",
        ),
    )

with DAG(
    dag_id="eval_chat_nq",
    default_args=default_args,
    description="Eval: chat on Natural Questions (base model)",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["eval", "chat", "stage1"],
    params={
        "rag_aliases": "none",
        "lora_aliases": "none",
    },
) as dag_chat_nq:
    BashOperator(
        task_id="run_eval",
        bash_command=_eval_bash(
            "chat", "nq",
            "{{ params.rag_aliases }}",
            "{{ params.lora_aliases }}",
        ),
    )

with DAG(
    dag_id="eval_summarization_arxiv",
    default_args=default_args,
    description="Eval: summarization on ArXiv (base model)",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["eval", "summarize", "stage1"],
    params={
        "lora_aliases": "none",
    },
) as dag_summarize:
    BashOperator(
        task_id="run_eval",
        bash_command=_eval_bash(
            "summarize", "arxiv_summarization",
            "none",
            "{{ params.lora_aliases }}",
        ),
    )

with DAG(
    dag_id="eval_code_humaneval",
    default_args=default_args,
    description="Eval: code generation on HumanEval (base model)",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["eval", "code", "stage1"],
    params={
        "rag_aliases": "none",
        "lora_aliases": "none",
    },
) as dag_code:
    BashOperator(
        task_id="run_eval",
        bash_command=_eval_bash(
            "code", "humaneval",
            "{{ params.rag_aliases }}",
            "{{ params.lora_aliases }}",
        ),
    )

# =========================================================================
# Stage 2: Base LLM + RAG
# =========================================================================

with DAG(
    dag_id="eval_retrieval_arxiv_beir_scifact",
    default_args=default_args,
    description="Eval: retrieval-only on BEIR-SciFact (arxiv KB config)",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["eval", "retrieval", "stage2"],
    params={
        "rag_aliases": "champion",
    },
) as dag_ret_arxiv_scifact:
    BashOperator(
        task_id="run_eval",
        bash_command=_eval_bash(
            "retrieval", "beir_scifact",
            "{{ params.rag_aliases }}",
            "none",
            kb="arxiv",
        ),
    )

with DAG(
    dag_id="eval_retrieval_arxiv_beir_nfcorpus",
    default_args=default_args,
    description="Eval: retrieval-only on BEIR-NFCorpus (arxiv KB config)",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["eval", "retrieval", "stage2"],
    params={
        "rag_aliases": "champion",
    },
) as dag_ret_arxiv_nfcorpus:
    BashOperator(
        task_id="run_eval",
        bash_command=_eval_bash(
            "retrieval", "beir_nfcorpus",
            "{{ params.rag_aliases }}",
            "none",
            kb="arxiv",
        ),
    )

with DAG(
    dag_id="eval_retrieval_pytorch_msmarco",
    default_args=default_args,
    description="Eval: retrieval-only on MS MARCO (pytorch_docs KB config)",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["eval", "retrieval", "stage2"],
    params={
        "rag_aliases": "champion",
    },
) as dag_ret_pytorch_msmarco:
    BashOperator(
        task_id="run_eval",
        bash_command=_eval_bash(
            "retrieval", "msmarco",
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
