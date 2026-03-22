"""DAG: Daily ArXiv RAG data update.

Downloads the latest ArXiv papers in target categories,
versions the data with DVC, and rebuilds the chat vector
index in Qdrant using the **incremental** strategy.

The build script resolves all aliases for the ``arxiv`` KB,
reads ``_meta`` from each to determine the build config, and
upserts new papers into every active collection.

Schedule: @daily
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
ASSETS_DIR = PROJECT_ROOT / "assets"
ARXIV_OUTPUT_DIR = ASSETS_DIR / "rag_data" / "arxiv"

ARXIV_CATEGORIES: list[str] = ["cs.LG", "cs.AI"]
ARXIV_MAX_RESULTS: int = 100

# Paths as strings for bash commands
_project_root = str(PROJECT_ROOT)
_arxiv_dir = str(ARXIV_OUTPUT_DIR)
_arxiv_rel = str(ARXIV_OUTPUT_DIR.relative_to(PROJECT_ROOT))
_arxiv_json = str(ARXIV_OUTPUT_DIR / "arxiv_papers.json")
_build_script = str(PROJECT_ROOT / "experiments" / "scripts" / "rag_data" / "build_vector_index.py")

# ---------------------------------------------------------------------------
# Default DAG arguments
# ---------------------------------------------------------------------------

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def _download_arxiv_papers() -> str:
    """Download papers from arXiv and save metadata + abstracts to JSON."""
    import arxiv  # noqa: E402 -- delay import so DAG parses even when lib is missing

    ARXIV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    query = " OR ".join(f"cat:{cat}" for cat in ARXIV_CATEGORIES)
    print(f"Searching arXiv: {query}  (max {ARXIV_MAX_RESULTS})")

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=ARXIV_MAX_RESULTS,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: list[dict] = []
    for i, result in enumerate(client.results(search), 1):
        papers.append(
            {
                "arxiv_id": result.entry_id.split("/")[-1],
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary,
                "published": result.published.isoformat(),
                "updated": result.updated.isoformat(),
                "categories": result.categories,
                "primary_category": result.primary_category,
                "pdf_url": result.pdf_url,
            }
        )
        if i % 20 == 0:
            print(f"  {i} papers fetched ...")

    output_file = ARXIV_OUTPUT_DIR / "arxiv_papers.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"Downloaded {len(papers)} papers -> {output_file}")
    return str(output_file)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="arxiv_rag_update",
    default_args=default_args,
    description="Daily download of ArXiv papers, DVC versioning, and RAG index update",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["rag", "arxiv", "data"],
) as dag:
    download = PythonOperator(
        task_id="download_arxiv_papers",
        python_callable=_download_arxiv_papers,
    )

    dvc_version = BashOperator(
        task_id="dvc_version_arxiv",
        bash_command=f"cd {_project_root} && dvc add {_arxiv_rel} && dvc push ",
    )

    # The build script handles all aliases for the arxiv KB automatically.
    build_index = BashOperator(
        task_id="build_arxiv_index",
        bash_command=(
            f"cd {_project_root} && "
            f"PYTHONPATH={_project_root}/src:$PYTHONPATH "
            f"python {_build_script} "
            f"--arxiv-file {_arxiv_json} "
            "--qdrant-host $QDRANT_HOST "
            "--qdrant-port $QDRANT_PORT "
            "--embedding-model $EMBEDDING_MODEL "
            "--task chat "
            "--kb arxiv "
        ),
    )

    download >> dvc_version >> build_index
