"""DAG: Daily ArXiv RAG data update.

Downloads the latest ArXiv papers in target categories,
versions the data with DVC, and refreshes the champion
chat collection in Qdrant using the production incremental
update workflow from ``rag.ops.update``.

Schedule: @daily
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
ASSETS_DIR = PROJECT_ROOT / "assets"
ARXIV_OUTPUT_DIR = ASSETS_DIR / "rag_data" / "arxiv"

ARXIV_CATEGORIES: list[str] = ["cs.LG", "cs.AI"]
ARXIV_MAX_RESULTS: int = 100

# Paths as strings for bash commands
_arxiv_json = str(ARXIV_OUTPUT_DIR / "arxiv_papers.json")


def _bootstrap_project_imports() -> None:
    """Ensure task-time imports can resolve the repo's src/ layout."""
    for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_bootstrap_project_imports()
sync_dvc_dataset_via_temp_clone = importlib.import_module(
    "shared.airflow_git_sync"
).sync_dvc_dataset_via_temp_clone

# ---------------------------------------------------------------------------
# Default DAG arguments
# ---------------------------------------------------------------------------

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
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


def _update_arxiv_index() -> dict[str, object]:
    """Refresh the champion ArXiv collection using production ops."""
    from rag.ops.update import update_arxiv_collection

    return update_arxiv_collection(
        arxiv_file=_arxiv_json,
        kb="arxiv",
        alias="champion",
    )


def _version_arxiv_dataset() -> dict[str, str | bool]:
    """Persist ArXiv dataset pointer updates through a temp clone."""
    return sync_dvc_dataset_via_temp_clone(
        repo_root=PROJECT_ROOT,
        dataset_rel_path=Path("assets/rag_data/arxiv"),
        commit_message="chore(data-sync): refresh arxiv rag dataset",
        pr_title="chore(data-sync): refresh arxiv rag dataset",
        pr_body=("Automated ArXiv RAG dataset refresh produced by the Airflow daily sync DAG."),
    )


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

    dvc_version = PythonOperator(
        task_id="dvc_version_arxiv",
        python_callable=_version_arxiv_dataset,
    )

    # Daily updates target only the champion alias.
    build_index = PythonOperator(
        task_id="build_arxiv_index",
        python_callable=_update_arxiv_index,
    )

    download >> dvc_version >> build_index
