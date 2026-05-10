"""DAG: Weekly PyTorch documentation RAG data update.

Scrapes a fresh copy of core PyTorch documentation pages,
versions the data with DVC, and refreshes the champion
code collection in Qdrant using the production replace
update workflow from ``rag.ops.update``.

The DAG only rebuilds the **champion** alias. It does NOT touch
challenger. The production update workflow creates a successor
collection from champion `_meta`, writes fresh metadata, and then
atomically swaps the champion alias.

Schedule: @weekly
"""

from __future__ import annotations

import importlib
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
PYTORCH_OUTPUT_DIR = ASSETS_DIR / "rag_data" / "pytorch_docs"

PYTORCH_BASE_URL: str
PYTORCH_SCRAPE_DELAY_SECONDS = 1
PYTORCH_MAX_CODE_EXAMPLES = 1000
PYTORCH_PAGES: list[str]

# Paths as strings for bash commands
_pytorch_json = str(PYTORCH_OUTPUT_DIR / "pytorch_docs.json")


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
rag_data_fetchers = importlib.import_module("shared.rag_data_fetchers")

PYTORCH_BASE_URL = rag_data_fetchers.DEFAULT_PYTORCH_BASE_URL
PYTORCH_PAGES = list(rag_data_fetchers.DEFAULT_PYTORCH_PAGES)

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


def _collect_pytorch_docs() -> str:
    """Scrape a list of PyTorch doc pages and save to JSON."""
    summary = rag_data_fetchers.collect_pytorch_docs(
        base_url=PYTORCH_BASE_URL,
        page_list=PYTORCH_PAGES,
        output_dir=PYTORCH_OUTPUT_DIR,
        delay_seconds=PYTORCH_SCRAPE_DELAY_SECONDS,
        max_code_examples=PYTORCH_MAX_CODE_EXAMPLES,
    )
    return str(summary["output_file"])


def _update_pytorch_index() -> dict[str, object]:
    """Refresh the champion PyTorch docs collection using production ops."""
    from rag.ops.update import update_pytorch_docs_collection

    return update_pytorch_docs_collection(
        pytorch_docs_file=_pytorch_json,
        kb="pytorch_docs",
        alias="champion",
    )


def _version_pytorch_dataset() -> dict[str, str | bool]:
    """Persist PyTorch docs dataset pointer updates through a temp clone."""
    return sync_dvc_dataset_via_temp_clone(
        repo_root=PROJECT_ROOT,
        dataset_rel_path=Path("assets/rag_data/pytorch_docs"),
        commit_message="chore(data-sync): refresh pytorch docs rag dataset",
        pr_title="chore(data-sync): refresh pytorch docs rag dataset",
        pr_body=(
            "Automated PyTorch docs RAG dataset refresh produced by the Airflow weekly sync DAG."
        ),
    )


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="pytorch_docs_rag_update",
    default_args=default_args,
    description="Weekly scrape of PyTorch docs, DVC versioning, and RAG index update",
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["rag", "pytorch", "data"],
) as dag:
    scrape = PythonOperator(
        task_id="scrape_pytorch_docs",
        python_callable=_collect_pytorch_docs,
    )

    dvc_version = PythonOperator(
        task_id="dvc_version_pytorch",
        python_callable=_version_pytorch_dataset,
    )

    # Champion-only rebuild via the replace strategy.
    # The build script reads _meta from the current champion, creates a
    # new timestamped collection with a staging alias, builds the index,
    # and then swaps the champion alias.
    build_index = PythonOperator(
        task_id="build_pytorch_index",
        python_callable=_update_pytorch_index,
    )

    scrape >> dvc_version >> build_index
