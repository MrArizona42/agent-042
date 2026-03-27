"""DAG: Weekly PyTorch documentation RAG data update.

Scrapes a fresh copy of core PyTorch documentation pages,
versions the data with DVC, and rebuilds the code vector
index in Qdrant using the **replace** strategy.

The DAG only rebuilds the **champion** alias.  It does NOT touch
challenger.  The build script creates a new timestamped collection,
writes ``_meta``, creates a staging alias, builds the index, and
then atomically swaps the champion alias.

Schedule: @weekly
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])
ASSETS_DIR = PROJECT_ROOT / "assets"
PYTORCH_OUTPUT_DIR = ASSETS_DIR / "rag_data" / "pytorch_docs"

PYTORCH_BASE_URL = "https://pytorch.org/docs/stable/"
PYTORCH_SCRAPE_DELAY_SECONDS = 1
PYTORCH_MAX_CODE_EXAMPLES = 1000
PYTORCH_PAGES: list[str] = [
    "generated/torch.nn.Module.html",
    "generated/torch.Tensor.html",
    "generated/torch.nn.Linear.html",
    "generated/torch.nn.Conv2d.html",
    "generated/torch.nn.functional.relu.html",
    "generated/torch.optim.Adam.html",
    "generated/torch.optim.SGD.html",
    "generated/torch.nn.CrossEntropyLoss.html",
    "generated/torch.nn.MSELoss.html",
    "generated/torch.autograd.backward.html",
    "tensors.html",
    "autograd.html",
    "nn.html",
    "optim.html",
    "torch.html",
]

# Paths as strings for bash commands
_project_root = str(PROJECT_ROOT)
_pytorch_dir = str(PYTORCH_OUTPUT_DIR)
_pytorch_rel = str(PYTORCH_OUTPUT_DIR.relative_to(PROJECT_ROOT))
_pytorch_json = str(PYTORCH_OUTPUT_DIR / "pytorch_docs.json")
_build_script = str(
    PROJECT_ROOT / "experiments" / "scripts" / "rag_data" / "build_pytorch_docs_index.py"
)

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


def _scrape_pytorch_doc_page(url: str) -> dict:
    """Scrape a single PyTorch documentation page."""
    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    title_tag = soup.find("h1")
    title_text = title_tag.get_text(strip=True) if title_tag else "Untitled"

    content_div = soup.find("div", {"role": "main"}) or soup.find("article")
    if content_div:
        for tag in content_div.find_all(["nav", "footer", "script", "style"]):
            tag.decompose()
        content = content_div.get_text(separator="\n", strip=True)
    else:
        content = ""

    code_blocks = soup.find_all("code") or soup.find_all("pre")
    code_examples = [b.get_text(strip=True) for b in code_blocks[:PYTORCH_MAX_CODE_EXAMPLES]]

    return {
        "url": url,
        "title": title_text,
        "content": content,
        "code_examples": code_examples,
        "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _collect_pytorch_docs() -> str:
    """Scrape a list of PyTorch doc pages and save to JSON."""
    PYTORCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pages: list[dict] = []
    for i, page_path in enumerate(PYTORCH_PAGES, 1):
        url = urljoin(PYTORCH_BASE_URL, page_path)
        print(f"[{i}/{len(PYTORCH_PAGES)}] {url}")
        try:
            pages.append(_scrape_pytorch_doc_page(url))
            time.sleep(PYTORCH_SCRAPE_DELAY_SECONDS)  # polite rate-limiting
        except Exception as exc:
            print(f"  Warning: {exc}")

    output_file = PYTORCH_OUTPUT_DIR / "pytorch_docs.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2, ensure_ascii=False)

    print(f"Scraped {len(pages)} pages -> {output_file}")
    return str(output_file)


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

    dvc_version = BashOperator(
        task_id="dvc_version_pytorch",
        bash_command=f"cd {_project_root} && dvc add {_pytorch_rel} && dvc push ",
    )

    # Champion-only rebuild via the replace strategy.
    # The build script reads _meta from the current champion, creates a
    # new timestamped collection with a staging alias, builds the index,
    # and then swaps the champion alias.
    build_index = BashOperator(
        task_id="build_pytorch_index",
        bash_command=(
            f"cd {_project_root} && "
            f"PYTHONPATH={_project_root}/src:$PYTHONPATH "
            f"python {_build_script} "
            f"--pytorch_docs_file {_pytorch_json} "
            "--qdrant_host $QDRANT_HOST "
            "--qdrant_port $QDRANT_PORT "
            "--embedding_model $EMBEDDING_MODEL "
            "--kb pytorch_docs "
            "--alias champion "
        ),
    )

    scrape >> dvc_version >> build_index
