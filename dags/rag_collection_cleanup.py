"""DAG: Daily Qdrant orphan-collection cleanup.

Deletes Qdrant collections that are no longer pointed to by any alias
and whose creation timestamp is older than ``RETENTION_DAYS`` (7).

Legacy collections that do not follow the ``rag__{kb}__{timestamp}`` naming
convention are on an explicit skip-list and are never deleted.

Schedule: @daily
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QDRANT_HOST = os.environ["PLATFORM__QDRANT_HOST"]
QDRANT_PORT = int(os.environ["PLATFORM__QDRANT_PORT"])
RETENTION_DAYS = 7

# Legacy collections that don't follow the rag__{kb}__{timestamp} naming convention.
# These pre-date the alias-based lifecycle and are never deleted by the
# cleanup DAG.  They will be migrated during rollout; update this list
# if new legacy collections are discovered.
SKIP_LIST: set[str] = {"chat_documents", "code_documents"}

# Regex: rag__{kb_id}__{YYYYMMDD}_{HHMMSS}
_TS_PATTERN = re.compile(r"^rag__(.+)__(\d{8}_\d{6})$")

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
# Task callable
# ---------------------------------------------------------------------------


def _cleanup_orphan_collections() -> None:
    """Delete orphan Qdrant collections older than RETENTION_DAYS."""
    from qdrant_client import QdrantClient

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Build a set of collections that have at least one alias
    aliased_collections: set[str] = set()
    for alias in client.get_aliases().aliases:
        aliased_collections.add(alias.collection_name)

    # List all collections
    all_collections = {c.name for c in client.get_collections().collections}

    # Find orphans
    orphans = all_collections - aliased_collections
    cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)

    deleted = 0
    for name in sorted(orphans):
        if name in SKIP_LIST:
            print(f"  SKIP (legacy): {name}")
            continue

        match = _TS_PATTERN.match(name)
        if not match:
            print(f"  SKIP (no timestamp): {name}")
            continue

        ts_str = match.group(2)
        try:
            created = datetime.strptime(ts_str, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"  SKIP (bad timestamp): {name}")
            continue

        if created < cutoff:
            print(f"  DELETE: {name} (created {created.isoformat()})")
            client.delete_collection(name)
            deleted += 1
        else:
            print(f"  KEEP (recent): {name} (created {created.isoformat()})")

    print(f"\nCleanup complete: {deleted} collection(s) deleted.")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="rag_collection_cleanup",
    default_args=default_args,
    description="Daily cleanup of orphaned Qdrant collections older than 7 days",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["rag", "cleanup", "maintenance"],
) as dag:
    cleanup = PythonOperator(
        task_id="cleanup_orphan_collections",
        python_callable=_cleanup_orphan_collections,
    )
