"""DAG: deployment/release-aware Qdrant collection cleanup.

Liveness is decided from Postgres (`rag_releases` / `rag_alias_deployments`),
never from whether a Qdrant alias happens to still point at a collection --
Qdrant aliases are a mirror of applied state, not the source of truth. A
release is marked retired in Postgres before its collection is deleted, and
its immutable release manifest on disk is left alone.

Collections with no matching `rag_releases` row at all are left untouched
and only logged: a release row is inserted only after a build finishes, so
those may belong to a build that is still running, or may predate this
migration. This DAG has no reliable way to tell those apart from a
genuinely abandoned collection, so it never deletes them.

Schedule: @daily
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator

QDRANT_HOST = os.environ["NETWORK__QDRANT_HTTP__INTERNAL_HOST"]
QDRANT_PORT = int(os.environ["NETWORK__QDRANT_HTTP__INTERNAL_PORT"])

# Active/pending deployments are always protected, regardless of age. This
# many of the next-most-recent superseded deployments per (kb_id, alias) are
# also retained, as a quick-rollback buffer.
RETAIN_SUPERSEDED_PER_ALIAS = 3

# Legacy collections that don't follow either RAG naming convention
# (`rag__{kb}__{16-hex fingerprint}` or the pre-release-system
# `rag__{kb}__{YYYYMMDD}_{HHMMSS}`). These predate the alias-based lifecycle
# and are never touched by this DAG.
SKIP_LIST: set[str] = {"chat_documents", "code_documents"}

_RAG_COLLECTION_PATTERN = re.compile(r"^rag__.+__([0-9a-f]{16}|\d{8}_\d{6})$")

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
}


@dataclass(frozen=True)
class _ReleaseRow:
    id: str
    collection_name: str
    retired_at: Any


@dataclass(frozen=True)
class _DeploymentRow:
    kb_id: str
    alias: str
    release_id: str
    status: str
    order_key: Any


def _is_rag_managed_collection(name: str) -> bool:
    return bool(_RAG_COLLECTION_PATTERN.match(name))


def _protected_release_ids(
    deployments: list[_DeploymentRow], *, retain_superseded: int
) -> set[str]:
    """Pure decision logic: which release ids must never be retired yet."""
    protected: set[str] = set()
    superseded_by_alias: dict[tuple[str, str], list[_DeploymentRow]] = {}

    for deployment in deployments:
        if deployment.status in ("active", "pending"):
            protected.add(deployment.release_id)
        elif deployment.status == "superseded":
            key = (deployment.kb_id, deployment.alias)
            superseded_by_alias.setdefault(key, []).append(deployment)

    for rows in superseded_by_alias.values():
        newest_first = sorted(rows, key=lambda row: row.order_key, reverse=True)
        for row in newest_first[:retain_superseded]:
            protected.add(row.release_id)

    return protected


def _retirable_release_ids(
    releases: list[_ReleaseRow], *, protected_release_ids: set[str]
) -> list[str]:
    """Pure decision logic: which non-retired releases are safe to retire now."""
    return sorted(
        release.id
        for release in releases
        if release.retired_at is None and release.id not in protected_release_ids
    )


def _cleanup_orphan_collections() -> None:
    from sqlalchemy import select, update

    from app_config.runtime import get_settings
    from clients.db.models import RagAliasDeploymentRow, RagReleaseRow
    from rag.control_plane.postgres import create_session_factory

    db_url = get_settings().auth.agent042_db_url
    session_factory = create_session_factory(db_url)

    with session_factory() as session:
        release_rows = session.execute(select(RagReleaseRow)).scalars().all()
        deployment_rows = session.execute(select(RagAliasDeploymentRow)).scalars().all()

        releases = [
            _ReleaseRow(id=row.id, collection_name=row.collection_name, retired_at=row.retired_at)
            for row in release_rows
        ]
        deployments = [
            _DeploymentRow(
                kb_id=row.kb_id,
                alias=row.alias,
                release_id=row.release_id,
                status=row.status,
                order_key=row.applied_at or row.created_at,
            )
            for row in deployment_rows
        ]

        protected = _protected_release_ids(
            deployments, retain_superseded=RETAIN_SUPERSEDED_PER_ALIAS
        )
        to_retire = _retirable_release_ids(releases, protected_release_ids=protected)

        now = datetime.now(timezone.utc)
        if to_retire:
            for release_id in to_retire:
                print(f"  RETIRE (Postgres): release {release_id}")
            session.execute(
                update(RagReleaseRow).where(RagReleaseRow.id.in_(to_retire)).values(retired_at=now)
            )
        session.commit()

        releases_by_id = {release.id: release for release in releases}
        # Also retry deletion for releases retired in a previous run whose
        # Qdrant collection somehow still exists (e.g. a prior delete failed).
        already_retired = [release.id for release in releases if release.retired_at is not None]
        deletable_release_ids = set(to_retire) | set(already_retired)
        deletable_collection_names = {
            releases_by_id[release_id].collection_name
            for release_id in deletable_release_ids
            if release_id in releases_by_id
        }
        known_collections = {release.collection_name for release in releases}

    from qdrant_client import QdrantClient

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    all_collections = {c.name for c in client.get_collections().collections}

    deleted = 0
    for name in sorted(all_collections):
        if name in SKIP_LIST:
            print(f"  SKIP (legacy): {name}")
            continue
        if not _is_rag_managed_collection(name):
            print(f"  SKIP (not RAG-managed): {name}")
            continue
        if name not in known_collections:
            print(f"  SKIP (no matching rag_releases row, possibly in-flight): {name}")
            continue
        if name in deletable_collection_names:
            print(f"  DELETE: {name}")
            client.delete_collection(name)
            deleted += 1
        else:
            print(f"  KEEP (active/pending/retained release): {name}")

    print(
        f"\nCleanup complete: {len(to_retire)} release(s) retired, {deleted} collection(s) deleted."
    )


with DAG(
    dag_id="rag_collection_cleanup",
    default_args=default_args,
    description="Daily, release-aware cleanup of Qdrant collections with no live deployment.",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["rag", "cleanup", "maintenance"],
) as dag:
    cleanup = PythonOperator(
        task_id="cleanup_orphan_collections",
        python_callable=_cleanup_orphan_collections,
    )
