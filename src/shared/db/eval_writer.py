"""Canonical persistence for project evaluation runs and samples."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from shared.db.models import EvalRun, EvalSample
from shared.db.urls import require_db_url, to_sync_url


def write_evaluation_results(
    rows: Sequence[dict[str, Any]],
    *,
    db_url: str,
    sample_rows: Sequence[dict[str, Any]] | None = None,
) -> None:
    """Persist evaluation rows atomically through the shared ORM schema."""
    db_url = require_db_url(db_url, purpose="Evaluation persistence")
    engine = create_engine(to_sync_url(db_url))
    try:
        with Session(engine) as session, session.begin():
            for values in rows:
                row = dict(values)
                row.setdefault("id", uuid.uuid4())
                session.add(EvalRun(**row))
            for values in sample_rows or ():
                row = dict(values)
                row.setdefault("id", uuid.uuid4())
                session.add(EvalSample(**row))
    finally:
        engine.dispose()


def _coverage_from_rows(
    rows: Sequence[tuple[str, str, str | None]],
    *,
    required_dataset_names: Sequence[str],
) -> bool:
    """Decide coverage from (dataset_name, status, eval_verdict) rows.

    Pure and DB-free so it's directly testable: every required benchmark
    dataset needs at least one 'completed' row, and no matching row may have
    eval_verdict='fail'. 'unscored' rows do not block coverage -- that is the
    explicit manual-review case the plan calls out.
    """
    if not required_dataset_names:
        return False
    completed_datasets = {dataset for dataset, status, _ in rows if status == "completed"}
    if not set(required_dataset_names) <= completed_datasets:
        return False
    return not any(verdict == "fail" for _, _, verdict in rows)


def check_evaluation_coverage(
    *,
    db_url: str,
    release_id: str,
    retrieval_config_digest: str,
    benchmark_source_instance_ids: Sequence[str],
) -> bool:
    """Return True if every listed benchmark has a completed, non-failing evaluation.

    Coverage is scoped to one exact (release_id, retrieval_config_digest)
    pair: a release/retrieval combination is "evaluated" only when every
    attached benchmark has run against that specific combination, per the
    plan's default-alias protection rule. Used as the
    `evaluation_coverage_checker` callback `rag.control_plane.alias_service.
    AliasService` accepts.
    """
    db_url = require_db_url(db_url, purpose="Evaluation coverage check")
    engine = create_engine(to_sync_url(db_url))
    try:
        with Session(engine) as session:
            rows = session.execute(
                select(EvalRun.dataset_name, EvalRun.status, EvalRun.eval_verdict).where(
                    EvalRun.rag_release_id == release_id,
                    EvalRun.retrieval_config_digest == retrieval_config_digest,
                )
            ).all()
    finally:
        engine.dispose()
    return _coverage_from_rows(rows, required_dataset_names=benchmark_source_instance_ids)
