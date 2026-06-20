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


def _orm_row_payload(row: EvalRun | EvalSample) -> dict[str, Any]:
    return {column.name: getattr(row, column.name) for column in row.__table__.columns}


def list_evaluation_runs(
    *, db_url: str, knowledge_base: str, limit: int = 100
) -> list[dict[str, Any]]:
    """Return recent persisted RAG metric runs for one knowledge base."""
    db_url = require_db_url(db_url, purpose="Evaluation history")
    engine = create_engine(to_sync_url(db_url))
    try:
        with Session(engine) as session:
            rows = session.scalars(
                select(EvalRun)
                .where(EvalRun.task == "rag", EvalRun.knowledge_base == knowledge_base)
                .order_by(EvalRun.created_at.desc())
                .limit(limit)
            ).all()
            return [_orm_row_payload(row) for row in rows]
    finally:
        engine.dispose()


def get_evaluation_run(*, db_url: str, eval_run_id: str) -> dict[str, Any] | None:
    """Return one persisted metric run and its ordered sample observations."""
    try:
        run_id = uuid.UUID(eval_run_id)
    except ValueError:
        return None
    db_url = require_db_url(db_url, purpose="Evaluation history")
    engine = create_engine(to_sync_url(db_url))
    try:
        with Session(engine) as session:
            run = session.get(EvalRun, run_id)
            if run is None:
                return None
            samples = session.scalars(
                select(EvalSample)
                .where(EvalSample.eval_run_id == run_id)
                .order_by(EvalSample.sample_idx)
            ).all()
            return {
                "run": _orm_row_payload(run),
                "samples": [_orm_row_payload(sample) for sample in samples],
            }
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
