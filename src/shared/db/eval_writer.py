"""Canonical persistence for project evaluation runs and samples."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from shared.db.models import EvalRun, EvalSample


def _sync_db_url(db_url: str) -> str:
    if not db_url:
        raise ValueError("Evaluation persistence requires a database URL")
    return db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)


def write_evaluation_results(
    rows: Sequence[dict[str, Any]],
    *,
    db_url: str,
    sample_rows: Sequence[dict[str, Any]] | None = None,
) -> None:
    """Persist evaluation rows atomically through the shared ORM schema."""
    engine = create_engine(_sync_db_url(db_url))
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
