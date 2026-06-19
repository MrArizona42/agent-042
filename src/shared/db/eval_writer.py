"""Canonical persistence for project evaluation runs and samples."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import create_engine
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
