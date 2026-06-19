"""SQLAlchemy (synchronous) implementations of the RAG control-plane repositories.

A synchronous engine is used deliberately: source processing, Qdrant
materialization, and `RagRuntime.retrieve()` are all synchronous today (see
the declarative alias workflow plan's Database Contract section). The
gateway's existing async ORM engine (`shared.db.engine`) is unrelated and
stays as-is for API code.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app_config.catalog.schema import AliasBuildConfig, AliasRetrievalConfig
from rag.control_plane.models import AliasDeployment, RagRelease, ReleaseBuildAttempt
from shared.db.models import RagAliasDeploymentRow, RagReleaseBuildRow, RagReleaseRow
from shared.db.urls import require_db_url, to_sync_url


def create_sync_engine(db_url: str):
    """Return a sync SQLAlchemy engine for the control plane's Postgres database."""
    db_url = require_db_url(db_url, purpose="The RAG control plane")
    return create_engine(to_sync_url(db_url))


def create_session_factory(db_url: str) -> sessionmaker[Session]:
    """Return a session factory bound to a freshly created sync engine."""
    return sessionmaker(create_sync_engine(db_url))


# ---------------------------------------------------------------------------
# Domain <-> ORM row conversions
# ---------------------------------------------------------------------------


def _attempt_to_row(attempt: ReleaseBuildAttempt) -> RagReleaseBuildRow:
    return RagReleaseBuildRow(
        id=attempt.id,
        kb_id=attempt.kb_id,
        requested_alias=attempt.requested_alias,
        status=attempt.status,
        catalog_digest=attempt.catalog_digest,
        build_config_digest=attempt.build_config_digest,
        retrieval_config_digest=attempt.retrieval_config_digest,
        source_declaration_digest=attempt.source_declaration_digest,
        source_snapshot_id=attempt.source_snapshot_id,
        release_id=attempt.release_id,
        collection_name=attempt.collection_name,
        started_at=attempt.started_at,
        finished_at=attempt.finished_at,
        error=attempt.error,
    )


def _row_to_attempt(row: RagReleaseBuildRow) -> ReleaseBuildAttempt:
    return ReleaseBuildAttempt(
        id=row.id,
        kb_id=row.kb_id,
        requested_alias=row.requested_alias,
        status=row.status,
        catalog_digest=row.catalog_digest,
        build_config_digest=row.build_config_digest,
        retrieval_config_digest=row.retrieval_config_digest,
        source_declaration_digest=row.source_declaration_digest,
        source_snapshot_id=row.source_snapshot_id,
        release_id=row.release_id,
        collection_name=row.collection_name,
        started_at=row.started_at,
        finished_at=row.finished_at,
        error=row.error,
    )


def _release_to_row(release: RagRelease, *, manifest_path: str) -> RagReleaseRow:
    return RagReleaseRow(
        id=release.id,
        kb_id=release.kb_id,
        collection_name=release.collection_name,
        manifest_id=release.manifest_id,
        manifest_path=manifest_path,
        release_fingerprint=release.release_fingerprint,
        catalog_digest=release.catalog_digest,
        build_config_digest=release.build_config_digest,
        source_declaration_digest=release.source_declaration_digest,
        source_snapshot_id=release.source_snapshot_id,
        build_config=release.build_config.model_dump(mode="json"),
        source_manifest_digests=release.source_manifest_digests,
        source_adapter_versions=release.source_adapter_versions,
        document_count=release.document_count,
        chunk_count=release.chunk_count,
        created_at=release.created_at,
    )


def _row_to_release(row: RagReleaseRow) -> RagRelease:
    return RagRelease(
        id=row.id,
        kb_id=row.kb_id,
        collection_name=row.collection_name,
        manifest_id=row.manifest_id,
        release_fingerprint=row.release_fingerprint,
        catalog_digest=row.catalog_digest,
        build_config_digest=row.build_config_digest,
        source_declaration_digest=row.source_declaration_digest,
        source_snapshot_id=row.source_snapshot_id,
        build_config=AliasBuildConfig.model_validate(row.build_config),
        source_manifest_digests=row.source_manifest_digests,
        source_adapter_versions=row.source_adapter_versions,
        document_count=row.document_count,
        chunk_count=row.chunk_count,
        created_at=row.created_at,
    )


def _deployment_to_row(deployment: AliasDeployment) -> RagAliasDeploymentRow:
    return RagAliasDeploymentRow(
        id=deployment.id,
        kb_id=deployment.kb_id,
        alias=deployment.alias,
        release_id=deployment.release_id,
        collection_name=deployment.collection_name,
        catalog_digest=deployment.catalog_digest,
        build_config_digest=deployment.build_config_digest,
        retrieval_config_digest=deployment.retrieval_config_digest,
        retrieval_config=deployment.retrieval_config.model_dump(mode="json"),
        status=deployment.status,
        applied_at=deployment.applied_at,
        superseded_at=deployment.superseded_at,
        error=deployment.error,
    )


def _row_to_deployment(row: RagAliasDeploymentRow) -> AliasDeployment:
    return AliasDeployment(
        id=row.id,
        kb_id=row.kb_id,
        alias=row.alias,
        release_id=row.release_id,
        collection_name=row.collection_name,
        catalog_digest=row.catalog_digest,
        build_config_digest=row.build_config_digest,
        retrieval_config_digest=row.retrieval_config_digest,
        retrieval_config=AliasRetrievalConfig.model_validate(row.retrieval_config),
        status=row.status,
        applied_at=row.applied_at,
        superseded_at=row.superseded_at,
        error=row.error,
    )


# ---------------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------------


class PostgresReleaseBuildRepository:
    """SQLAlchemy-backed `ReleaseBuildRepository`."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def create(self, attempt: ReleaseBuildAttempt) -> None:
        with self._session_factory() as session, session.begin():
            session.add(_attempt_to_row(attempt))

    def mark_completed(
        self,
        attempt_id: UUID,
        *,
        release_id: str,
        collection_name: str,
        source_snapshot_id: str,
        finished_at: datetime,
    ) -> None:
        with self._session_factory() as session, session.begin():
            row = session.get(RagReleaseBuildRow, attempt_id)
            if row is None:
                raise ValueError(f"build attempt '{attempt_id}' not found")
            row.status = "completed"
            row.release_id = release_id
            row.collection_name = collection_name
            row.source_snapshot_id = source_snapshot_id
            row.finished_at = finished_at

    def mark_failed(self, attempt_id: UUID, *, error: str, finished_at: datetime) -> None:
        with self._session_factory() as session, session.begin():
            row = session.get(RagReleaseBuildRow, attempt_id)
            if row is None:
                raise ValueError(f"build attempt '{attempt_id}' not found")
            row.status = "failed"
            row.error = error
            row.finished_at = finished_at

    def get(self, attempt_id: UUID) -> ReleaseBuildAttempt | None:
        with self._session_factory() as session:
            row = session.get(RagReleaseBuildRow, attempt_id)
            return _row_to_attempt(row) if row is not None else None

    def list_for_kb(self, kb_id: str) -> list[ReleaseBuildAttempt]:
        with self._session_factory() as session:
            rows = session.scalars(
                select(RagReleaseBuildRow)
                .where(RagReleaseBuildRow.kb_id == kb_id)
                .order_by(RagReleaseBuildRow.started_at.desc())
            ).all()
            return [_row_to_attempt(row) for row in rows]


class PostgresReleaseRepository:
    """SQLAlchemy-backed `ReleaseRepository`."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def insert(self, release: RagRelease, *, manifest_path: str) -> None:
        with self._session_factory() as session, session.begin():
            session.add(_release_to_row(release, manifest_path=manifest_path))

    def get(self, release_id: str) -> RagRelease | None:
        with self._session_factory() as session:
            row = session.get(RagReleaseRow, release_id)
            return _row_to_release(row) if row is not None else None

    def get_by_fingerprint(self, release_fingerprint: str) -> RagRelease | None:
        with self._session_factory() as session:
            row = session.scalars(
                select(RagReleaseRow).where(
                    RagReleaseRow.release_fingerprint == release_fingerprint
                )
            ).one_or_none()
            return _row_to_release(row) if row is not None else None

    def find_reusable(
        self,
        *,
        build_config_digest: str,
        source_declaration_digest: str,
    ) -> list[RagRelease]:
        with self._session_factory() as session:
            rows = session.scalars(
                select(RagReleaseRow).where(
                    RagReleaseRow.build_config_digest == build_config_digest,
                    RagReleaseRow.source_declaration_digest == source_declaration_digest,
                    RagReleaseRow.retired_at.is_(None),
                )
            ).all()
            return [_row_to_release(row) for row in rows]

    def list_for_kb(self, kb_id: str) -> list[RagRelease]:
        with self._session_factory() as session:
            rows = session.scalars(
                select(RagReleaseRow)
                .where(RagReleaseRow.kb_id == kb_id)
                .order_by(RagReleaseRow.created_at.desc())
            ).all()
            return [_row_to_release(row) for row in rows]

    def retire(self, release_id: str, *, retired_at: datetime) -> None:
        with self._session_factory() as session, session.begin():
            row = session.get(RagReleaseRow, release_id)
            if row is None:
                raise ValueError(f"release '{release_id}' not found")
            row.retired_at = retired_at


class PostgresAliasDeploymentRepository:
    """SQLAlchemy-backed `AliasDeploymentRepository`.

    `activate()` supersedes the previous active row and activates the new
    one in one transaction, so no reader ever observes zero or two active
    deployments for the same (kb_id, alias). The database's partial unique
    index on (kb_id, alias) WHERE status = 'active' is the final guard
    against a second process racing the same activation.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def get_active(self, *, kb_id: str, alias: str) -> AliasDeployment | None:
        with self._session_factory() as session:
            row = session.scalars(
                select(RagAliasDeploymentRow).where(
                    RagAliasDeploymentRow.kb_id == kb_id,
                    RagAliasDeploymentRow.alias == alias,
                    RagAliasDeploymentRow.status == "active",
                )
            ).one_or_none()
            return _row_to_deployment(row) if row is not None else None

    def get(self, deployment_id: UUID) -> AliasDeployment | None:
        with self._session_factory() as session:
            row = session.get(RagAliasDeploymentRow, deployment_id)
            return _row_to_deployment(row) if row is not None else None

    def create_pending(self, deployment: AliasDeployment) -> None:
        with self._session_factory() as session, session.begin():
            session.add(_deployment_to_row(deployment))

    def activate(self, deployment_id: UUID, *, applied_at: datetime) -> None:
        with self._session_factory() as session, session.begin():
            pending = session.get(RagAliasDeploymentRow, deployment_id)
            if pending is None:
                raise ValueError(f"deployment '{deployment_id}' not found")
            previous_active = session.scalars(
                select(RagAliasDeploymentRow).where(
                    RagAliasDeploymentRow.kb_id == pending.kb_id,
                    RagAliasDeploymentRow.alias == pending.alias,
                    RagAliasDeploymentRow.status == "active",
                    RagAliasDeploymentRow.id != deployment_id,
                )
            ).one_or_none()
            if previous_active is not None:
                previous_active.status = "superseded"
                previous_active.superseded_at = applied_at
            pending.status = "active"
            pending.applied_at = applied_at

    def mark_failed(self, deployment_id: UUID, *, error: str) -> None:
        with self._session_factory() as session, session.begin():
            row = session.get(RagAliasDeploymentRow, deployment_id)
            if row is None:
                raise ValueError(f"deployment '{deployment_id}' not found")
            row.status = "failed"
            row.error = error

    def list_history(self, *, kb_id: str, alias: str) -> list[AliasDeployment]:
        with self._session_factory() as session:
            rows = session.scalars(
                select(RagAliasDeploymentRow)
                .where(
                    RagAliasDeploymentRow.kb_id == kb_id,
                    RagAliasDeploymentRow.alias == alias,
                )
                .order_by(RagAliasDeploymentRow.created_at.desc())
            ).all()
            return [_row_to_deployment(row) for row in rows]
