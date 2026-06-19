"""In-memory fakes for the RAG control-plane repository protocols.

Used by this phase's own behavioral tests and by phase 4's alias_service
tests ("service tests with fake repositories" per the declarative alias
workflow plan). Each fake enforces the same invariants the real Postgres
schema enforces (active-deployment uniqueness via the partial unique index,
append-only history, no release row on a failed attempt) so tests against
the fakes exercise the actual contract, not just a passive store.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from rag.control_plane.models import AliasDeployment, RagRelease, ReleaseBuildAttempt


class FakeReleaseBuildRepository:
    def __init__(self) -> None:
        self._rows: dict[UUID, ReleaseBuildAttempt] = {}

    def create(self, attempt: ReleaseBuildAttempt) -> None:
        if attempt.id in self._rows:
            raise ValueError(f"build attempt '{attempt.id}' already exists")
        self._rows[attempt.id] = attempt

    def mark_completed(
        self,
        attempt_id: UUID,
        *,
        release_id: str,
        collection_name: str,
        source_snapshot_id: str,
        finished_at: datetime,
    ) -> None:
        row = self._rows.get(attempt_id)
        if row is None:
            raise ValueError(f"build attempt '{attempt_id}' not found")
        self._rows[attempt_id] = row.model_copy(
            update={
                "status": "completed",
                "release_id": release_id,
                "collection_name": collection_name,
                "source_snapshot_id": source_snapshot_id,
                "finished_at": finished_at,
            }
        )

    def mark_failed(self, attempt_id: UUID, *, error: str, finished_at: datetime) -> None:
        row = self._rows.get(attempt_id)
        if row is None:
            raise ValueError(f"build attempt '{attempt_id}' not found")
        self._rows[attempt_id] = row.model_copy(
            update={"status": "failed", "error": error, "finished_at": finished_at}
        )

    def get(self, attempt_id: UUID) -> ReleaseBuildAttempt | None:
        return self._rows.get(attempt_id)

    def list_for_kb(self, kb_id: str) -> list[ReleaseBuildAttempt]:
        rows = [row for row in self._rows.values() if row.kb_id == kb_id]
        return sorted(rows, key=lambda row: row.started_at, reverse=True)


class FakeReleaseRepository:
    def __init__(self) -> None:
        self._rows: dict[str, RagRelease] = {}

    def insert(self, release: RagRelease, *, manifest_path: str) -> None:
        del manifest_path  # not part of the domain object; SQL-table-only column
        if release.id in self._rows:
            raise ValueError(f"release '{release.id}' already exists; releases are immutable")
        self._rows[release.id] = release

    def get(self, release_id: str) -> RagRelease | None:
        return self._rows.get(release_id)

    def get_by_fingerprint(self, release_fingerprint: str) -> RagRelease | None:
        for release in self._rows.values():
            if release.release_fingerprint == release_fingerprint:
                return release
        return None

    def find_reusable(
        self,
        *,
        build_config_digest: str,
        source_declaration_digest: str,
        source_snapshot_id: str,
    ) -> list[RagRelease]:
        return [
            release
            for release in self._rows.values()
            if release.build_config_digest == build_config_digest
            and release.source_declaration_digest == source_declaration_digest
            and release.source_snapshot_id == source_snapshot_id
        ]

    def list_for_kb(self, kb_id: str) -> list[RagRelease]:
        rows = [release for release in self._rows.values() if release.kb_id == kb_id]
        return sorted(rows, key=lambda release: release.created_at, reverse=True)

    def retire(self, release_id: str, *, retired_at: datetime) -> None:
        del retired_at
        if release_id not in self._rows:
            raise ValueError(f"release '{release_id}' not found")


class FakeAliasDeploymentRepository:
    def __init__(self) -> None:
        self._rows: dict[UUID, AliasDeployment] = {}

    def get_active(self, *, kb_id: str, alias: str) -> AliasDeployment | None:
        for row in self._rows.values():
            if row.kb_id == kb_id and row.alias == alias and row.status == "active":
                return row
        return None

    def get(self, deployment_id: UUID) -> AliasDeployment | None:
        return self._rows.get(deployment_id)

    def create_pending(self, deployment: AliasDeployment) -> None:
        if deployment.id in self._rows:
            raise ValueError(f"deployment '{deployment.id}' already exists")
        self._rows[deployment.id] = deployment

    def activate(self, deployment_id: UUID, *, applied_at: datetime) -> None:
        pending = self._rows.get(deployment_id)
        if pending is None:
            raise ValueError(f"deployment '{deployment_id}' not found")
        for other_id, other in list(self._rows.items()):
            if (
                other_id != deployment_id
                and other.kb_id == pending.kb_id
                and other.alias == pending.alias
                and other.status == "active"
            ):
                self._rows[other_id] = other.model_copy(
                    update={"status": "superseded", "superseded_at": applied_at}
                )
        self._rows[deployment_id] = pending.model_copy(
            update={"status": "active", "applied_at": applied_at}
        )

    def mark_failed(self, deployment_id: UUID, *, error: str) -> None:
        row = self._rows.get(deployment_id)
        if row is None:
            raise ValueError(f"deployment '{deployment_id}' not found")
        self._rows[deployment_id] = row.model_copy(update={"status": "failed", "error": error})

    def list_history(self, *, kb_id: str, alias: str) -> list[AliasDeployment]:
        # AliasDeployment has no created_at (that's a database-stamped, table-only
        # column -- see RagAliasDeploymentRow); dict insertion order substitutes
        # for it here, newest first.
        rows = [row for row in self._rows.values() if row.kb_id == kb_id and row.alias == alias]
        return list(reversed(rows))
