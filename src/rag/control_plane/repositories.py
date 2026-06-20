"""Repository protocols for the RAG control plane.

Application services (`rag.control_plane.alias_service`, phase 4) depend on
these protocols, not on `rag.control_plane.postgres` directly, so tests can
inject fakes instead of a real database.
"""

from __future__ import annotations

from datetime import datetime
from typing import ContextManager, Protocol
from uuid import UUID

from rag.control_plane.models import AliasDeployment, RagRelease, ReleaseBuildAttempt


class ReleaseBuildRepository(Protocol):
    """Persistence for release build attempts. Never a runtime source of truth."""

    def create(self, attempt: ReleaseBuildAttempt) -> None:
        """Insert a new build attempt, normally with status='running'."""
        ...

    def mark_completed(
        self,
        attempt_id: UUID,
        *,
        release_id: str,
        collection_name: str,
        source_snapshot_id: str,
        finished_at: datetime,
    ) -> None:
        """Mark an attempt completed and record the release it produced."""
        ...

    def mark_failed(self, attempt_id: UUID, *, error: str, finished_at: datetime) -> None:
        """Mark an attempt failed. Never creates a release row."""
        ...

    def get(self, attempt_id: UUID) -> ReleaseBuildAttempt | None:
        """Return one build attempt by id."""
        ...

    def list_for_kb(self, kb_id: str) -> list[ReleaseBuildAttempt]:
        """Return all build attempts for a KB, newest first."""
        ...


class ReleaseRepository(Protocol):
    """Persistence for immutable, content-identified releases."""

    def insert(self, release: RagRelease, *, manifest_path: str) -> None:
        """Insert a new release row. Release rows are never updated in place.

        *manifest_path* is a SQL-table-only convenience column (where the
        release manifest JSON lives on disk); it is not part of the
        `RagRelease` domain contract because it is fully determined by
        `rag_data_root`, `kb_id`, and `release.id`. Callers that need it back
        recompute it via `rag.contracts.manifests.release_manifest_path`.
        """
        ...

    def get(self, release_id: str) -> RagRelease | None:
        """Return one release by id."""
        ...

    def get_by_fingerprint(self, release_fingerprint: str) -> RagRelease | None:
        """Return the release with this exact full fingerprint, if any."""
        ...

    def release_lock(self, release_fingerprint: str) -> ContextManager[None]:
        """Serialize materialization/registration for one full release fingerprint."""
        ...

    def find_reusable(
        self,
        *,
        build_config_digest: str,
        source_declaration_digest: str,
    ) -> list[RagRelease]:
        """Return non-retired releases matching this build config and source declaration.

        Deliberately does not match on source_snapshot_id: a manifest left
        unchanged can still have produced releases with different snapshots
        if remote source content drifted between builds. Multiple results
        here is exactly the "ambiguous reusable release" case alias apply
        must refuse unless disambiguated with --release.
        """
        ...

    def list_for_kb(self, kb_id: str) -> list[RagRelease]:
        """Return all releases for a KB, newest first."""
        ...

    def retire(self, release_id: str, *, retired_at: datetime) -> None:
        """Mark a release retired. Does not delete the row or its manifest."""
        ...


class AliasDeploymentRepository(Protocol):
    """Persistence for applied alias deployment history.

    At most one deployment is active per (kb_id, alias). Rows are append-only:
    activation supersedes the previous active row rather than mutating it.
    """

    def get_active(self, *, kb_id: str, alias: str) -> AliasDeployment | None:
        """Return the active deployment for (kb_id, alias), if any."""
        ...

    def get(self, deployment_id: UUID) -> AliasDeployment | None:
        """Return one deployment by id."""
        ...

    def create_pending(self, deployment: AliasDeployment) -> None:
        """Insert a new deployment row with status='pending'."""
        ...

    def activate(self, deployment_id: UUID, *, applied_at: datetime) -> None:
        """Supersede the previous active deployment and activate this one.

        Both updates happen in one transaction: no request ever observes a
        state with zero or two active deployments for the same (kb_id, alias).
        """
        ...

    def mark_failed(self, deployment_id: UUID, *, error: str) -> None:
        """Mark a pending deployment failed without superseding the active one."""
        ...

    def list_history(self, *, kb_id: str, alias: str) -> list[AliasDeployment]:
        """Return all deployments for (kb_id, alias), newest first."""
        ...
