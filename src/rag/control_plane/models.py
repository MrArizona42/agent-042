"""Release, build-attempt, deployment, and diff contracts for the RAG control plane."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app_config.catalog.schema import AliasBuildConfig, AliasRetrievalConfig

BuildStatus = Literal["running", "failed", "completed"]
DeploymentStatus = Literal["pending", "active", "superseded", "failed"]


class ReleaseBuildAttempt(BaseModel):
    """Execution and failure record for one release build. Never a runtime source of truth."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    kb_id: str
    requested_alias: str
    status: BuildStatus
    catalog_digest: str
    build_config_digest: str
    retrieval_config_digest: str
    source_declaration_digest: str
    source_snapshot_id: str | None = None
    release_id: str | None = None
    collection_name: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    error: str | None = None


class RagRelease(BaseModel):
    """An immutable, content-identified, reusable build result. Carries no alias field."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    id: str
    kb_id: str
    collection_name: str
    manifest_id: str
    release_fingerprint: str
    catalog_digest: str
    build_config_digest: str
    source_declaration_digest: str
    source_snapshot_id: str
    build_config: AliasBuildConfig
    source_manifest_digests: dict[str, str]
    source_adapter_versions: dict[str, str]
    document_count: int
    chunk_count: int
    created_at: datetime


class AliasDeployment(BaseModel):
    """An applied alias deployment: at most one active row per (kb_id, alias)."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    kb_id: str
    alias: str
    release_id: str
    collection_name: str
    catalog_digest: str
    build_config_digest: str
    retrieval_config_digest: str
    retrieval_config: AliasRetrievalConfig
    status: DeploymentStatus
    applied_at: datetime | None = None
    superseded_at: datetime | None = None
    error: str | None = None


class AliasDiff(BaseModel):
    """Comparison of desired catalog state against applied deployment state."""

    model_config = ConfigDict(extra="forbid")

    kb_id: str
    alias: str
    desired_catalog_digest: str
    desired_build_config_digest: str
    desired_retrieval_config_digest: str
    applied_deployment_id: UUID | None
    applied_release_id: str | None
    build_drift: bool
    retrieval_drift: bool
    source_declaration_drift: bool
    provider_mismatches: list[str]
    reusable_release_ids: list[str]
