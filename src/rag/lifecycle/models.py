"""RAG lifecycle request and build-run contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

BuildRunStatus = Literal[
    "planned",
    "running",
    "failed",
    "succeeded",
    "promoted",
    "rolled_back",
]


class BuildRequest(BaseModel):
    """Operator request for a RAG build lifecycle stage."""

    model_config = ConfigDict(extra="forbid")

    catalog_path: str
    kb_id: str
    rag_data_root: str
    source_ids: list[str] | None = None
    alias_config: str | None = None
    collection_name: str | None = None
    document_ids: list[str] | None = None
    limit: int | None = Field(default=None, ge=0)
    force_fetch: bool = False
    force_extract: bool = False
    force_chunk: bool = False
    force_recreate: bool = False
    dry_run: bool = False

    @field_validator("catalog_path", "kb_id", "rag_data_root")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()

    @field_validator("source_ids", "document_ids")
    @classmethod
    def _blank_items_are_ignored(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [item.strip() for item in value if item.strip()]
        return cleaned or None

    @field_validator("alias_config", "collection_name")
    @classmethod
    def _blank_optional_strings_are_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class BuildRun(BaseModel):
    """Persistable status record for one RAG build run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    kb_id: str
    source_ids: list[str] | None = None
    status: BuildRunStatus = "planned"
    current_stage: str | None = None
    catalog_path: str
    rag_data_root: str
    alias_config: str | None = None
    collection_name: str | None = None
    catalog_digest: str | None = None
    manifest_digests: dict[str, str] = Field(default_factory=dict)
    adapter_versions: dict[str, str] = Field(default_factory=dict)
    build_profile_digest: str | None = None
    report_ref: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    finished_at: datetime | None = None
    stage_results: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)

    @field_validator("run_id", "kb_id", "catalog_path", "rag_data_root")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()

    def to_summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "current_stage": self.current_stage,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "source_ids": self.source_ids,
            "errors": self.errors,
        }


class LifecycleStageResult(BaseModel):
    """Shared wrapper returned by lifecycle stage functions."""

    model_config = ConfigDict(extra="forbid")

    build_run: BuildRun
    result: Any


class SourcePlanEntry(BaseModel):
    """Preflight validation result for one catalog source."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    adapter_id: str
    adapter_version: str
    manifest_ref: str
    manifest_reachable: bool
    adapter_registered: bool
    source_type_matches: bool
    errors: list[str] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


class PlanResult(BaseModel):
    """Output of a preflight plan check for a KB build."""

    model_config = ConfigDict(extra="forbid")

    kb_id: str
    catalog_path: str
    catalog_reachable: bool
    kb_found: bool
    sources: list[SourcePlanEntry] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        return (
            self.catalog_reachable
            and self.kb_found
            and not self.errors
            and all(entry.valid for entry in self.sources)
        )
