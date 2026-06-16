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


class LifecycleStageResult(BaseModel):
    """Shared wrapper returned by lifecycle stage functions."""

    model_config = ConfigDict(extra="forbid")

    build_run: BuildRun
    result: Any

