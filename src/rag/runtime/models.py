"""Runtime retrieval contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rag.domain import RetrievalHit


class RagRuntimeSource(BaseModel):
    """One requested KB/alias pair for runtime retrieval."""

    model_config = ConfigDict(extra="forbid")

    knowledge_base: str
    alias: str | None = None


class RuntimeSkippedSource(BaseModel):
    """A requested source that could not be queried."""

    model_config = ConfigDict(extra="forbid")

    knowledge_base: str
    alias: str | None = None
    reason: str


class RagRuntimeResult(BaseModel):
    """Citation-ready retrieval result plus runtime provenance."""

    model_config = ConfigDict(extra="forbid")

    hits: list[RetrievalHit] = Field(default_factory=list)
    skipped_sources: list[RuntimeSkippedSource] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
