"""TOML schema models for the shared catalog."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from shared.catalog.models import AdapterConfig, AliasConfig


class CatalogAliasConfig(AliasConfig):
    """Explicit per-KB alias configuration in the catalog file."""


class CatalogTaskConfig(BaseModel):
    """Task entry in the catalog TOML file."""

    id: str
    enabled: bool = True
    label: str = ""
    routing_description: str
    kb_refs: list[str] = Field(default_factory=list)
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)


class CatalogKBConfig(BaseModel):
    """Knowledge-base entry in the catalog TOML file."""

    id: str
    enabled: bool = True
    default_alias: str
    aliases: dict[str, CatalogAliasConfig]
    update_strategy: Literal["incremental", "replace"] = "replace"
    label: str = ""
    description: str = ""
    selection_description: str


class SourceIngestAdapterConfig(BaseModel):
    """Source-level adapter contract for ingest lifecycle behavior."""

    id: str
    version: str = "1"
    settings: dict[str, object] = Field(default_factory=dict)

    @field_validator("id", "version")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()


class SourceConfig(BaseModel):
    """Source instance metadata for a knowledge-base build pipeline."""

    type: str
    kb: str
    id: str
    manifest: str
    ingest_adapter: SourceIngestAdapterConfig | None = None
    settings: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _default_ingest_adapter_from_type(self) -> "SourceConfig":
        if self.ingest_adapter is None:
            self.ingest_adapter = SourceIngestAdapterConfig(id=self.type, version="legacy")
        return self


class CatalogConfig(BaseModel):
    """Root schema for the TOML-backed shared catalog."""

    schema_version: int = 1
    tasks: list[CatalogTaskConfig] = Field(default_factory=list)
    knowledge_bases: list[CatalogKBConfig] = Field(default_factory=list)
    sources: list[SourceConfig] = Field(default_factory=list)
