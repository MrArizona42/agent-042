"""TOML schema models for the shared catalog."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

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
    source_ref: str | None = None


class SourceConfig(BaseModel):
    """Source metadata for a knowledge-base build pipeline."""

    id: str
    type: str
    manifest: str | None = None
    settings: dict[str, object] = Field(default_factory=dict)


class CatalogConfig(BaseModel):
    """Root schema for the TOML-backed shared catalog."""

    schema_version: int = 1
    tasks: list[CatalogTaskConfig] = Field(default_factory=list)
    knowledge_bases: list[CatalogKBConfig] = Field(default_factory=list)
    sources: list[SourceConfig] = Field(default_factory=list)
