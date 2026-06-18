"""TOML schema models for the application catalog."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app_config.catalog.models import AdapterConfig, AliasConfig


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
    ingest_adapter: SourceIngestAdapterConfig
    settings: dict[str, object] = Field(default_factory=dict)


SourceInstanceRole = Literal["corpus", "benchmark"]
BenchmarkSuite = Literal["retrieval_quality", "context_quality", "generation_quality"]


class SourceAdapterConfig(BaseModel):
    """Declarative `[[source_adapters]]` entry: a factory for a source-capable adapter."""

    id: str
    version: str = "1"
    description: str
    factory: str

    @field_validator("id", "version", "description", "factory")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()


class BenchmarkAdapterConfig(BaseModel):
    """Declarative `[[benchmark_adapters]]` entry: a factory for a benchmark-capable adapter."""

    id: str
    version: str = "1"
    description: str
    factory: str

    @field_validator("id", "version", "description", "factory")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()


class BenchmarkSourceConfig(BaseModel):
    """The `benchmark = { ... }` block on a `role = "benchmark"` source instance."""

    suites: list[BenchmarkSuite] = Field(min_length=1)

    @field_validator("suites")
    @classmethod
    def _suites_must_not_have_duplicates(cls, value: list[BenchmarkSuite]) -> list[BenchmarkSuite]:
        if len(set(value)) != len(value):
            raise ValueError("benchmark.suites must not contain duplicate values")
        return value


class SourceInstanceAdapterRef(BaseModel):
    """Reference from a source instance to a declared source or benchmark adapter."""

    id: str
    version: str = "1"

    @field_validator("id", "version")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()


class SourceInstanceConfig(BaseModel):
    """Declarative `[[source_instances]]` entry: a globally addressable source or benchmark."""

    id: str
    description: str
    role: SourceInstanceRole = "corpus"
    knowledge_base: str
    adapter: SourceInstanceAdapterRef
    benchmark: BenchmarkSourceConfig | None = None

    @field_validator("id", "description", "knowledge_base")
    @classmethod
    def _required_strings_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be non-empty")
        return value.strip()

    @model_validator(mode="after")
    def _benchmark_block_matches_role(self) -> "SourceInstanceConfig":
        if self.role == "benchmark" and self.benchmark is None:
            raise ValueError(
                f"source instance '{self.id}' has role 'benchmark' but no benchmark block"
            )
        if self.role == "corpus" and self.benchmark is not None:
            raise ValueError(
                f"source instance '{self.id}' has role 'corpus' and must not have a benchmark block"
            )
        return self


class CatalogConfig(BaseModel):
    """Root schema for the TOML-backed application catalog."""

    schema_version: int = 1
    tasks: list[CatalogTaskConfig] = Field(default_factory=list)
    knowledge_bases: list[CatalogKBConfig] = Field(default_factory=list)
    sources: list[SourceConfig] = Field(default_factory=list)
    source_adapters: list[SourceAdapterConfig] = Field(default_factory=list)
    benchmark_adapters: list[BenchmarkAdapterConfig] = Field(default_factory=list)
    source_instances: list[SourceInstanceConfig] = Field(default_factory=list)
