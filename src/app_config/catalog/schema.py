"""TOML schema models for the application catalog."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app_config.catalog.models import AdapterConfig

# ---------------------------------------------------------------------------
# Schema-version-4 alias build/retrieve shapes.
# ---------------------------------------------------------------------------

RetrievalStrategy = Literal["dense", "sparse", "hybrid"]
ChunkingStrategy = Literal["sentence"]


class DenseEncoderConfig(BaseModel):
    """Dense vector producer identity: model and expected output dimension."""

    model_config = ConfigDict(extra="forbid")

    model: str
    dimension: int = Field(gt=0)


class SparseEncoderConfig(BaseModel):
    """Sparse vector producer identity."""

    model_config = ConfigDict(extra="forbid")

    model: str


class AliasChunkingConfig(BaseModel):
    """Chunking behavior that determines node/chunk artifact identity."""

    model_config = ConfigDict(extra="forbid")

    strategy: ChunkingStrategy
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def _overlap_must_be_smaller_than_size(self) -> "AliasChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self


class AliasBuildConfig(BaseModel):
    """Desired index-time configuration for one alias: artifacts that must exist."""

    model_config = ConfigDict(extra="forbid")

    chunking: AliasChunkingConfig
    dense_encoder: DenseEncoderConfig
    sparse_encoder: SparseEncoderConfig | None = None


class AliasRetrievalConfig(BaseModel):
    """Desired query-time configuration for one alias: how a release is queried."""

    model_config = ConfigDict(extra="forbid")

    strategy: RetrievalStrategy
    top_k: int = Field(gt=0)
    score_threshold: float
    reranker: str | None = None
    reranker_multiplier: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def _reranker_multiplier_requires_reranker(self) -> "AliasRetrievalConfig":
        if self.reranker is None and self.reranker_multiplier != 1:
            raise ValueError("reranker_multiplier must be 1 when reranker is omitted")
        return self


class CatalogAliasConfig(BaseModel):
    """Complete desired state for one named KB deployment: build and retrieve."""

    model_config = ConfigDict(extra="forbid")

    build: AliasBuildConfig
    retrieve: AliasRetrievalConfig

    @model_validator(mode="after")
    def _retrieval_strategy_matches_available_encoders(self) -> "CatalogAliasConfig":
        strategy = self.retrieve.strategy
        has_sparse = self.build.sparse_encoder is not None
        if strategy == "sparse" and not has_sparse:
            raise ValueError("retrieve.strategy 'sparse' requires build.sparse_encoder")
        if strategy == "hybrid" and not has_sparse:
            raise ValueError("retrieve.strategy 'hybrid' requires build.sparse_encoder")
        return self


class CatalogTaskConfig(BaseModel):
    """Task entry in the catalog TOML file."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    knowledge_bases: list[str] = Field(default_factory=list)
    lora_adapter: AdapterConfig = Field(default_factory=AdapterConfig)


class CatalogKBConfig(BaseModel):
    """Knowledge-base entry in the catalog TOML file."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    default_alias: str
    aliases: dict[str, CatalogAliasConfig]
    update_strategy: Literal["incremental", "replace"] = "replace"


SourceInstanceRole = Literal["corpus", "benchmark"]
BenchmarkSuite = Literal["retrieval_quality", "context_quality", "generation_quality"]


class SourceAdapterConfig(BaseModel):
    """Declarative `[[source_adapters]]` entry: a factory for a source-capable adapter."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    suites: list[BenchmarkSuite] = Field(min_length=1)

    @field_validator("suites")
    @classmethod
    def _suites_must_not_have_duplicates(cls, value: list[BenchmarkSuite]) -> list[BenchmarkSuite]:
        if len(set(value)) != len(value):
            raise ValueError("benchmark.suites must not contain duplicate values")
        return value


class SourceInstanceAdapterRef(BaseModel):
    """Reference from a source instance to a declared source or benchmark adapter."""

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 4
    tasks: list[CatalogTaskConfig] = Field(default_factory=list)
    knowledge_bases: list[CatalogKBConfig] = Field(default_factory=list)
    source_adapters: list[SourceAdapterConfig] = Field(default_factory=list)
    benchmark_adapters: list[BenchmarkAdapterConfig] = Field(default_factory=list)
    source_instances: list[SourceInstanceConfig] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def _schema_version_must_be_current(cls, value: int) -> int:
        if value != 4:
            raise ValueError(
                f"catalog schema_version {value} is not supported; only version 4 "
                "(nested alias build/retrieve blocks) is accepted. Migrate flat "
                "top_k/score_threshold/retrieval_strategy alias fields into "
                "[knowledge_bases.aliases.<name>.build] and "
                "[knowledge_bases.aliases.<name>.retrieve] tables."
            )
        return value
