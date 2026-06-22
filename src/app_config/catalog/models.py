"""Runtime catalog models used by gateway, RAG, and eval code."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class AliasConfig(BaseModel):
    """Per-alias query-time RAG configuration used at runtime."""

    top_k: int
    score_threshold: float
    reranker: str | None = None
    retrieval_strategy: Literal["dense", "hybrid", "sparse"]
    reranker_multiplier: int


class AdapterConfig(BaseModel):
    """Per-task LoRA routing configuration."""

    name: str = ""
    alias: str = ""
    enabled: bool = False

    @model_validator(mode="after")
    def _enabled_adapter_must_be_complete(self) -> "AdapterConfig":
        if self.enabled and (not self.name.strip() or not self.alias.strip()):
            raise ValueError("enabled adapter requires non-empty name and alias")
        return self


class KBConfig(BaseModel):
    """Single knowledge-base entry within a task group."""

    name: str
    default_alias: str
    aliases: dict[str, AliasConfig]
    update_strategy: Literal["incremental", "replace"] = "replace"
    description: str

    @model_validator(mode="before")
    @classmethod
    def _compat_description_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("description"):
            for key in ("selection_description", "label"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return {**data, "description": value}
        return data

    @property
    def label(self) -> str:
        """Compatibility display text derived from the canonical description."""
        return self.description

    @property
    def selection_description(self) -> str:
        """Compatibility selection text derived from the canonical description."""
        return self.description

    @model_validator(mode="after")
    def _default_alias_must_exist(self) -> "KBConfig":
        if self.default_alias not in self.aliases:
            raise ValueError(
                f"default_alias '{self.default_alias}' is not a declared alias "
                f"(available: {list(self.aliases.keys())})"
            )
        return self


class TaskConfig(BaseModel):
    """Materialized task entry used at runtime."""

    task: str
    description: str
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    knowledge_bases: list[KBConfig] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _compat_description_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict) and not data.get("description"):
            for key in ("routing_description", "label"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return {**data, "description": value}
        return data

    @property
    def label(self) -> str:
        """Compatibility display text derived from the canonical description."""
        return self.description

    @property
    def routing_description(self) -> str:
        """Compatibility routing text derived from the canonical description."""
        return self.description
