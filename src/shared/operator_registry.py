"""Operator registry models and loading helpers.

This module owns the TOML-backed operator registry that materializes the
runtime task/knowledge-base catalog used by gateway and RAG code.
"""

from __future__ import annotations

import logging
import os
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, Field, model_validator

from shared.local_env import get_repo_root

logger = logging.getLogger(__name__)

_DEFAULT_OPERATOR_REGISTRY_PATH = Path(__file__).resolve().parent / "operator_registry.toml"
_NESTED_OPERATOR_REGISTRY_PATH_ENV = "REGISTRY__OPERATOR_REGISTRY_PATH"


class AliasValueConfig(BaseModel):
    """Reusable query-time alias shape shared by runtime and registry profiles."""

    top_k: int
    score_threshold: float
    reranker: str | None = None
    retrieval_strategy: Literal["dense", "hybrid", "sparse"]
    reranker_multiplier: int


class AliasConfig(AliasValueConfig):
    """Per-alias query-time RAG configuration used at runtime."""


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
    label: str = ""
    description: str = ""
    selection_description: str
    source_ref: str | None = None

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
    label: str = ""
    routing_description: str
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    knowledge_bases: list[KBConfig] = Field(default_factory=list)


class SourceConfig(BaseModel):
    """Source metadata for a knowledge base build pipeline."""

    id: str
    type: str
    manifest: str | None = None
    settings: dict[str, object] = Field(default_factory=dict)


class RegistryAliasConfig(AliasValueConfig):
    """Explicit per-KB alias configuration in the operator registry."""


class RegistryTaskConfig(BaseModel):
    """Normalized task entry in the TOML operator registry."""

    id: str
    enabled: bool = True
    label: str = ""
    routing_description: str
    kb_refs: list[str] = Field(default_factory=list)
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)


class RegistryKBConfig(BaseModel):
    """Normalized knowledge-base entry in the TOML operator registry."""

    id: str
    enabled: bool = True
    default_alias: str
    aliases: dict[str, RegistryAliasConfig]
    update_strategy: Literal["incremental", "replace"] = "replace"
    label: str = ""
    description: str = ""
    selection_description: str
    source_ref: str | None = None


class OperatorRegistryConfig(BaseModel):
    """Root model for the TOML-based operator registry."""

    schema_version: int = 1
    tasks: list[RegistryTaskConfig] = Field(default_factory=list)
    knowledge_bases: list[RegistryKBConfig] = Field(default_factory=list)
    sources: list[SourceConfig] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_mapping_sections(cls, value: object) -> object:
        """Accept older mapping-style sections while canonical TOML uses arrays."""
        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        for key in ("tasks", "knowledge_bases", "sources"):
            section = normalized.get(key)
            if not isinstance(section, dict):
                continue
            normalized[key] = [
                {"id": item_id, **item_payload}
                for item_id, item_payload in section.items()
                if isinstance(item_payload, dict)
            ]
        return normalized


class RegistryPathSettings(Protocol):
    """Structural type for settings objects that provide registry path overrides."""

    operator_registry_path: Path | None


_RegistryItem = TypeVar("_RegistryItem", RegistryTaskConfig, RegistryKBConfig, SourceConfig)


def _index_by_id(items: list[_RegistryItem], section: str) -> dict[str, _RegistryItem]:
    """Return items keyed by id, preserving TOML order and rejecting duplicates."""
    index: dict[str, _RegistryItem] = {}
    for item in items:
        item_id = item.id.strip()
        if not item_id:
            raise ValueError(f"{section} entries require non-empty id")
        if item_id in index:
            raise ValueError(f"Duplicate {section} id '{item_id}'")
        index[item_id] = item
    return index


def _materialize_operator_registry(
    registry_cfg: OperatorRegistryConfig,
) -> tuple[dict[str, TaskConfig], dict[str, KBConfig]]:
    """Build the runtime task registry from the normalized TOML shape."""

    registry_kbs = _index_by_id(registry_cfg.knowledge_bases, "knowledge_bases")
    registry_tasks = _index_by_id(registry_cfg.tasks, "tasks")
    registry_sources = _index_by_id(registry_cfg.sources, "sources")

    kb_index: dict[str, KBConfig] = {}
    for kb_name, kb_cfg in registry_kbs.items():
        if not kb_cfg.enabled:
            continue
        if kb_cfg.source_ref is not None and kb_cfg.source_ref not in registry_sources:
            raise ValueError(
                f"KB '{kb_name}' references unknown source '{kb_cfg.source_ref}'"
            )

        aliases: dict[str, AliasConfig] = {}
        for alias_name, alias_cfg in kb_cfg.aliases.items():
            aliases[alias_name] = AliasConfig(
                top_k=alias_cfg.top_k,
                score_threshold=alias_cfg.score_threshold,
                reranker=alias_cfg.reranker,
                retrieval_strategy=alias_cfg.retrieval_strategy,
                reranker_multiplier=alias_cfg.reranker_multiplier,
            )

        kb_index[kb_name] = KBConfig(
            name=kb_name,
            default_alias=kb_cfg.default_alias,
            aliases=aliases,
            update_strategy=kb_cfg.update_strategy,
            label=kb_cfg.label,
            description=kb_cfg.description,
            selection_description=kb_cfg.selection_description,
            source_ref=kb_cfg.source_ref,
        )

    task_registry: dict[str, TaskConfig] = {}
    for task_name, task_cfg in registry_tasks.items():
        if not task_cfg.enabled:
            continue

        task_knowledge_bases: list[KBConfig] = []
        for kb_name in task_cfg.kb_refs:
            if kb_name not in registry_kbs:
                raise ValueError(f"Task '{task_name}' references unknown KB '{kb_name}'")
            kb_runtime_cfg = kb_index.get(kb_name)
            if kb_runtime_cfg is not None:
                task_knowledge_bases.append(kb_runtime_cfg)

        task_registry[task_name] = TaskConfig(
            task=task_name,
            label=task_cfg.label,
            routing_description=task_cfg.routing_description,
            adapter=task_cfg.adapter.model_copy(deep=True),
            knowledge_bases=task_knowledge_bases,
        )

    return task_registry, kb_index


def load_knowledge_bases(
    path: Path | str,
) -> tuple[dict[str, TaskConfig], dict[str, KBConfig]]:
    """Load the operator registry from a TOML file."""

    path = Path(path)

    if not path.exists():
        logger.warning("Operator registry not found at %s — using empty registry", path)
        return {}, {}

    if path.suffix.lower() != ".toml":
        raise ValueError(f"Operator registry must be a TOML file (got '{path.name}')")

    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    return _materialize_operator_registry(OperatorRegistryConfig(**raw))


@lru_cache(maxsize=None)
def _load_knowledge_bases_cached(
    path: str,
) -> tuple[dict[str, TaskConfig], dict[str, KBConfig]]:
    """Cached path-based loader for the materialized runtime registry."""

    return load_knowledge_bases(path)


_KB_OVERRIDE_REGISTRY: dict[str, TaskConfig] | None = None
_KB_OVERRIDE_INDEX: dict[str, KBConfig] | None = None


def _build_kb_index(registry: dict[str, TaskConfig]) -> dict[str, KBConfig]:
    """Build a flat KB index from a task-scoped runtime registry."""

    index: dict[str, KBConfig] = {}
    for task_cfg in registry.values():
        for kb_cfg in task_cfg.knowledge_bases:
            if kb_cfg.name in index:
                if index[kb_cfg.name] is not kb_cfg:
                    raise ValueError(
                        f"Duplicate KB name '{kb_cfg.name}' found across tasks. "
                        f"KB names must be unique."
                    )
                continue
            index[kb_cfg.name] = kb_cfg
    return index


def _restore_knowledge_base_registry_override(
    registry: dict[str, TaskConfig] | None,
    index: dict[str, KBConfig] | None,
) -> None:
    global _KB_OVERRIDE_REGISTRY, _KB_OVERRIDE_INDEX  # noqa: PLW0603

    _KB_OVERRIDE_REGISTRY = registry
    _KB_OVERRIDE_INDEX = index


def set_knowledge_base_registry_override(
    registry: dict[str, TaskConfig],
    *,
    index: dict[str, KBConfig] | None = None,
) -> None:
    """Install an in-memory registry override used ahead of disk-backed loading."""

    _restore_knowledge_base_registry_override(
        registry,
        index if index is not None else _build_kb_index(registry),
    )


def clear_knowledge_base_registry_override() -> None:
    """Remove any installed in-memory registry override."""

    _restore_knowledge_base_registry_override(None, None)


@contextmanager
def registry_override(
    registry: dict[str, TaskConfig],
    *,
    index: dict[str, KBConfig] | None = None,
) -> Iterator[None]:
    """Temporarily install an in-memory registry override and restore prior state."""

    previous_registry = _KB_OVERRIDE_REGISTRY
    previous_index = _KB_OVERRIDE_INDEX
    set_knowledge_base_registry_override(registry, index=index)
    try:
        yield
    finally:
        _restore_knowledge_base_registry_override(previous_registry, previous_index)


def _configured_operator_registry_path(settings: RegistryPathSettings | None = None) -> Path | None:
    """Resolve the configured operator registry path before path normalization."""

    if settings is not None:
        return settings.operator_registry_path

    raw_value = os.environ.get(_NESTED_OPERATOR_REGISTRY_PATH_ENV)
    if raw_value is None:
        return None

    stripped = raw_value.strip()
    if not stripped:
        return None
    return Path(stripped)


def resolve_knowledge_bases_path(settings: RegistryPathSettings | None = None) -> Path:
    """Resolve the active knowledge-base registry path."""

    configured_path = _configured_operator_registry_path(settings)
    if configured_path is None:
        return _DEFAULT_OPERATOR_REGISTRY_PATH

    path = configured_path.expanduser()
    if path.is_absolute():
        return path

    try:
        return get_repo_root() / path
    except FileNotFoundError:
        return path


def _get_loaded_knowledge_base_state(
    *, settings: RegistryPathSettings | None = None
) -> tuple[dict[str, TaskConfig], dict[str, KBConfig]]:
    """Return the effective runtime registry and flat index."""

    if _KB_OVERRIDE_REGISTRY is not None and _KB_OVERRIDE_INDEX is not None:
        return _KB_OVERRIDE_REGISTRY, _KB_OVERRIDE_INDEX

    path = resolve_knowledge_bases_path(settings).resolve()
    return _load_knowledge_bases_cached(str(path))


def get_knowledge_bases(*, settings: RegistryPathSettings | None = None) -> dict[str, TaskConfig]:
    """Return the knowledge-base registry (cached after first call)."""

    registry, _ = _get_loaded_knowledge_base_state(settings=settings)
    return registry


def get_kb_config(
    kb_name: str,
    *,
    settings: RegistryPathSettings | None = None,
) -> KBConfig | None:
    """Look up a KB by name (O(1) dict lookup)."""

    _, index = _get_loaded_knowledge_base_state(settings=settings)
    return index.get(kb_name)


def get_kb_names(*, settings: RegistryPathSettings | None = None) -> list[str]:
    """Flat list of all KB names across all tasks."""

    _, index = _get_loaded_knowledge_base_state(settings=settings)
    return list(index.keys())


def validate_kb_alias(
    kb: str,
    alias: str | None = None,
    *,
    settings: RegistryPathSettings | None = None,
) -> None:
    """Raise ValueError with a consistent message if kb or alias is unknown."""

    kb_cfg = get_kb_config(kb, settings=settings)
    if kb_cfg is None:
        raise ValueError(f"KB '{kb}' not found. Available: {get_kb_names(settings=settings)}")
    if alias is not None and alias not in kb_cfg.aliases:
        raise ValueError(
            f"Alias '{alias}' not valid for KB '{kb}'. Available: {list(kb_cfg.aliases.keys())}"
        )


def clear_operator_registry_caches() -> None:
    """Clear disk-backed cache and in-memory overrides for the operator registry."""

    clear_knowledge_base_registry_override()
    _load_knowledge_bases_cached.cache_clear()


# Backward-compatible aliases kept while callers migrate off underscored names.
_load_knowledge_bases = load_knowledge_bases
_resolve_knowledge_bases_path = resolve_knowledge_bases_path


__all__ = [
    "AdapterConfig",
    "AliasConfig",
    "KBConfig",
    "OperatorRegistryConfig",
    "RegistryKBConfig",
    "RegistryAliasConfig",
    "SourceConfig",
    "RegistryTaskConfig",
    "TaskConfig",
    "clear_knowledge_base_registry_override",
    "clear_operator_registry_caches",
    "get_kb_config",
    "get_kb_names",
    "get_knowledge_bases",
    "load_knowledge_bases",
    "registry_override",
    "resolve_knowledge_bases_path",
    "set_knowledge_base_registry_override",
    "validate_kb_alias",
]
