"""Loads source/benchmark adapters declared in the catalog via factory references."""

from __future__ import annotations

import importlib
from typing import Any, Callable

from app_config.catalog import BenchmarkAdapterConfig, CatalogConfig, SourceAdapterConfig
from rag.ingest.adapters import AdapterCapability, SourceAdapter, SourceAdapterRegistry

_REQUIRED_SOURCE_METHODS = ("validate_manifest", "list_documents", "fetcher", "extractor")


def import_factory(factory_ref: str) -> Callable[[], Any]:
    """Import a `module:function` factory reference."""
    module_name, sep, function_name = factory_ref.partition(":")
    if not sep or not module_name or not function_name:
        raise ValueError(f"Adapter factory '{factory_ref}' must be in 'module:function' form")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"Cannot import adapter factory module '{module_name}': {exc}") from exc
    factory = getattr(module, function_name, None)
    if factory is None or not callable(factory):
        raise ValueError(
            f"Adapter factory module '{module_name}' has no callable '{function_name}'"
        )
    return factory


def _validate_adapter(
    adapter: Any,
    *,
    config: SourceAdapterConfig | BenchmarkAdapterConfig,
    required_capabilities: frozenset[AdapterCapability],
) -> SourceAdapter:
    adapter_ref = f"{config.id}@{config.version}"

    adapter_id = getattr(adapter, "adapter_id", None)
    version = getattr(adapter, "version", None)
    if adapter_id != config.id or version != config.version:
        raise ValueError(
            f"Adapter factory '{config.factory}' for declared adapter '{adapter_ref}' "
            f"returned an adapter identified as '{adapter_id}@{version}'"
        )

    capabilities = getattr(adapter, "capabilities", None)
    if capabilities is None:
        raise ValueError(f"Adapter '{adapter_ref}' has no 'capabilities' attribute")
    missing = required_capabilities - frozenset(capabilities)
    if missing:
        raise ValueError(
            f"Adapter '{adapter_ref}' is missing required capabilities {sorted(missing)}; "
            f"has {sorted(capabilities)}"
        )

    required_methods = _REQUIRED_SOURCE_METHODS
    if "benchmark" in required_capabilities:
        required_methods = (*required_methods, "prepare_benchmark")
    for method_name in required_methods:
        method = getattr(adapter, method_name, None)
        if method is None or not callable(method):
            raise ValueError(f"Adapter '{adapter_ref}' is missing callable '{method_name}'")

    return adapter


def load_adapter(
    config: SourceAdapterConfig | BenchmarkAdapterConfig,
    *,
    required_capabilities: frozenset[AdapterCapability],
) -> SourceAdapter:
    """Import and call a declared adapter factory, validating its capabilities."""
    factory = import_factory(config.factory)
    try:
        adapter = factory()
    except Exception as exc:
        raise ValueError(
            f"Adapter factory '{config.factory}' for '{config.id}@{config.version}' raised: {exc}"
        ) from exc
    return _validate_adapter(
        adapter,
        config=config,
        required_capabilities=required_capabilities,
    )


def build_catalog_adapter_registry(catalog_cfg: CatalogConfig) -> SourceAdapterRegistry:
    """Build an adapter registry from declared `[[source_adapters]]`/`[[benchmark_adapters]]`."""
    registry = SourceAdapterRegistry()
    for source_adapter_cfg in catalog_cfg.source_adapters:
        registry.register(
            load_adapter(source_adapter_cfg, required_capabilities=frozenset({"source"}))
        )
    for benchmark_adapter_cfg in catalog_cfg.benchmark_adapters:
        registry.register(
            load_adapter(
                benchmark_adapter_cfg,
                required_capabilities=frozenset({"source", "benchmark"}),
            )
        )
    return registry
