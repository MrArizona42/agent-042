"""Adapter capabilities and source-build context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from llama_index.core import Document

from rag.evaluation.models import BenchmarkPreparedArtifacts

AdapterCapability = Literal["source", "benchmark"]


@dataclass(frozen=True, slots=True)
class SourceAdapterContext:
    """Project identity supplied to an adapter for one source build."""

    kb_id: str
    source_instance_id: str
    manifest_digest: str


class SourceAdapter(Protocol):
    """Adapter contract for one source ingest family."""

    adapter_id: str
    version: str
    capabilities: frozenset[AdapterCapability]

    def validate_manifest(self, manifest: Any) -> Any:
        """Validate an already-loaded source manifest."""
        ...

    def list_documents(
        self,
        manifest: Any,
        *,
        context: SourceAdapterContext,
    ) -> list[Document]:
        """Return LlamaIndex documents declared by the manifest."""
        ...

    def fetcher(self):
        """Return the fetcher selected by this adapter."""
        ...

    def extractor(self):
        """Return the extractor selected by this adapter."""
        ...


class BenchmarkAdapter(SourceAdapter, Protocol):
    """A source adapter that also implements benchmark preparation."""

    def prepare_benchmark(self, manifest: Any) -> BenchmarkPreparedArtifacts:
        """Emit normalized benchmark cases and labels for this manifest."""
        ...


class SourceAdapterRegistry:
    """Explicit adapter registry used by tests and injected runtimes."""

    def __init__(self) -> None:
        self._adapters: dict[tuple[str, str], SourceAdapter] = {}

    def register(self, adapter: SourceAdapter) -> None:
        key = (adapter.adapter_id, adapter.version)
        if key in self._adapters:
            raise ValueError(
                f"Source adapter '{adapter.adapter_id}@{adapter.version}' already registered"
            )
        self._adapters[key] = adapter

    def get(self, adapter_id: str, *, version: str = "1") -> SourceAdapter:
        key = (adapter_id, version)
        adapter = self._adapters.get(key)
        if adapter is None:
            raise ValueError(f"Unknown source adapter '{adapter_id}@{version}'")
        return adapter
