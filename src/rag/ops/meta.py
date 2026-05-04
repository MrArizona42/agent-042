"""Metadata models and validation helpers for RAG collections."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Mapping

from rag.vector_store import QdrantVectorStore

QueryStrategy = Literal["dense", "hybrid", "sparse"]


def _require_non_empty_str(value: Any, *, field_name: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: '{field_name}' must be a non-empty string")
    return value.strip()


def _require_int(value: Any, *, field_name: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}: '{field_name}' must be an integer")
    return value


@dataclass(frozen=True, slots=True)
class BuildConfig:
    """Validated build-time configuration stored in collection metadata."""

    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    sparse_encoder: str | None
    retrieval_capability: Literal["dense", "hybrid", "sparse"]

    def __post_init__(self) -> None:
        if not self.chunking_strategy.strip():
            raise ValueError("BuildConfig.chunking_strategy must be a non-empty string")
        if self.chunk_size <= 0:
            raise ValueError("BuildConfig.chunk_size must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("BuildConfig.chunk_overlap must be zero or greater")
        if not self.embedding_model.strip():
            raise ValueError("BuildConfig.embedding_model must be a non-empty string")

    def to_payload(self) -> dict[str, Any]:
        return {
            "chunking_strategy": self.chunking_strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "sparse_encoder": self.sparse_encoder,
            "retrieval_capability": self.retrieval_capability,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        context: str = "build_config",
    ) -> "BuildConfig":
        retrieval_capability = payload.get("retrieval_capability")
        if retrieval_capability not in ("dense", "hybrid", "sparse"):
            raise ValueError(
                f"{context}: 'retrieval_capability' must be one of "
                f"'dense', 'hybrid', 'sparse' (got {retrieval_capability!r})"
            )

        return cls(
            chunking_strategy=_require_non_empty_str(
                payload.get("chunking_strategy"),
                field_name="chunking_strategy",
                context=context,
            ),
            chunk_size=_require_int(
                payload.get("chunk_size"),
                field_name="chunk_size",
                context=context,
            ),
            chunk_overlap=_require_int(
                payload.get("chunk_overlap"),
                field_name="chunk_overlap",
                context=context,
            ),
            embedding_model=_require_non_empty_str(
                payload.get("embedding_model"),
                field_name="embedding_model",
                context=context,
            ),
            sparse_encoder=payload.get("sparse_encoder"),
            retrieval_capability=retrieval_capability,
        )


def validate_query_compatibility(
    *,
    query_strategy: QueryStrategy,
    build_config: BuildConfig,
    runtime_sparse_encoder: str | None = None,
    context: str = "build_config",
) -> None:
    """Validate that a query-time strategy is compatible with build metadata."""
    build_capability = build_config.retrieval_capability
    has_dense_leg = build_capability in ("dense", "hybrid")
    has_sparse_leg = build_capability in ("hybrid", "sparse")

    if query_strategy == "dense":
        if not has_dense_leg:
            raise ValueError(
                f"{context}: query strategy 'dense' requires a dense leg, "
                f"but build capability is '{build_capability}'"
            )
        return

    if query_strategy == "hybrid":
        if build_capability != "hybrid":
            raise ValueError(
                f"{context}: query strategy 'hybrid' requires build capability "
                f"'hybrid' (got '{build_capability}')"
            )
        _validate_sparse_encoder_compatibility(
            build_config=build_config,
            runtime_sparse_encoder=runtime_sparse_encoder,
            query_strategy=query_strategy,
            context=context,
        )
        return

    if query_strategy == "sparse":
        if not has_sparse_leg:
            raise ValueError(
                f"{context}: query strategy 'sparse' requires a sparse leg, "
                f"but build capability is '{build_capability}'"
            )
        _validate_sparse_encoder_compatibility(
            build_config=build_config,
            runtime_sparse_encoder=runtime_sparse_encoder,
            query_strategy=query_strategy,
            context=context,
        )
        return

    raise ValueError(f"{context}: unsupported query strategy {query_strategy!r}")


def _validate_sparse_encoder_compatibility(
    *,
    build_config: BuildConfig,
    runtime_sparse_encoder: str | None,
    query_strategy: QueryStrategy,
    context: str,
) -> None:
    build_sparse_encoder = build_config.sparse_encoder
    if not isinstance(build_sparse_encoder, str) or not build_sparse_encoder.strip():
        raise ValueError(
            f"{context}: query strategy '{query_strategy}' requires collection metadata "
            "with a non-empty sparse_encoder"
        )

    if not isinstance(runtime_sparse_encoder, str) or not runtime_sparse_encoder.strip():
        raise ValueError(
            f"{context}: query strategy '{query_strategy}' requires the runtime "
            "SPARSE_ENCODER_MODEL configuration"
        )

    runtime_sparse_encoder = runtime_sparse_encoder.strip()
    if runtime_sparse_encoder != build_sparse_encoder:
        raise ValueError(
            f"{context}: runtime sparse encoder '{runtime_sparse_encoder}' does not "
            f"match build sparse encoder '{build_sparse_encoder}'"
        )


@dataclass(frozen=True, slots=True)
class ImplementationInfo:
    """Identity of the implementation that built a collection."""

    module: str
    experimental: bool = False
    identifier: str | None = None
    git_sha: str | None = None

    def __post_init__(self) -> None:
        if not self.module.strip():
            raise ValueError("ImplementationInfo.module must be a non-empty string")

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "module": self.module,
            "experimental": self.experimental,
        }
        if self.identifier:
            payload["identifier"] = self.identifier
        if self.git_sha:
            payload["git_sha"] = self.git_sha
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        context: str = "implementation",
    ) -> "ImplementationInfo":
        return cls(
            module=_require_non_empty_str(
                payload.get("module"),
                field_name="module",
                context=context,
            ),
            experimental=bool(payload.get("experimental", False)),
            identifier=payload.get("identifier") or None,
            git_sha=payload.get("git_sha") or None,
        )


@dataclass(frozen=True, slots=True)
class CollectionMeta:
    """Validated collection metadata payload."""

    kb_name: str
    build_config: BuildConfig
    created_at: str
    implementation: ImplementationInfo | None = None

    def __post_init__(self) -> None:
        if not self.kb_name.strip():
            raise ValueError("CollectionMeta.kb_name must be a non-empty string")
        if not self.created_at.strip():
            raise ValueError("CollectionMeta.created_at must be a non-empty string")

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "build_config": self.build_config.to_payload(),
            "kb_name": self.kb_name,
            "created_at": self.created_at,
        }
        if self.implementation is not None:
            payload["implementation"] = self.implementation.to_payload()
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        context: str = "collection_meta",
    ) -> "CollectionMeta":
        build_payload = payload.get("build_config")
        if not isinstance(build_payload, Mapping):
            raise ValueError(f"{context}: 'build_config' must be an object")

        implementation_payload = payload.get("implementation")
        implementation = None
        if implementation_payload is not None:
            if not isinstance(implementation_payload, Mapping):
                raise ValueError(f"{context}: 'implementation' must be an object")
            implementation = ImplementationInfo.from_payload(
                implementation_payload,
                context=f"{context}.implementation",
            )

        return cls(
            kb_name=_require_non_empty_str(
                payload.get("kb_name"),
                field_name="kb_name",
                context=context,
            ),
            build_config=BuildConfig.from_payload(
                build_payload,
                context=f"{context}.build_config",
            ),
            created_at=_require_non_empty_str(
                payload.get("created_at"),
                field_name="created_at",
                context=context,
            ),
            implementation=implementation,
        )


def build_collection_meta(
    *,
    kb_name: str,
    build_config: BuildConfig,
    implementation: ImplementationInfo | None = None,
    created_at: str | None = None,
) -> CollectionMeta:
    """Create validated collection metadata for a fresh materialization."""
    return CollectionMeta(
        kb_name=kb_name,
        build_config=build_config,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        implementation=implementation,
    )


def read_collection_meta(
    vector_store: QdrantVectorStore,
    *,
    context: str | None = None,
) -> CollectionMeta:
    """Read and validate collection metadata from Qdrant."""
    payload = vector_store.read_meta()
    context = context or vector_store.collection_name
    if payload is None:
        raise RuntimeError(f"Missing _meta for '{context}'")
    return CollectionMeta.from_payload(payload, context=context)


def write_collection_meta(
    vector_store: QdrantVectorStore,
    meta: CollectionMeta,
    *,
    dimension: int,
) -> None:
    """Write validated collection metadata to Qdrant."""
    vector_store.write_meta(payload=meta.to_payload(), dimension=dimension)


def read_build_config_for_alias(
    *,
    kb_name: str,
    rag_alias: str,
    qdrant_host: str,
    qdrant_port: int,
) -> BuildConfig:
    """Read validated build config from a production alias."""
    alias_name = f"{kb_name}_{rag_alias}"
    vector_store = QdrantVectorStore(
        host=qdrant_host,
        port=qdrant_port,
        collection_name=alias_name,
    )
    if not vector_store.collection_exists():
        raise RuntimeError(f"Alias '{alias_name}' does not resolve in Qdrant")
    return read_collection_meta(vector_store, context=alias_name).build_config
