"""Lazy application-service factories for the `rag` CLI.

Typer command functions call these to construct real services from runtime
settings. Tests monkeypatch the factory functions themselves to inject fakes
instead of constructing services that need live Postgres/Qdrant/provider
connections (`typer.testing.CliRunner` + injected services, not real ones).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app_config.catalog import build_source_instance_index, read_catalog_config
from app_config.catalog.schema import CatalogConfig
from app_config.runtime import get_settings
from clients.db.eval_writer import check_evaluation_coverage
from rag.adapters import SourceAdapterRegistry, build_catalog_adapter_registry
from rag.control_plane.alias_service import AliasService
from rag.control_plane.postgres import (
    PostgresAliasDeploymentRepository,
    PostgresReleaseBuildRepository,
    PostgresReleaseRepository,
    create_session_factory,
)
from rag.embeddings import EmbeddingService
from rag.indexing.llamaindex_qdrant import QdrantCollectionManager
from rag.indexing.materialize import qdrant_alias_name
from rag.reranker import get_reranker
from rag.sparse_encoder import SparseEncoderService


@dataclass(frozen=True, slots=True)
class RagContext:
    """Global CLI options, shared by every command.

    `catalog_path`/`data_root` resolve `get_settings()` lazily on first
    access rather than at construction time: the root callback that builds
    this object runs even for `rag <group> <command> --help`, and `--help`
    must never need a configured settings environment to render.
    """

    catalog_path_override: Path | None
    data_root_override: Path | None
    as_json: bool

    @property
    def catalog_path(self) -> Path:
        if self.catalog_path_override is not None:
            return self.catalog_path_override
        return get_settings().catalog.path

    @property
    def data_root(self) -> Path:
        if self.data_root_override is not None:
            return self.data_root_override
        return get_settings().rag.data_root


def load_catalog_config(ctx: RagContext) -> CatalogConfig:
    """Read and validate the catalog TOML at the resolved path."""
    return read_catalog_config(ctx.catalog_path)


def build_adapter_registry(catalog_cfg: CatalogConfig) -> SourceAdapterRegistry:
    return build_catalog_adapter_registry(catalog_cfg)


def _require_db_url() -> str:
    settings = get_settings()
    db_url = settings.auth.agent042_db_url
    if not db_url:
        raise RuntimeError(
            "agent042 database URL is not configured; the control plane requires Postgres"
        )
    return db_url


def build_alias_service(
    ctx: RagContext, *, catalog_cfg: CatalogConfig | None = None
) -> AliasService:
    """Construct a real AliasService wired to Postgres, Qdrant, and provider services."""
    settings = get_settings()
    platform = settings.platform
    catalog_cfg = catalog_cfg or load_catalog_config(ctx)
    session_factory = create_session_factory(_require_db_url())
    adapter_registry = build_adapter_registry(catalog_cfg)
    source_index = build_source_instance_index(catalog_cfg)

    def _manager_factory(collection_name: str) -> QdrantCollectionManager:
        return QdrantCollectionManager.connect(
            host=platform.qdrant_host,
            port=platform.qdrant_port,
            collection_name=collection_name,
        )

    def _alias_updater(kb_id: str, alias: str, collection_name: str) -> None:
        manager = _manager_factory(collection_name)
        try:
            manager.update_alias(qdrant_alias_name(kb_id=kb_id, alias=alias), collection_name)
        finally:
            manager.close()

    def _evaluation_coverage_checker(kb_id: str, release_id: str, retrieval_digest: str) -> bool:
        benchmark_ids = [instance.id for instance in source_index.benchmark_for_kb(kb_id)]
        return check_evaluation_coverage(
            db_url=settings.auth.agent042_db_url,
            release_id=release_id,
            retrieval_config_digest=retrieval_digest,
            benchmark_source_instance_ids=benchmark_ids,
        )

    return AliasService(
        catalog_cfg=catalog_cfg,
        rag_data_root=ctx.data_root,
        release_build_repo=PostgresReleaseBuildRepository(session_factory),
        release_repo=PostgresReleaseRepository(session_factory),
        deployment_repo=PostgresAliasDeploymentRepository(session_factory),
        collection_manager_factory=_manager_factory,
        qdrant_alias_updater=_alias_updater,
        embedding_client_factory=EmbeddingService,
        sparse_encoder_client_factory=lambda: SparseEncoderService(
            embeddings_url=platform.embeddings_url
        ),
        reranker_client_factory=get_reranker,
        adapter_registry=adapter_registry,
        evaluation_coverage_checker=_evaluation_coverage_checker,
    )


def build_release_repository(ctx: RagContext) -> PostgresReleaseRepository:
    return PostgresReleaseRepository(create_session_factory(_require_db_url()))


def build_deployment_repository(ctx: RagContext) -> PostgresAliasDeploymentRepository:
    return PostgresAliasDeploymentRepository(create_session_factory(_require_db_url()))
