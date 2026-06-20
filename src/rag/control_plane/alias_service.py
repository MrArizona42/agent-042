"""Alias diff and apply: desired (catalog) vs applied (Postgres) state reconciliation.

`AliasService.diff()` is read-only. `AliasService.apply()` is the mutation
operation -- it may validate, reuse, or build a release, then activate a
deployment. There is no separate "promotion" operation; reapplying the
challenger's accepted state to the default alias reuses the same evaluated
release rather than rebuilding it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from app_config.catalog.schema import CatalogAliasConfig, CatalogConfig, CatalogKBConfig
from app_config.catalog.source_instances import build_source_instance_index
from rag.adapters import SourceAdapterRegistry
from rag.contracts.manifests import compare_release_attestation, release_manifest_path
from rag.control_plane import fingerprints as fp
from rag.control_plane.models import AliasDeployment, AliasDiff, RagRelease, ReleaseBuildAttempt
from rag.control_plane.provider_validation import (
    validate_build_provider_identity,
    validate_retrieval_provider_identity,
)
from rag.control_plane.release_builder import build_release, source_declaration
from rag.control_plane.repositories import (
    AliasDeploymentRepository,
    ReleaseBuildRepository,
    ReleaseRepository,
)

logger = logging.getLogger(__name__)

ApplyAction = Literal["no_drift", "retrieval_only", "reused_release", "built_release"]


class AliasApplyError(RuntimeError):
    """An alias apply request was refused or failed. Not necessarily transient."""


class AliasDiffRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str
    alias: str


class AliasApplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kb_id: str
    alias: str
    release_id: str | None = None
    allow_unevaluated: bool = False
    allow_build_default: bool = False


class AliasApplyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deployment: AliasDeployment
    release: RagRelease
    action: ApplyAction


class AliasService:
    """Desired/applied-state reconciliation for one catalog and Postgres registry."""

    def __init__(
        self,
        *,
        catalog_cfg: CatalogConfig,
        rag_data_root: Path | str,
        release_build_repo: ReleaseBuildRepository,
        release_repo: ReleaseRepository,
        deployment_repo: AliasDeploymentRepository,
        collection_manager_factory: Callable[[str], object],
        qdrant_alias_updater: Callable[[str, str, str], None],
        embedding_client_factory: Callable[[], object],
        sparse_encoder_client_factory: Callable[[], object] | None = None,
        reranker_client_factory: Callable[[str], object] | None = None,
        adapter_registry: SourceAdapterRegistry | None = None,
        evaluation_coverage_checker: Callable[[str, str, str], bool] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._catalog_cfg = catalog_cfg
        self._rag_data_root = rag_data_root
        self._release_build_repo = release_build_repo
        self._release_repo = release_repo
        self._deployment_repo = deployment_repo
        self._collection_manager_factory = collection_manager_factory
        self._qdrant_alias_updater = qdrant_alias_updater
        self._embedding_client_factory = embedding_client_factory
        self._sparse_encoder_client_factory = sparse_encoder_client_factory
        self._reranker_client_factory = reranker_client_factory
        self._adapter_registry = adapter_registry
        self._evaluation_coverage_checker = evaluation_coverage_checker
        self._clock = clock
        self._source_index = build_source_instance_index(catalog_cfg)

    # ------------------------------------------------------------------
    # Catalog lookups
    # ------------------------------------------------------------------

    def _kb_config(self, kb_id: str) -> CatalogKBConfig:
        for kb in self._catalog_cfg.knowledge_bases:
            if kb.id == kb_id:
                return kb
        raise AliasApplyError(f"Unknown KB '{kb_id}'")

    def _alias_config(self, kb_cfg: CatalogKBConfig, alias: str) -> CatalogAliasConfig:
        alias_cfg = kb_cfg.aliases.get(alias)
        if alias_cfg is None:
            raise AliasApplyError(f"Unknown alias '{alias}' for KB '{kb_cfg.id}'")
        return alias_cfg

    def _source_declaration_digest(self, kb_id: str) -> str:
        source_instance_ids = [instance.id for instance in self._source_index.corpus_for_kb(kb_id)]
        digest, _manifest_digests, _adapter_versions = source_declaration(
            source_instance_ids=source_instance_ids,
            source_index=self._source_index,
            rag_data_root=self._rag_data_root,
        )
        return digest

    def _provider_mismatches(self, alias_cfg: CatalogAliasConfig) -> list[str]:
        embedding_client = self._embedding_client_factory()
        sparse_client = (
            self._sparse_encoder_client_factory()
            if alias_cfg.build.sparse_encoder is not None and self._sparse_encoder_client_factory
            else None
        )
        reranker_client = (
            self._reranker_client_factory(alias_cfg.retrieve.reranker)
            if alias_cfg.retrieve.reranker is not None and self._reranker_client_factory
            else None
        )
        return validate_build_provider_identity(
            alias_cfg.build,
            embedding_client=embedding_client,
            sparse_encoder_client=sparse_client,
        ) + validate_retrieval_provider_identity(
            alias_cfg.retrieve,
            reranker_client=reranker_client,
        )

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    def diff(self, request: AliasDiffRequest) -> AliasDiff:
        kb_cfg = self._kb_config(request.kb_id)
        alias_cfg = self._alias_config(kb_cfg, request.alias)

        desired_build_digest = fp.build_config_digest(alias_cfg.build)
        desired_retrieval_digest = fp.retrieval_config_digest(alias_cfg.retrieve)
        desired_catalog_digest = fp.catalog_digest(alias_cfg.build, alias_cfg.retrieve)
        desired_source_declaration_digest = self._source_declaration_digest(request.kb_id)

        active = self._deployment_repo.get_active(kb_id=request.kb_id, alias=request.alias)
        active_release = self._release_repo.get(active.release_id) if active else None

        build_drift = active is None or active.build_config_digest != desired_build_digest
        retrieval_drift = (
            active is None or active.retrieval_config_digest != desired_retrieval_digest
        )
        source_declaration_drift = (
            active_release is None
            or active_release.source_declaration_digest != desired_source_declaration_digest
        )

        reusable_release_ids: list[str] = []
        if build_drift or source_declaration_drift:
            reusable_release_ids = [
                release.id
                for release in self._release_repo.find_reusable(
                    build_config_digest=desired_build_digest,
                    source_declaration_digest=desired_source_declaration_digest,
                )
            ]

        return AliasDiff(
            kb_id=request.kb_id,
            alias=request.alias,
            desired_catalog_digest=desired_catalog_digest,
            desired_build_config_digest=desired_build_digest,
            desired_retrieval_config_digest=desired_retrieval_digest,
            applied_deployment_id=active.id if active else None,
            applied_release_id=active.release_id if active else None,
            build_drift=build_drift,
            retrieval_drift=retrieval_drift,
            source_declaration_drift=source_declaration_drift,
            provider_mismatches=self._provider_mismatches(alias_cfg),
            reusable_release_ids=reusable_release_ids,
        )

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(self, request: AliasApplyRequest) -> AliasApplyResult:
        kb_cfg = self._kb_config(request.kb_id)
        alias_cfg = self._alias_config(kb_cfg, request.alias)
        is_default_alias = kb_cfg.default_alias == request.alias

        diff = self.diff(AliasDiffRequest(kb_id=request.kb_id, alias=request.alias))

        if diff.provider_mismatches:
            raise AliasApplyError(
                f"provider identity mismatch, refusing to apply: {diff.provider_mismatches}"
            )

        if not diff.build_drift and not diff.source_declaration_drift and not diff.retrieval_drift:
            active = self._deployment_repo.get_active(kb_id=request.kb_id, alias=request.alias)
            release = self._release_repo.get(active.release_id)
            return AliasApplyResult(deployment=active, release=release, action="no_drift")

        if not diff.build_drift and not diff.source_declaration_drift:
            active = self._deployment_repo.get_active(kb_id=request.kb_id, alias=request.alias)
            release = self._release_repo.get(active.release_id)
            return self._activate(
                kb_id=request.kb_id,
                alias=request.alias,
                release=release,
                alias_cfg=alias_cfg,
                diff=diff,
                action="retrieval_only",
            )

        release, built = self._resolve_release_for_apply(
            request=request,
            diff=diff,
            alias=request.alias,
            alias_cfg=alias_cfg,
            is_default_alias=is_default_alias,
        )

        needs_sparse = alias_cfg.retrieve.strategy in ("hybrid", "sparse")
        if needs_sparse and release.build_config.sparse_encoder is None:
            raise AliasApplyError(
                f"retrieve.strategy '{alias_cfg.retrieve.strategy}' is incompatible with "
                f"release '{release.id}', which has no sparse encoder"
            )

        if is_default_alias:
            self._enforce_default_alias_evaluation_gate(
                release=release, alias_cfg=alias_cfg, request=request
            )

        return self._activate(
            kb_id=request.kb_id,
            alias=request.alias,
            release=release,
            alias_cfg=alias_cfg,
            diff=diff,
            action="built_release" if built else "reused_release",
        )

    def _resolve_release_for_apply(
        self,
        *,
        request: AliasApplyRequest,
        diff: AliasDiff,
        alias: str,
        alias_cfg: CatalogAliasConfig,
        is_default_alias: bool,
    ) -> tuple[RagRelease, bool]:
        desired_source_declaration_digest = self._source_declaration_digest(request.kb_id)

        if request.release_id is not None:
            release = self._release_repo.get(request.release_id)
            if release is None:
                raise AliasApplyError(f"release '{request.release_id}' not found")
            if release.build_config_digest != diff.desired_build_config_digest:
                raise AliasApplyError(
                    f"explicit release '{release.id}' does not match desired build configuration"
                )
            if release.source_declaration_digest != desired_source_declaration_digest:
                raise AliasApplyError(
                    f"explicit release '{release.id}' does not match desired source declaration"
                )
            return release, False

        candidates = self._release_repo.find_reusable(
            build_config_digest=diff.desired_build_config_digest,
            source_declaration_digest=desired_source_declaration_digest,
        )
        if len(candidates) > 1:
            raise AliasApplyError(
                "multiple releases match the desired build and source state "
                f"({sorted(c.id for c in candidates)}); disambiguate with an explicit release_id"
            )
        if len(candidates) == 1:
            return candidates[0], False

        if is_default_alias and not request.allow_build_default:
            raise AliasApplyError(
                "default alias apply refuses to silently build an unevaluated release; "
                "apply a non-default alias first, or pass allow_build_default to override "
                "(recorded as a bootstrap/emergency action)"
            )

        release = self._build_release(
            kb_id=request.kb_id,
            alias=alias,
            alias_cfg=alias_cfg,
            catalog_digest=diff.desired_catalog_digest,
        )
        return release, True

    def _build_release(
        self,
        *,
        kb_id: str,
        alias: str,
        alias_cfg: CatalogAliasConfig,
        catalog_digest: str,
    ) -> RagRelease:
        build_digest = fp.build_config_digest(alias_cfg.build)
        retrieval_digest = fp.retrieval_config_digest(alias_cfg.retrieve)
        source_declaration_digest = self._source_declaration_digest(kb_id)

        attempt = ReleaseBuildAttempt(
            id=uuid4(),
            kb_id=kb_id,
            requested_alias=alias,
            status="running",
            catalog_digest=catalog_digest,
            build_config_digest=build_digest,
            retrieval_config_digest=retrieval_digest,
            source_declaration_digest=source_declaration_digest,
            started_at=self._clock(),
        )
        self._release_build_repo.create(attempt)

        sparse_client = (
            self._sparse_encoder_client_factory()
            if alias_cfg.build.sparse_encoder is not None and self._sparse_encoder_client_factory
            else None
        )
        try:
            release = build_release(
                kb_id=kb_id,
                build_config=alias_cfg.build,
                catalog_digest=catalog_digest,
                catalog_cfg=self._catalog_cfg,
                rag_data_root=self._rag_data_root,
                collection_manager_factory=self._collection_manager_factory,
                embedding_client=self._embedding_client_factory(),
                sparse_encoder_client=sparse_client,
                adapter_registry=self._adapter_registry,
            )
        except Exception as exc:
            self._release_build_repo.mark_failed(
                attempt.id, error=str(exc), finished_at=self._clock()
            )
            raise AliasApplyError(f"release build failed: {exc}") from exc

        if self._release_repo.get(release.id) is None:
            manifest_path = release_manifest_path(
                rag_data_root=self._rag_data_root, kb_id=kb_id, release_id=release.id
            )
            self._release_repo.insert(release, manifest_path=manifest_path.as_posix())

        self._release_build_repo.mark_completed(
            attempt.id,
            release_id=release.id,
            collection_name=release.collection_name,
            source_snapshot_id=release.source_snapshot_id,
            finished_at=self._clock(),
        )
        return release

    def _enforce_default_alias_evaluation_gate(
        self,
        *,
        release: RagRelease,
        alias_cfg: CatalogAliasConfig,
        request: AliasApplyRequest,
    ) -> None:
        if request.allow_unevaluated:
            return
        retrieval_digest = fp.retrieval_config_digest(alias_cfg.retrieve)
        evaluated = (
            self._evaluation_coverage_checker(request.kb_id, release.id, retrieval_digest)
            if self._evaluation_coverage_checker is not None
            else False
        )
        if not evaluated:
            raise AliasApplyError(
                f"release '{release.id}' has no evaluation coverage for this retrieval "
                "configuration; pass allow_unevaluated to override as a bootstrap/emergency action"
            )

    def _activate(
        self,
        *,
        kb_id: str,
        alias: str,
        release: RagRelease,
        alias_cfg: CatalogAliasConfig,
        diff: AliasDiff,
        action: ApplyAction,
    ) -> AliasApplyResult:
        active = self._deployment_repo.get_active(kb_id=kb_id, alias=alias)
        if (
            active is not None
            and active.release_id == release.id
            and active.retrieval_config_digest == diff.desired_retrieval_config_digest
            and active.catalog_digest == diff.desired_catalog_digest
        ):
            return AliasApplyResult(deployment=active, release=release, action="no_drift")

        deployment = AliasDeployment(
            id=uuid4(),
            kb_id=kb_id,
            alias=alias,
            release_id=release.id,
            collection_name=release.collection_name,
            catalog_digest=diff.desired_catalog_digest,
            build_config_digest=diff.desired_build_config_digest,
            retrieval_config_digest=diff.desired_retrieval_config_digest,
            retrieval_config=alias_cfg.retrieve,
            status="pending",
        )
        self._deployment_repo.create_pending(deployment)

        try:
            self._validate_release_attestation(release)
            self._qdrant_alias_updater(kb_id, alias, release.collection_name)
        except Exception as exc:
            self._deployment_repo.mark_failed(deployment.id, error=str(exc))
            raise AliasApplyError(
                f"alias apply failed while updating the Qdrant mirror: {exc}"
            ) from exc

        self._deployment_repo.activate(deployment.id, applied_at=self._clock())
        activated = self._deployment_repo.get(deployment.id)
        return AliasApplyResult(deployment=activated, release=release, action=action)

    def _validate_release_attestation(self, release: RagRelease) -> None:
        manager = self._collection_manager_factory(release.collection_name)
        try:
            if not manager.collection_exists():
                raise AliasApplyError(f"collection '{release.collection_name}' does not exist")
            attestation = manager.read_release_attestation()
            if attestation is None:
                raise AliasApplyError(
                    f"collection '{release.collection_name}' has no release attestation"
                )
            comparison = compare_release_attestation(release, attestation)
            if not comparison.matches:
                raise AliasApplyError(f"release attestation mismatch: {comparison.mismatches}")
        finally:
            close = getattr(manager, "close", None)
            if close is not None:
                close()
