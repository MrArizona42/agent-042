"""Catalog-aware LlamaIndex runtime retrieval and query synthesis."""

from __future__ import annotations

import logging
from time import perf_counter

from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore
from qdrant_client import AsyncQdrantClient, QdrantClient

from app_config.catalog import AliasConfig, KBConfig, get_catalog, get_kb_config
from app_config.catalog.schema import AliasRetrievalConfig
from app_config.runtime import JudgeSettings, get_settings, secret_value
from rag.clients.qdrant import create_qdrant_clients
from rag.contracts import (
    DEFAULT_RAG_QUERY_PROMPTS,
    ProjectQueryPrompts,
)
from rag.control_plane.postgres import (
    PostgresAliasDeploymentRepository,
    PostgresReleaseRepository,
    create_session_factory,
)
from rag.control_plane.repositories import AliasDeploymentRepository, ReleaseRepository
from rag.embeddings import EmbeddingService
from rag.indexing.materialize import validate_strategy_supported
from rag.reranker import Reranker, get_reranker
from rag.runtime.engines import RuntimeRetriever, build_runtime_retriever
from rag.runtime.models import (
    RagQueryResult,
    RagRuntimeResult,
    RagRuntimeSource,
    RuntimeSkippedSource,
)
from rag.runtime.resolver import LlamaIndexRuntimeResolver, RuntimeAliasState
from rag.sparse_encoder import SparseEncoderService

logger = logging.getLogger(__name__)


class RagDatabaseUnavailableError(RuntimeError):
    """RAG is enabled but no agent042 Postgres database is configured.

    Postgres holds the applied alias deployment state that the runtime
    resolves against; without it there is no source of truth to serve from.
    """


def to_flat_alias_config(retrieval_config: AliasRetrievalConfig) -> AliasConfig:
    """Adapt the snapshotted deployment retrieval config to the runtime's flat shape.

    `rag.runtime.engines.build_runtime_retriever` and the postprocessor stack
    take `app_config.catalog.AliasConfig` (field name `retrieval_strategy`);
    `AliasDeployment.retrieval_config` is `AliasRetrievalConfig` (field name
    `strategy`). Converting here keeps that call site catalog-schema-agnostic
    instead of churning engines.py for a field rename.
    """
    return AliasConfig(
        top_k=retrieval_config.top_k,
        score_threshold=retrieval_config.score_threshold,
        reranker=retrieval_config.reranker,
        retrieval_strategy=retrieval_config.strategy,
        reranker_multiplier=retrieval_config.reranker_multiplier,
    )


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)


def _score_summary(nodes: list[NodeWithScore]) -> dict[str, float | list[float] | None]:
    scores = [float(node.score) for node in nodes if node.score is not None]
    if not scores:
        return {
            "score_min": None,
            "score_max": None,
            "score_avg": None,
            "top_scores": [],
        }
    return {
        "score_min": min(scores),
        "score_max": max(scores),
        "score_avg": round(sum(scores) / len(scores), 6),
        "top_scores": scores[:5],
    }


class RagRuntime:
    """Resolve catalog KB aliases into LlamaIndex retrieval and query operations."""

    def __init__(
        self,
        *,
        settings=None,
        embedding_service: EmbeddingService | None = None,
        qdrant_client: QdrantClient | None = None,
        qdrant_aclient: AsyncQdrantClient | None = None,
        resolver: LlamaIndexRuntimeResolver | None = None,
        reranker_factory=get_reranker,
        sparse_encoder_factory=None,
        deployment_repo: AliasDeploymentRepository | None = None,
        release_repo: ReleaseRepository | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.platform_settings = self.settings.platform
        self.rag_settings = self.settings.rag
        self.embedding_service = embedding_service or EmbeddingService()
        self._reranker_factory = reranker_factory
        self._sparse_encoder_factory = sparse_encoder_factory or self._default_sparse_encoder

        if deployment_repo is None or release_repo is None:
            db_url = self.settings.auth.agent042_db_url
            if not db_url:
                raise RagDatabaseUnavailableError(
                    "RAG is enabled but no agent042 database URL is configured "
                    "(settings.auth.agent042_db_url). The applied alias deployment "
                    "control plane requires Postgres; set GATEWAY_AGENT042_DB_URL or "
                    "the underlying postgres settings."
                )
            session_factory = create_session_factory(db_url)
            deployment_repo = deployment_repo or PostgresAliasDeploymentRepository(session_factory)
            release_repo = release_repo or PostgresReleaseRepository(session_factory)
        self._deployment_repo: AliasDeploymentRepository = deployment_repo
        self._release_repo: ReleaseRepository = release_repo

        if qdrant_client is None or qdrant_aclient is None:
            default_client, default_aclient = create_qdrant_clients(
                host=self.platform_settings.qdrant_host,
                port=self.platform_settings.qdrant_port,
            )
        client = qdrant_client or default_client
        aclient = qdrant_aclient or default_aclient
        self._resolver = resolver or LlamaIndexRuntimeResolver(
            qdrant_client=client,
            qdrant_aclient=aclient,
            embedding_service=self.embedding_service,
            qdrant_batch_size=self.rag_settings.build.qdrant_upsert_batch_size,
            sparse_encoder_factory=self._sparse_encoder_factory,
            release_repo=self._release_repo,
        )
        self._retrievers: dict[object, RuntimeRetriever] = {}
        self._alias_states: dict[tuple[str, str], RuntimeAliasState] = {}

    def invalidate_caches(self) -> None:
        self._retrievers.clear()
        self._alias_states.clear()

    def close(self) -> None:
        """Close the underlying Qdrant clients (sync context, e.g. CLI commands)."""
        self._resolver.close()

    async def aclose(self) -> None:
        """Close the underlying Qdrant clients (async context, e.g. gateway lifecycle)."""
        await self._resolver.aclose()

    @property
    def resolver(self) -> LlamaIndexRuntimeResolver:
        """Expose the configured resolver to collection-scoped runtime workflows."""
        return self._resolver

    def sparse_encoder(self) -> SparseEncoderService:
        """Create the runtime-configured sparse encoder."""
        return self._sparse_encoder_factory()

    def reranker(self, model: str | None) -> Reranker | None:
        """Create the runtime-configured reranker when requested."""
        return self._reranker_factory(model) if model else None

    def _default_sparse_encoder(self) -> SparseEncoderService:
        return SparseEncoderService(embeddings_url=self.platform_settings.embeddings_url)

    @staticmethod
    def _effective_alias(kb_cfg: KBConfig, alias: str | None) -> str:
        return alias or kb_cfg.default_alias

    def _resolve_alias_state(
        self,
        *,
        kb_id: str,
        alias: str,
        strict: bool,
    ) -> RuntimeAliasState | None:
        """Resolve the active applied deployment for (kb_id, alias).

        Looks up the active deployment fresh on every call -- cheap relative
        to the Qdrant attestation read it gates -- so a new deployment from
        `alias apply` becomes visible without a process restart. The more
        expensive release/attestation validation only reruns when the active
        deployment id actually changed since the last resolution.
        """
        cache_key = (kb_id, alias)
        deployment = self._deployment_repo.get_active(kb_id=kb_id, alias=alias)
        if deployment is None:
            self._alias_states.pop(cache_key, None)
            if strict:
                raise RuntimeError(f"No active alias deployment for kb='{kb_id}' alias='{alias}'")
            return None

        cached = self._alias_states.get(cache_key)
        if cached is not None and cached.deployment_id == deployment.id:
            return cached

        state = self._resolver.resolve(
            kb_id=kb_id, alias=alias, deployment=deployment, strict=strict
        )
        if state is not None:
            self._alias_states[cache_key] = state
        else:
            self._alias_states.pop(cache_key, None)
        return state

    def _get_retriever(self, *, state: RuntimeAliasState) -> RuntimeRetriever:
        cache_key = state.deployment_id or state.qdrant_alias
        cached = self._retrievers.get(cache_key)
        if cached is not None:
            return cached

        retrieval_config = state.retrieval_config
        assert retrieval_config is not None, "applied-state resolution always sets retrieval_config"
        flat_alias_cfg = to_flat_alias_config(retrieval_config)
        index = self._resolver.open_index(state, strategy=retrieval_config.strategy)
        reranker: Reranker | None = (
            self._reranker_factory(retrieval_config.reranker) if retrieval_config.reranker else None
        )
        runtime_retriever = build_runtime_retriever(
            index=index,
            alias_config=flat_alias_cfg,
            reranker_client=reranker,
        )
        self._retrievers[cache_key] = runtime_retriever
        return runtime_retriever

    def resolve_alias_profile(
        self,
        *,
        kb_id: str,
        alias: str,
    ) -> tuple[KBConfig, RuntimeAliasState, RuntimeRetriever]:
        """Resolve one explicit alias and its reusable retrieval profile."""
        kb_cfg = get_kb_config(kb_id)
        if kb_cfg is None:
            raise ValueError(f"Unknown knowledge base '{kb_id}'")
        if alias not in kb_cfg.aliases:
            raise ValueError(f"Unknown alias '{alias}' for KB '{kb_id}'")
        state = self._resolve_alias_state(kb_id=kb_id, alias=alias, strict=True)
        assert state is not None
        assert state.retrieval_config is not None
        validate_strategy_supported(
            retrieval_strategy=state.retrieval_config.strategy,
            retrieval_capability=state.retrieval_capability,
        )
        return kb_cfg, state, self._get_retriever(state=state)

    @staticmethod
    def _enrich_node(
        node: NodeWithScore,
        *,
        state: RuntimeAliasState,
    ) -> NodeWithScore:
        node.node.metadata.update(
            {
                "kb_id": state.kb_id,
                "alias": state.alias,
                "qdrant_alias": state.qdrant_alias,
                "collection_name": state.collection_name,
                "manifest_id": state.manifest_id,
                "retrieval_capability": state.retrieval_capability,
            }
        )
        return node

    def validate_aliases(self, *, strict: bool = False) -> None:
        """Validate every declared catalog alias resolves to a healthy applied deployment.

        Checks the *applied* deployment's retrieval config against the
        applied release's capability -- not the current desired catalog
        values, which may differ until the next `alias apply`.
        """
        for task_cfg in get_catalog().values():
            for kb_cfg in task_cfg.knowledge_bases:
                for alias in kb_cfg.aliases:
                    state = self._resolve_alias_state(
                        kb_id=kb_cfg.name,
                        alias=alias,
                        strict=strict,
                    )
                    if state is None or state.retrieval_config is None:
                        continue
                    try:
                        validate_strategy_supported(
                            retrieval_strategy=state.retrieval_config.strategy,
                            retrieval_capability=state.retrieval_capability,
                        )
                    except ValueError:
                        if strict:
                            raise
                        logger.warning(
                            "RAG alias is incompatible: kb=%s alias=%s collection=%s",
                            kb_cfg.name,
                            alias,
                            state.collection_name,
                            exc_info=True,
                        )

    def retrieve(
        self,
        *,
        query: str,
        sources: list[RagRuntimeSource],
    ) -> RagRuntimeResult:
        """Retrieve native nodes and compatibility hits from requested KB aliases."""
        started_at = perf_counter()
        result = RagRuntimeResult()
        if not query.strip():
            result.timings_ms["total"] = _elapsed_ms(started_at)
            result.diagnostics = self._diagnostics(result, requested=len(sources))
            return result

        for source in sources:
            source_started_at = perf_counter()
            kb_cfg = get_kb_config(source.knowledge_base)
            if kb_cfg is None:
                result.skipped_sources.append(
                    RuntimeSkippedSource(
                        knowledge_base=source.knowledge_base,
                        alias=source.alias,
                        reason="unknown_knowledge_base",
                    )
                )
                continue

            alias = self._effective_alias(kb_cfg, source.alias)
            if alias not in kb_cfg.aliases:
                result.skipped_sources.append(
                    RuntimeSkippedSource(
                        knowledge_base=source.knowledge_base,
                        alias=alias,
                        reason="unknown_alias",
                    )
                )
                continue

            resolve_started_at = perf_counter()
            state = self._resolve_alias_state(kb_id=kb_cfg.name, alias=alias, strict=False)
            resolve_ms = _elapsed_ms(resolve_started_at)
            if state is None:
                result.skipped_sources.append(
                    RuntimeSkippedSource(
                        knowledge_base=source.knowledge_base,
                        alias=alias,
                        reason="alias_unavailable",
                    )
                )
                continue
            assert state.retrieval_config is not None
            try:
                validate_strategy_supported(
                    retrieval_strategy=state.retrieval_config.strategy,
                    retrieval_capability=state.retrieval_capability,
                )
            except ValueError as exc:
                result.skipped_sources.append(
                    RuntimeSkippedSource(
                        knowledge_base=source.knowledge_base,
                        alias=alias,
                        reason=str(exc),
                    )
                )
                continue

            retrieve_started_at = perf_counter()
            nodes = self._get_retriever(state=state).retrieve(query)
            nodes = [self._enrich_node(node, state=state) for node in nodes]
            retrieve_ms = _elapsed_ms(retrieve_started_at)
            result.nodes.extend(nodes)
            result.provenance.append(
                {
                    "knowledge_base": kb_cfg.name,
                    "alias": alias,
                    "qdrant_alias": state.qdrant_alias,
                    "collection_name": state.collection_name,
                    "manifest_id": state.manifest_id,
                    "deployment_id": str(state.deployment_id) if state.deployment_id else None,
                    "release_id": state.release.id if state.release else None,
                    "retrieval_strategy": state.retrieval_config.strategy,
                    "retrieval_capability": state.retrieval_capability,
                    "hit_count": len(nodes),
                    "no_hit": not nodes,
                    **_score_summary(nodes),
                    "timings_ms": {
                        "resolve": resolve_ms,
                        "retrieve": retrieve_ms,
                        "total": _elapsed_ms(source_started_at),
                    },
                }
            )

        result.timings_ms["total"] = _elapsed_ms(started_at)
        result.diagnostics = self._diagnostics(result, requested=len(sources))
        return result

    @staticmethod
    def _diagnostics(result: RagRuntimeResult, *, requested: int) -> dict[str, object]:
        return {
            "requested_source_count": requested,
            "resolved_source_count": len(result.provenance),
            "skipped_source_count": len(result.skipped_sources),
            "hit_count": len(result.nodes),
            "no_hit": not result.nodes,
        }

    def generation_llm(self) -> LLM:
        """Return the runtime-configured OpenAI-compatible generation client.

        Uses ``OpenAILike`` rather than ``OpenAI`` because the generation model
        is served locally by vLLM under a non-OpenAI model name; LlamaIndex's
        plain ``OpenAI`` client hard-validates model names against an OpenAI
        allowlist and would raise on any access to ``.metadata``.
        """
        from llama_index.llms.openai_like import OpenAILike

        api_key = secret_value(getattr(self.settings.gateway, "api_key", None)) or "not-needed"
        return OpenAILike(
            model=self.settings.vllm.model,
            api_base=f"{str(self.platform_settings.vllm_base_url).rstrip('/')}/v1",
            api_key=api_key,
            context_window=self.settings.gateway.budget.model_max_tokens,
            is_chat_model=True,
            temperature=0.0,
            timeout=float(self.settings.gateway.vllm_timeout),
        )

    def judge_settings(self) -> JudgeSettings:
        """Resolve the LLM-as-judge transport config, independent of the generation model."""
        return self.settings.eval.resolve_judge_settings(
            self.platform_settings,
            local_context_window=self.settings.gateway.budget.model_max_tokens,
        )

    def judge_llm(self) -> LLM:
        """Return the LLM-as-judge client resolved from ``settings.eval.judge``.

        Deliberately separate from ``generation_llm()``: the judge backend can
        differ from the model serving RAG generation (e.g. an external
        OpenAI-compatible judge), and conflating the two previously meant
        benchmark runs silently scored answers using the generation model as
        if it were the configured judge.
        """
        from llama_index.llms.openai_like import OpenAILike

        judge = self.judge_settings()
        if judge.context_window is None:
            raise ValueError(
                f"eval.judge.context_window must be set for backend '{judge.backend}' "
                "before it can back a LlamaIndex judge LLM client"
            )
        return OpenAILike(
            model=judge.model,
            api_base=f"{judge.base_url.rstrip('/')}/v1",
            api_key=judge.api_key or "not-needed",
            context_window=judge.context_window,
            is_chat_model=True,
            temperature=0.0,
            timeout=judge.timeout,
        )

    def query(
        self,
        *,
        query: str,
        source: RagRuntimeSource,
        llm: LLM | None = None,
        prompts: ProjectQueryPrompts = DEFAULT_RAG_QUERY_PROMPTS,
    ) -> RagQueryResult:
        """Run one KB/alias query engine and return answer plus source nodes."""
        if not query.strip():
            raise ValueError("query must not be blank")
        kb_cfg = get_kb_config(source.knowledge_base)
        if kb_cfg is None:
            raise ValueError(f"Unknown knowledge base '{source.knowledge_base}'")
        alias = self._effective_alias(kb_cfg, source.alias)
        if alias not in kb_cfg.aliases:
            raise ValueError(f"Unknown alias '{alias}' for KB '{kb_cfg.name}'")
        state = self._resolve_alias_state(kb_id=kb_cfg.name, alias=alias, strict=True)
        assert state is not None
        assert state.retrieval_config is not None
        validate_strategy_supported(
            retrieval_strategy=state.retrieval_config.strategy,
            retrieval_capability=state.retrieval_capability,
        )
        runtime_retriever = self._get_retriever(state=state)
        response = runtime_retriever.query_engine(
            llm=llm or self.generation_llm(),
            prompts=prompts,
        ).query(query)
        source_nodes = [self._enrich_node(node, state=state) for node in response.source_nodes]
        return RagQueryResult(
            answer=str(response),
            source_nodes=source_nodes,
            prompt_identity=prompts.identity,
            provenance={
                "knowledge_base": kb_cfg.name,
                "alias": alias,
                "qdrant_alias": state.qdrant_alias,
                "collection_name": state.collection_name,
                "manifest_id": state.manifest_id,
                "deployment_id": str(state.deployment_id) if state.deployment_id else None,
                "release_id": state.release.id if state.release else None,
                "retrieval_strategy": state.retrieval_config.strategy,
                "retrieval_capability": state.retrieval_capability,
            },
        )
