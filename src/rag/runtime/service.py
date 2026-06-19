"""Catalog-aware LlamaIndex runtime retrieval and query synthesis."""

from __future__ import annotations

import logging
from time import perf_counter

from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore
from qdrant_client import AsyncQdrantClient, QdrantClient

from app_config.catalog import KBConfig, get_catalog, get_kb_config
from app_config.runtime import JudgeSettings, get_settings, secret_value
from rag.contracts import (
    DEFAULT_RAG_QUERY_PROMPTS,
    ProjectQueryPrompts,
)
from rag.embeddings import EmbeddingService
from rag.indexing.materialize import qdrant_alias_name, validate_strategy_supported
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
    ) -> None:
        self.settings = settings or get_settings()
        self.platform_settings = self.settings.platform
        self.rag_settings = self.settings.rag
        self.embedding_service = embedding_service or EmbeddingService(
            model_name=self.rag_settings.embedding_model,
            device=self.rag_settings.embedding_device,
            batch_size=self.rag_settings.build.embedding_batch_size,
        )
        self._reranker_factory = reranker_factory
        self._sparse_encoder_factory = sparse_encoder_factory or self._default_sparse_encoder
        client = qdrant_client or QdrantClient(
            host=self.platform_settings.qdrant_host,
            port=self.platform_settings.qdrant_port,
        )
        aclient = qdrant_aclient or AsyncQdrantClient(
            host=self.platform_settings.qdrant_host,
            port=self.platform_settings.qdrant_port,
        )
        self._resolver = resolver or LlamaIndexRuntimeResolver(
            qdrant_client=client,
            qdrant_aclient=aclient,
            embedding_service=self.embedding_service,
            embedding_model=self.rag_settings.embedding_model,
            qdrant_batch_size=self.rag_settings.build.qdrant_upsert_batch_size,
            sparse_encoder_factory=self._sparse_encoder_factory,
        )
        self._retrievers: dict[str, RuntimeRetriever] = {}
        self._alias_states: dict[str, RuntimeAliasState] = {}

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
    def _cache_key(kb_id: str, alias: str) -> str:
        return qdrant_alias_name(kb_id=kb_id, alias=alias)

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
        cache_key = self._cache_key(kb_id, alias)
        cached = self._alias_states.get(cache_key)
        if cached is not None:
            if self._resolver.alias_target(cache_key) == cached.collection_name:
                return cached
            self._retrievers.pop(cache_key, None)
            self._alias_states.pop(cache_key, None)

        state = self._resolver.resolve(
            kb_id=kb_id,
            alias=alias,
            qdrant_alias=cache_key,
            strict=strict,
        )
        if state is not None:
            self._alias_states[cache_key] = state
        return state

    def _get_retriever(
        self,
        *,
        kb_cfg: KBConfig,
        alias: str,
        state: RuntimeAliasState,
    ) -> RuntimeRetriever:
        cached = self._retrievers.get(state.qdrant_alias)
        if cached is not None:
            return cached

        alias_cfg = kb_cfg.aliases[alias]
        index = self._resolver.open_index(
            state,
            strategy=alias_cfg.retrieval_strategy,
        )
        reranker: Reranker | None = (
            self._reranker_factory(alias_cfg.reranker) if alias_cfg.reranker else None
        )
        runtime_retriever = build_runtime_retriever(
            index=index,
            alias_config=alias_cfg,
            reranker_client=reranker,
        )
        self._retrievers[state.qdrant_alias] = runtime_retriever
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
        validate_strategy_supported(
            retrieval_strategy=kb_cfg.aliases[alias].retrieval_strategy,
            retrieval_capability=state.attestation.retrieval_capability.value,
        )
        return kb_cfg, state, self._get_retriever(kb_cfg=kb_cfg, alias=alias, state=state)

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
                "manifest_id": state.attestation.manifest_id,
                "retrieval_capability": state.attestation.retrieval_capability.value,
            }
        )
        return node

    def validate_aliases(self, *, strict: bool = False) -> None:
        """Validate every declared catalog alias that is currently available."""
        for task_cfg in get_catalog().values():
            for kb_cfg in task_cfg.knowledge_bases:
                for alias, alias_cfg in kb_cfg.aliases.items():
                    state = self._resolve_alias_state(
                        kb_id=kb_cfg.name,
                        alias=alias,
                        strict=strict,
                    )
                    if state is None:
                        continue
                    try:
                        validate_strategy_supported(
                            retrieval_strategy=alias_cfg.retrieval_strategy,
                            retrieval_capability=state.attestation.retrieval_capability.value,
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
            alias_cfg = kb_cfg.aliases.get(alias)
            if alias_cfg is None:
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
            try:
                validate_strategy_supported(
                    retrieval_strategy=alias_cfg.retrieval_strategy,
                    retrieval_capability=state.attestation.retrieval_capability.value,
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
            nodes = self._get_retriever(kb_cfg=kb_cfg, alias=alias, state=state).retrieve(query)
            nodes = [self._enrich_node(node, state=state) for node in nodes]
            retrieve_ms = _elapsed_ms(retrieve_started_at)
            result.nodes.extend(nodes)
            result.provenance.append(
                {
                    "knowledge_base": kb_cfg.name,
                    "alias": alias,
                    "qdrant_alias": state.qdrant_alias,
                    "collection_name": state.collection_name,
                    "manifest_id": state.attestation.manifest_id,
                    "retrieval_strategy": alias_cfg.retrieval_strategy,
                    "retrieval_capability": state.attestation.retrieval_capability.value,
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
        alias_cfg = kb_cfg.aliases.get(alias)
        if alias_cfg is None:
            raise ValueError(f"Unknown alias '{alias}' for KB '{kb_cfg.name}'")
        state = self._resolve_alias_state(kb_id=kb_cfg.name, alias=alias, strict=True)
        assert state is not None
        validate_strategy_supported(
            retrieval_strategy=alias_cfg.retrieval_strategy,
            retrieval_capability=state.attestation.retrieval_capability.value,
        )
        runtime_retriever = self._get_retriever(kb_cfg=kb_cfg, alias=alias, state=state)
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
                "manifest_id": state.attestation.manifest_id,
                "retrieval_strategy": alias_cfg.retrieval_strategy,
                "retrieval_capability": state.attestation.retrieval_capability.value,
            },
        )
