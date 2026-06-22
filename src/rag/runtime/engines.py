"""LlamaIndex retriever and query-engine construction."""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms import LLM
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from app_config.catalog import AliasConfig
from rag.contracts import ProjectQueryPrompts
from rag.runtime.llamaindex_postprocessors import ProjectRerankerPostprocessor


def _query_mode(strategy: str) -> VectorStoreQueryMode:
    if strategy == "hybrid":
        return VectorStoreQueryMode.HYBRID
    if strategy == "sparse":
        return VectorStoreQueryMode.SPARSE
    return VectorStoreQueryMode.DEFAULT


@dataclass(slots=True)
class RuntimeRetriever:
    """Retriever plus the project alias profile's postprocessing policy."""

    retriever: BaseRetriever
    node_postprocessors: list[BaseNodePostprocessor]

    def retrieve(self, query: str) -> list[NodeWithScore]:
        query_bundle = QueryBundle(query)
        nodes = self.retriever.retrieve(query_bundle)
        for postprocessor in self.node_postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes, query_bundle=query_bundle)
        return nodes

    def query_engine(
        self,
        *,
        llm: LLM,
        prompts: ProjectQueryPrompts,
    ) -> RetrieverQueryEngine:
        return RetrieverQueryEngine.from_args(
            self.retriever,
            llm=llm,
            node_postprocessors=self.node_postprocessors,
            text_qa_template=prompts.text_qa_template,
            refine_template=prompts.refine_template,
        )


def build_runtime_retriever(
    *,
    index: VectorStoreIndex,
    alias_config: AliasConfig,
    reranker_client=None,
) -> RuntimeRetriever:
    candidate_count = alias_config.top_k
    postprocessors: list[BaseNodePostprocessor] = []
    if reranker_client is not None:
        candidate_count *= alias_config.reranker_multiplier
        postprocessors.append(
            ProjectRerankerPostprocessor(
                reranker_client=reranker_client,
                top_n=alias_config.top_k,
            )
        )
    postprocessors.append(SimilarityPostprocessor(similarity_cutoff=alias_config.score_threshold))
    retriever = index.as_retriever(
        similarity_top_k=candidate_count,
        vector_store_query_mode=_query_mode(alias_config.retrieval_strategy),
    )
    return RuntimeRetriever(
        retriever=retriever,
        node_postprocessors=postprocessors,
    )
