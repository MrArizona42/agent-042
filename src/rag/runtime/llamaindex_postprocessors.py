"""LlamaIndex node postprocessor adapter over the project's reranker service."""

from __future__ import annotations

from typing import Any, Protocol

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import Field


class RerankerProtocol(Protocol):
    """Reranker client contract used by :class:`ProjectRerankerPostprocessor`."""

    def rerank(
        self, query: str, nodes: list[NodeWithScore], top_k: int
    ) -> list[NodeWithScore]:
        """Rerank nodes against a query, returning them sorted descending by score."""
        ...


class ProjectRerankerPostprocessor(BaseNodePostprocessor):
    """Reranks retrieved nodes through the project's cross-encoder reranker service."""

    reranker_client: Any = Field(exclude=True)
    top_n: int | None = None

    def __init__(
        self,
        *,
        reranker_client: RerankerProtocol,
        top_n: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(reranker_client=reranker_client, top_n=top_n, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "ProjectRerankerPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes:
            return []
        if query_bundle is None:
            raise ValueError("ProjectRerankerPostprocessor requires a query_bundle")

        reranked = self.reranker_client.rerank(
            query_bundle.query_str,
            nodes,
            top_k=len(nodes),
        )
        return reranked[: self.top_n] if self.top_n is not None else reranked
