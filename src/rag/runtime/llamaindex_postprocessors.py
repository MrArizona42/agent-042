"""LlamaIndex node postprocessor adapter over the project's reranker service."""

from __future__ import annotations

from typing import Any, Protocol

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from pydantic import Field

from rag.vector_store import Document


class RerankerProtocol(Protocol):
    """Reranker client contract used by :class:`ProjectRerankerPostprocessor`."""

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        """Rerank documents against a query, returning them sorted descending by score."""
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

        documents = [
            Document(content=node.get_content(metadata_mode=MetadataMode.EMBED), metadata={})
            for node in nodes
        ]
        node_by_document_id = {id(doc): node.node for doc, node in zip(documents, nodes)}

        reranked = self.reranker_client.rerank(
            query_bundle.query_str,
            documents,
            top_k=len(documents),
        )
        result = [
            NodeWithScore(node=node_by_document_id[id(doc)], score=doc.score) for doc in reranked
        ]
        return result[: self.top_n] if self.top_n is not None else result
