"""Unit tests for the LlamaIndex reranker node postprocessor adapter (Phase 1).

Uses a fake reranker client so no network call ever happens.
"""

from __future__ import annotations

import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from rag.runtime.llamaindex_postprocessors import ProjectRerankerPostprocessor


class _FakeReranker:
    """Reverses node order and assigns descending scores, mirroring the real client's contract."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, nodes: list[NodeWithScore], top_k: int) -> list[NodeWithScore]:
        self.calls.append((query, [node.node.get_content() for node in nodes]))
        reversed_nodes = list(reversed(nodes))
        for score, node in enumerate(reversed_nodes):
            node.score = float(score)
        return reversed_nodes


def _node(node_id: str, text: str) -> NodeWithScore:
    return NodeWithScore(node=TextNode(id_=node_id, text=text), score=0.0)


class TestProjectRerankerPostprocessor:
    def test_class_name(self) -> None:
        assert ProjectRerankerPostprocessor.class_name() == "ProjectRerankerPostprocessor"

    def test_postprocess_nodes_reorders_by_reranker_score(self) -> None:
        reranker = _FakeReranker()
        postprocessor = ProjectRerankerPostprocessor(reranker_client=reranker)
        nodes = [_node("a", "alpha"), _node("b", "beta"), _node("c", "gamma")]

        result = postprocessor.postprocess_nodes(nodes, query_bundle=QueryBundle(query_str="q"))

        assert [n.node.id_ for n in result] == ["c", "b", "a"]
        assert [n.score for n in result] == [0.0, 1.0, 2.0]
        assert reranker.calls == [("q", ["alpha", "beta", "gamma"])]

    def test_top_n_truncates_results(self) -> None:
        reranker = _FakeReranker()
        postprocessor = ProjectRerankerPostprocessor(reranker_client=reranker, top_n=1)
        nodes = [_node("a", "alpha"), _node("b", "beta")]

        result = postprocessor.postprocess_nodes(nodes, query_bundle=QueryBundle(query_str="q"))

        assert [n.node.id_ for n in result] == ["b"]

    def test_empty_nodes_short_circuits(self) -> None:
        reranker = _FakeReranker()
        postprocessor = ProjectRerankerPostprocessor(reranker_client=reranker)

        result = postprocessor.postprocess_nodes([], query_bundle=QueryBundle(query_str="q"))

        assert result == []
        assert reranker.calls == []

    def test_missing_query_bundle_raises(self) -> None:
        reranker = _FakeReranker()
        postprocessor = ProjectRerankerPostprocessor(reranker_client=reranker)

        with pytest.raises(ValueError, match="requires a query_bundle"):
            postprocessor.postprocess_nodes([_node("a", "alpha")], query_bundle=None)
