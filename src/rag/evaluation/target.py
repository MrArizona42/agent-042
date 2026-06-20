"""Temporary LlamaIndex collection lifecycle for benchmark corpora."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from llama_index.core.llms import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, NodeWithScore, TextNode

from app_config.catalog import AliasConfig, KBConfig
from rag.contracts import DEFAULT_RAG_QUERY_PROMPTS, ProjectQueryPrompts
from rag.contracts.manifests import manifest_path
from rag.evaluation.models import BenchmarkPreparedArtifacts
from rag.indexing.llamaindex_qdrant import QdrantCollectionManager
from rag.indexing.materialize import (
    materialize_kb_collection_llamaindex,
    retrieval_capability_for_strategy,
)
from rag.runtime.engines import RuntimeRetriever, build_runtime_retriever
from rag.runtime.resolver import RuntimeAliasState
from rag.runtime.service import to_flat_alias_config
from rag.sources.bundles import SourceNodeBundle


@dataclass(slots=True)
class BenchmarkTarget:
    """Materialized benchmark target and the KB profile it mirrors."""

    source_instance_id: str
    kb: KBConfig
    alias: str
    alias_config: AliasConfig
    state: RuntimeAliasState
    runtime_retriever: RuntimeRetriever
    parameter_state: RuntimeAliasState
    build_profile: dict[str, object]
    collection_manager: QdrantCollectionManager | None = None
    manifest_artifact: Path | None = None

    @property
    def retriever(self):
        return self.runtime_retriever.retriever

    @property
    def node_postprocessors(self):
        return self.runtime_retriever.node_postprocessors

    def retrieve(self, query: str) -> list[NodeWithScore]:
        return self.runtime_retriever.retrieve(query)

    def query(
        self,
        query: str,
        *,
        llm: LLM,
        prompts: ProjectQueryPrompts = DEFAULT_RAG_QUERY_PROMPTS,
    ):
        return self.runtime_retriever.query_engine(llm=llm, prompts=prompts).query(query)

    def close(self) -> None:
        if self.collection_manager and self.collection_manager.collection_exists():
            self.collection_manager.client.delete_collection(self.state.collection_name)
        if self.manifest_artifact and self.manifest_artifact.is_file():
            self.manifest_artifact.unlink()


def _benchmark_nodes(documents: list[Document], *, chunk_size: int, chunk_overlap: int):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)
    for node in nodes:
        document_id = str(node.metadata.get("document_id") or node.ref_doc_id or node.node_id)
        node.metadata.update(
            {
                "document_id": document_id,
                "chunk_id": node.node_id,
            }
        )
    return [TextNode.model_validate(node.model_dump()) for node in nodes]


def materialize_benchmark_target(
    *,
    runtime,
    source_instance_id: str,
    kb: KBConfig,
    alias: str,
    artifacts: BenchmarkPreparedArtifacts,
    rag_data_root: Path | str,
) -> BenchmarkTarget:
    """Mirror an attached KB alias profile into a disposable benchmark collection.

    Mirrors the *applied* deployment's build and retrieval state, not the
    current desired catalog values, so a benchmark run reflects exactly what
    is being served rather than an unapplied catalog edit.
    """
    _, parameter_state, parameter_retriever = runtime.resolve_alias_profile(
        kb_id=kb.name,
        alias=alias,
    )
    if parameter_state.release is None or parameter_state.retrieval_config is None:
        raise RuntimeError(
            f"Active deployment for kb='{kb.name}' alias='{alias}' resolved without a "
            "release; cannot mirror its build profile for benchmark preparation"
        )
    alias_config = to_flat_alias_config(parameter_state.retrieval_config)
    build_config = parameter_state.release.build_config
    chunking = build_config.chunking
    build_profile: dict[str, object] = {
        "strategy": chunking.strategy,
        "chunk_size": chunking.chunk_size,
        "chunk_overlap": chunking.chunk_overlap,
    }

    if not artifacts.documents:
        return BenchmarkTarget(
            source_instance_id=source_instance_id,
            kb=kb,
            alias=alias,
            alias_config=alias_config,
            state=parameter_state,
            runtime_retriever=parameter_retriever,
            parameter_state=parameter_state,
            build_profile=build_profile,
        )

    chunk_size = int(build_profile["chunk_size"])
    chunk_overlap = int(build_profile["chunk_overlap"])
    nodes = _benchmark_nodes(
        artifacts.documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not nodes:
        raise ValueError("benchmark corpus produced no nodes")

    stamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S_%f")
    collection_name = f"eval__{kb.name}__{source_instance_id.replace('.', '_')}__{stamp}"
    manager = QdrantCollectionManager(
        client=runtime.resolver.qdrant_client,
        aclient=runtime.resolver.qdrant_aclient,
        collection_name=collection_name,
    )
    corpus_payload = json.dumps(
        [document.to_dict() for document in artifacts.documents],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    corpus_digest = f"sha256:{hashlib.sha256(corpus_payload).hexdigest()}"
    bundle = SourceNodeBundle(
        kb_id=kb.name,
        source_instance_id=source_instance_id,
        node_artifact_paths=[],
        node_artifact_checksums={"prepared_corpus": corpus_digest},
        nodes=nodes,
        document_count=len(artifacts.documents),
        node_count=len(nodes),
    )
    capability = retrieval_capability_for_strategy(alias_config.retrieval_strategy)
    try:
        result = materialize_kb_collection_llamaindex(
            kb_id=kb.name,
            collection_name=collection_name,
            bundles=[bundle],
            collection_manager=manager,
            embedding_client=runtime.embedding_service,
            embedding_model=build_config.dense_encoder.model,
            retrieval_capability=capability,
            rag_data_root=rag_data_root,
            sparse_encoder_model=(
                build_config.sparse_encoder.model
                if capability == "hybrid" and build_config.sparse_encoder is not None
                else None
            ),
            sparse_encoder_client=(runtime.sparse_encoder() if capability == "hybrid" else None),
            qdrant_upsert_batch_size=runtime.rag_settings.build.qdrant_upsert_batch_size,
            benchmark_scope=source_instance_id,
        )
    except Exception:
        if manager.collection_exists():
            manager.client.delete_collection(collection_name)
        temp_manifest = manifest_path(
            rag_data_root=rag_data_root,
            kb_id=kb.name,
            collection_name=collection_name,
        )
        if temp_manifest.is_file():
            temp_manifest.unlink()
        raise
    state = RuntimeAliasState(
        kb_id=kb.name,
        alias=alias,
        qdrant_alias=parameter_state.qdrant_alias,
        collection_name=collection_name,
        attestation=result.manifest.to_attestation(),
        vector_size=runtime.embedding_service.dimension,
    )
    index = runtime.resolver.open_index(state, strategy=alias_config.retrieval_strategy)
    reranker = runtime.reranker(alias_config.reranker)
    return BenchmarkTarget(
        source_instance_id=source_instance_id,
        kb=kb,
        alias=alias,
        alias_config=alias_config,
        state=state,
        runtime_retriever=build_runtime_retriever(
            index=index,
            alias_config=alias_config,
            reranker_client=reranker,
        ),
        parameter_state=parameter_state,
        build_profile=build_profile,
        collection_manager=manager,
        manifest_artifact=Path(result.manifest_path),
    )
