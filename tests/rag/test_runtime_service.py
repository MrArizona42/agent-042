from __future__ import annotations

from datetime import UTC, datetime

from rag.domain import CollectionAttestation, RetrievalCapability, attestation_payload
from rag.runtime import RagRuntime, RagRuntimeSource
from rag.vector_store import Document
from shared.catalog import AliasConfig, KBConfig, TaskConfig, catalog_override
from shared.config import load_settings


class _Embedding:
    dimension = 3

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class _Sparse:
    def encode_query(self, query: str):
        from qdrant_client.models import SparseVector

        return SparseVector(indices=[1], values=[1.0])


class _Store:
    def __init__(self, *, name: str, stores: dict[str, "_Store"]):
        self.collection_name = name
        self._stores = stores
        self.alias_target: str | None = None
        self.meta: dict | None = None
        self.documents: list[Document] = []
        self.search_calls: list[dict] = []

    def collection_exists(self) -> bool:
        return self.collection_name in self._stores

    def resolve_alias(self, alias_name: str) -> str | None:
        return self.alias_target

    def read_meta(self) -> dict | None:
        return self.meta

    def get_collection_info(self) -> dict:
        return {"vector_size": 3}

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.documents[: kwargs["top_k"]]


def _alias(
    *,
    strategy: str,
    reranker: str | None = None,
) -> AliasConfig:
    return AliasConfig(
        top_k=5,
        score_threshold=0.1,
        reranker=reranker,
        retrieval_strategy=strategy,  # type: ignore[arg-type]
        reranker_multiplier=1,
    )


def _catalog() -> dict[str, TaskConfig]:
    kb = KBConfig(
        name="pytorch_reference",
        default_alias="champion",
        aliases={
            "champion": _alias(strategy="dense"),
            "challenger": _alias(strategy="hybrid"),
        },
        label="PyTorch",
        description="PyTorch docs",
        selection_description="PyTorch docs",
    )
    return {
        "code": TaskConfig(
            task="code",
            routing_description="Code help",
            knowledge_bases=[kb],
        )
    }


def _attestation(*, collection_name: str, capability: str) -> dict:
    return attestation_payload(
        CollectionAttestation(
            manifest_id=f"sha256:{collection_name}",
            kb_id="pytorch_reference",
            collection_name=collection_name,
            embedding_model="test-embedding",
            sparse_encoder="Qdrant/bm25" if capability == "hybrid" else None,
            retrieval_capability=RetrievalCapability(capability),
            chunk_count=1,
            created_at=datetime(2026, 6, 5, tzinfo=UTC),
        )
    )


def _stores(*, capability: str = "dense") -> dict[str, _Store]:
    stores: dict[str, _Store] = {}
    collection_name = "rag__pytorch_reference__20260605_120000"
    alias_name = "rag__pytorch_reference__champion"
    collection = _Store(name=collection_name, stores=stores)
    collection.meta = _attestation(collection_name=collection_name, capability=capability)
    collection.documents = [
        Document(
            content="torch.nn.Module is the base class.",
            score=0.9,
            metadata={
                "chunk_id": "torch.nn:chunk:0001",
                "document_id": "torch.nn",
                "source_document_id": "torch.nn",
                "source_type": "html_docs",
                "source_uri": "https://pytorch.org/docs/stable/generated/torch.nn.Module.html",
                "title": "torch.nn.Module",
                "section_title": "Module",
            },
        )
    ]
    alias = _Store(name=alias_name, stores=stores)
    alias.alias_target = collection_name
    stores[collection_name] = collection
    stores[alias_name] = alias
    return stores


def _runtime(stores: dict[str, _Store]) -> RagRuntime:
    return RagRuntime(
        settings=load_settings(
            overrides={
                "platform": {
                    "qdrant_host": "localhost",
                    "qdrant_port": 6333,
                    "embeddings_url": "http://embeddings:8100",
                },
                "gateway": {"embeddings_timeout": 10.0},
                "rag": {
                    "embedding_model": "test-embedding",
                    "embedding_device": "cpu",
                    "build": {"embedding_batch_size": 32, "qdrant_upsert_batch_size": 128},
                },
            }
        ),
        embedding_service=_Embedding(),
        vector_store_factory=lambda name: stores.get(name) or _Store(name=name, stores=stores),
        sparse_encoder_factory=lambda: _Sparse(),
    )


def test_runtime_uses_default_alias_when_source_alias_is_none() -> None:
    stores = _stores(capability="dense")
    with catalog_override(_catalog()):
        result = _runtime(stores).retrieve(
            query="How do I define a module?",
            sources=[RagRuntimeSource(knowledge_base="pytorch_reference")],
        )

    assert result.skipped_sources == []
    assert len(result.hits) == 1
    assert result.hits[0].metadata["qdrant_alias"] == "rag__pytorch_reference__champion"
    assert result.hits[0].metadata["collection_name"] == "rag__pytorch_reference__20260605_120000"
    assert result.provenance[0]["qdrant_alias"] == "rag__pytorch_reference__champion"
    assert result.provenance[0]["collection_name"] == "rag__pytorch_reference__20260605_120000"
    assert result.provenance[0]["manifest_id"] == "sha256:rag__pytorch_reference__20260605_120000"
    assert result.provenance[0]["hit_count"] == 1
    assert result.provenance[0]["no_hit"] is False
    assert result.provenance[0]["score_max"] == 0.9
    assert result.provenance[0]["top_scores"] == [0.9]
    assert result.provenance[0]["timings_ms"]["retrieve"] >= 0.0
    assert result.timings_ms["total"] >= 0.0
    assert result.diagnostics == {
        "requested_source_count": 1,
        "resolved_source_count": 1,
        "skipped_source_count": 0,
        "hit_count": 1,
        "no_hit": False,
    }


def test_runtime_allows_dense_alias_on_hybrid_collection() -> None:
    stores = _stores(capability="hybrid")
    with catalog_override(_catalog()):
        result = _runtime(stores).retrieve(
            query="How do I define a module?",
            sources=[RagRuntimeSource(knowledge_base="pytorch_reference", alias="champion")],
        )

    assert result.skipped_sources == []
    assert len(result.hits) == 1
    collection = stores["rag__pytorch_reference__20260605_120000"]
    assert collection.search_calls[0]["top_k"] == 5
    assert "strategy" not in collection.search_calls[0]


def test_runtime_rejects_hybrid_alias_on_dense_collection() -> None:
    stores = _stores(capability="dense")
    challenger_alias = _Store(name="rag__pytorch_reference__challenger", stores=stores)
    challenger_alias.alias_target = "rag__pytorch_reference__20260605_120000"
    stores[challenger_alias.collection_name] = challenger_alias

    with catalog_override(_catalog()):
        result = _runtime(stores).retrieve(
            query="How do I define a module?",
            sources=[RagRuntimeSource(knowledge_base="pytorch_reference", alias="challenger")],
        )

    assert result.hits == []
    assert result.skipped_sources[0].knowledge_base == "pytorch_reference"
    assert result.skipped_sources[0].alias == "challenger"
    assert "hybrid" in result.skipped_sources[0].reason
    assert result.diagnostics["requested_source_count"] == 1
    assert result.diagnostics["resolved_source_count"] == 0
    assert result.diagnostics["skipped_source_count"] == 1
    assert result.diagnostics["no_hit"] is True


def test_runtime_uses_explicit_hybrid_alias() -> None:
    stores = _stores(capability="hybrid")
    challenger_alias = _Store(name="rag__pytorch_reference__challenger", stores=stores)
    challenger_alias.alias_target = "rag__pytorch_reference__20260605_120000"
    stores[challenger_alias.collection_name] = challenger_alias

    with catalog_override(_catalog()):
        result = _runtime(stores).retrieve(
            query="How do I define a module?",
            sources=[RagRuntimeSource(knowledge_base="pytorch_reference", alias="challenger")],
        )

    assert result.skipped_sources == []
    assert len(result.hits) == 1
    collection = stores["rag__pytorch_reference__20260605_120000"]
    assert collection.search_calls[0]["strategy"] == "hybrid"


def test_runtime_marks_resolved_source_with_no_hits() -> None:
    stores = _stores(capability="dense")
    stores["rag__pytorch_reference__20260605_120000"].documents = []

    with catalog_override(_catalog()):
        result = _runtime(stores).retrieve(
            query="How do I define a module?",
            sources=[RagRuntimeSource(knowledge_base="pytorch_reference")],
        )

    assert result.hits == []
    assert result.skipped_sources == []
    assert result.provenance[0]["hit_count"] == 0
    assert result.provenance[0]["no_hit"] is True
    assert result.provenance[0]["score_min"] is None
    assert result.provenance[0]["top_scores"] == []
    assert result.diagnostics["resolved_source_count"] == 1
    assert result.diagnostics["hit_count"] == 0
    assert result.diagnostics["no_hit"] is True


def test_runtime_reports_empty_query_diagnostics() -> None:
    stores = _stores(capability="dense")

    with catalog_override(_catalog()):
        result = _runtime(stores).retrieve(
            query=" ",
            sources=[RagRuntimeSource(knowledge_base="pytorch_reference")],
        )

    assert result.hits == []
    assert result.skipped_sources == []
    assert result.provenance == []
    assert result.timings_ms["total"] >= 0.0
    assert result.diagnostics == {
        "requested_source_count": 1,
        "resolved_source_count": 0,
        "skipped_source_count": 0,
        "hit_count": 0,
        "no_hit": True,
    }
