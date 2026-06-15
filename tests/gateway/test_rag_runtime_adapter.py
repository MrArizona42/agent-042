from __future__ import annotations

from gateway.services.rag_service import RAGService
from rag.domain import RetrievalHit
from rag.runtime import RagRuntimeResult
from shared.catalog import AliasConfig, KBConfig, TaskConfig, catalog_override
from shared.config import Settings, load_settings


class _Embedding:
    dimension = 3

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]


class _Runtime:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def invalidate_caches(self) -> None:
        pass

    def validate_aliases(self, *, strict: bool = False) -> None:
        pass

    def retrieve(self, *, query, sources):
        self.calls.append({"query": query, "sources": sources})
        return RagRuntimeResult(
            hits=[
                RetrievalHit(
                    chunk_id="torch.nn:chunk:0001",
                    document_id="torch.nn",
                    text="torch.nn.Module is the base class.",
                    score=0.9,
                    source_type="html_docs",
                    title="torch.nn.Module",
                    uri="https://pytorch.org/docs/stable/generated/torch.nn.Module.html",
                    section_title="Module",
                    metadata={
                        "collection_name": "rag__pytorch_reference__20260605_120000",
                        "manifest_id": "sha256:test",
                    },
                )
            ]
        )


def _settings() -> Settings:
    return load_settings(
        overrides={
            "vllm": {"model": "base-model"},
            "platform": {
                "qdrant_host": "localhost",
                "qdrant_port": 6333,
                "embeddings_url": "http://embeddings:8100",
                "vllm_base_url": "http://localhost:8000",
            },
            "gateway": {"embeddings_timeout": 10.0},
            "rag": {
                "enabled": True,
                "embedding_model": "test-embedding",
                "embedding_device": "cpu",
                "build": {"embedding_batch_size": 32, "qdrant_upsert_batch_size": 128},
                "kb_selection_threshold": 0.3,
                "strict_startup": False,
            },
        }
    )


def _catalog() -> dict[str, TaskConfig]:
    alias = AliasConfig(
        top_k=5,
        score_threshold=0.1,
        retrieval_strategy="dense",
        reranker=None,
        reranker_multiplier=1,
    )
    kb = KBConfig(
        name="pytorch_reference",
        default_alias="champion",
        aliases={"champion": alias, "challenger": alias},
        label="PyTorch",
        description="PyTorch docs",
        selection_description="PyTorch docs",
    )
    return {
        "code": TaskConfig(
            task="code",
            routing_description="Coding help",
            knowledge_bases=[kb],
        )
    }


def test_retrieve_documents_delegates_explicit_alias_to_runtime(monkeypatch) -> None:
    with catalog_override(_catalog()):
        monkeypatch.setattr(
            "gateway.services.rag_service.EmbeddingService",
            lambda **_: _Embedding(),
        )
        service = RAGService(settings=_settings())
        runtime = _Runtime()
        service.runtime = runtime

        docs = service.retrieve_documents(
            query="How do I define a module?",
            knowledge_base="pytorch_reference",
            alias="challenger",
        )

    assert len(docs) == 1
    assert docs[0].metadata["chunk_id"] == "torch.nn:chunk:0001"
    assert docs[0].metadata["collection_name"] == "rag__pytorch_reference__20260605_120000"
    assert runtime.calls[0]["sources"][0].knowledge_base == "pytorch_reference"
    assert runtime.calls[0]["sources"][0].alias == "challenger"


def test_retrieve_documents_uses_default_alias_when_alias_is_omitted(monkeypatch) -> None:
    with catalog_override(_catalog()):
        monkeypatch.setattr(
            "gateway.services.rag_service.EmbeddingService",
            lambda **_: _Embedding(),
        )
        service = RAGService(settings=_settings())
        runtime = _Runtime()
        service.runtime = runtime

        service.retrieve_documents(
            query="How do I define a module?",
            knowledge_base="pytorch_reference",
        )

    assert runtime.calls[0]["sources"][0].alias == "champion"
