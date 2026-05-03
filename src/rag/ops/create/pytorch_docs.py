"""Create a fresh PyTorch docs collection from explicit build configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.chunking import get_chunker
from rag.embeddings import EmbeddingService
from rag.ops.aliases import assign_alias_to_collection
from rag.ops.materialize import (
    batch_embed_and_upsert,
    create_collection_with_meta,
    make_collection_name,
)
from rag.ops.meta import BuildConfig, ImplementationInfo, build_collection_meta
from rag.sparse_encoder import SparseEncoderService
from shared.config import get_settings, validate_kb_alias


def create_pytorch_docs_collection(
    *,
    build_config: BuildConfig,
    pytorch_docs_file: str = "assets/rag_data/pytorch_docs/pytorch_docs.json",
    kb: str = "pytorch_docs",
    alias: str | None = None,
    collection_name: str | None = None,
    implementation: ImplementationInfo | None = None,
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
    embeddings_url: str | None = None,
) -> dict[str, Any]:
    """Create a fresh PyTorch docs collection and optionally attach an alias."""
    validate_kb_alias(kb, alias)
    settings = get_settings()
    qdrant_host = qdrant_host or settings.qdrant_host
    qdrant_port = qdrant_port or settings.qdrant_port

    docs_path = Path(pytorch_docs_file)
    if not docs_path.exists():
        raise FileNotFoundError(f"PyTorch docs file not found: {docs_path}")

    with open(docs_path, encoding="utf-8") as file_handle:
        docs = json.load(file_handle)

    embedding_service: EmbeddingService | None = None
    dimension = 0
    if build_config.retrieval_capability != "sparse":
        embedding_service = EmbeddingService(
            model_name=build_config.embedding_model,
            embeddings_url=embeddings_url,
        )
        dimension = embedding_service.dimension
    chunker = get_chunker(
        strategy=build_config.chunking_strategy,
        chunk_size=build_config.chunk_size,
        chunk_overlap=build_config.chunk_overlap,
    )

    collection_name = collection_name or make_collection_name(kb)
    implementation = implementation or ImplementationInfo(module=__name__, experimental=False)
    meta = build_collection_meta(
        kb_name=kb,
        build_config=build_config,
        implementation=implementation,
    )
    vector_store = create_collection_with_meta(
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        collection_name=collection_name,
        dimension=dimension,
        meta=meta,
    )

    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for doc in docs:
        chunks = chunker.chunk(doc["content"])
        for chunk in chunks:
            documents.append(chunk)
            metadatas.append(
                {
                    "task": "code",
                    "source": "pytorch_docs",
                    "url": doc["url"],
                    "title": doc["title"],
                    "scraped_at": doc["scraped_at"],
                }
            )

    sparse_encoder_service = (
        SparseEncoderService(embeddings_url=embeddings_url)
        if build_config.retrieval_capability in {"hybrid", "sparse"}
        else None
    )
    batch_embed_and_upsert(
        vector_store=vector_store,
        embedding_service=embedding_service,
        documents=documents,
        metadatas=metadatas,
        sparse_encoder_service=sparse_encoder_service,
    )

    alias_result = None
    if alias is not None:
        alias_result = assign_alias_to_collection(
            kb=kb,
            alias=alias,
            collection_name=collection_name,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
        )

    info = vector_store.get_collection_info()
    return {
        "collection_name": collection_name,
        "alias": alias_result,
        "meta": meta.to_payload(),
        "points_count": info.get("points_count", 0),
    }
