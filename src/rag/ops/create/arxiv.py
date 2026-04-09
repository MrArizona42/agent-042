"""Create a fresh ArXiv collection from explicit build configuration."""

from __future__ import annotations

import json
import uuid
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
from shared.config import get_settings, validate_kb_alias

_POINT_ID_NS = uuid.UUID("b8c9d0e1-f2a3-4b5c-6d7e-8f9a0b1c2d3e")


def create_arxiv_collection(
    *,
    build_config: BuildConfig,
    arxiv_file: str = "assets/rag_data/arxiv/arxiv_papers.json",
    kb: str = "arxiv",
    alias: str | None = None,
    collection_name: str | None = None,
    implementation: ImplementationInfo | None = None,
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
    embeddings_url: str | None = None,
) -> dict[str, Any]:
    """Create a fresh ArXiv collection and optionally attach an alias."""
    validate_kb_alias(kb, alias)
    settings = get_settings()
    qdrant_host = qdrant_host or settings.qdrant_host
    qdrant_port = qdrant_port or settings.qdrant_port

    arxiv_path = Path(arxiv_file)
    if not arxiv_path.exists():
        raise FileNotFoundError(f"ArXiv file not found: {arxiv_path}")

    with open(arxiv_path, encoding="utf-8") as file_handle:
        papers = json.load(file_handle)

    embedding_service = EmbeddingService(
        model_name=build_config.embedding_model,
        embeddings_url=embeddings_url,
    )
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
        dimension=embedding_service.dimension,
        meta=meta,
    )

    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []
    for paper in papers:
        full_text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
        chunks = chunker.chunk(full_text)
        for chunk_idx, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append(
                {
                    "task": "chat",
                    "source": "arxiv",
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "primary_category": paper["primary_category"],
                    "published": paper["published"],
                }
            )
            ids.append(str(uuid.uuid5(_POINT_ID_NS, f"arxiv:{paper['arxiv_id']}:{chunk_idx}")))

    batch_embed_and_upsert(
        vector_store=vector_store,
        embedding_service=embedding_service,
        documents=documents,
        metadatas=metadatas,
        ids=ids,
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
