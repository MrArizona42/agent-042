"""Build ArXiv vector index (incremental strategy).

Loads ArXiv papers, chunks them, generates embeddings, and upserts
into Qdrant.  For each resolved alias of the KB, new papers are
appended to the existing collection.

Usage::

    python -m experiments.scripts.rag_data.build_arxiv_index \
        --arxiv_file assets/rag_data/arxiv/arxiv_papers.json \
        --kb arxiv
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add src to path to import rag module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from rag.chunking import get_chunker
from rag.embeddings import EmbeddingService
from rag.vector_store import QdrantVectorStore
from shared.config import get_knowledge_bases, get_settings

# Arbitrary but fixed namespace for UUID5-based point IDs.  Must remain
# constant across runs so the same (source, chunk_index) pair always
# produces the same UUID, enabling upsert-based deduplication.
_POINT_ID_NS = uuid.UUID("b8c9d0e1-f2a3-4b5c-6d7e-8f9a0b1c2d3e")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_arxiv(
    arxiv_file: str = "assets/rag_data/arxiv/arxiv_papers.json",
    kb: str = "arxiv",
    alias: str | None = None,
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
    embedding_model: str | None = None,
    embeddings_url: str | None = None,
    chunking_strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> None:
    """Build / update vector index for ArXiv papers (incremental mode).

    For each resolved alias of the KB, new papers are upserted into the
    existing collection.  If ``_meta`` exists the build config is read
    from it (CLI args are ignored for that alias).  On first build the
    CLI args are used and ``_meta`` is written.

    Args:
        arxiv_file: Path to ArXiv papers JSON.
        kb: Knowledge base name.
        alias: Single alias to update (omit to update all aliases).
        qdrant_host: Qdrant server host.
        qdrant_port: Qdrant server port.
        embedding_model: (deprecated, ignored) Configured on embeddings service.
        embeddings_url: URL of the embeddings microservice.
        chunking_strategy: e.g. fixed_token, code, section_aware.
        chunk_size: Chunk size (tokens or characters).
        chunk_overlap: Overlap between chunks.
    """
    _settings = get_settings()
    kb_registry = get_knowledge_bases()

    # Resolve defaults from config
    qdrant_host = qdrant_host or _settings.qdrant_host
    qdrant_port = qdrant_port or _settings.qdrant_port
    embedding_model = embedding_model or _settings.embedding_model

    if kb not in kb_registry:
        available = ", ".join(kb_registry) or "(none)"
        print(f"Error: knowledge base '{kb}' not found. Available: {available}")
        sys.exit(1)

    kb_cfg = kb_registry[kb]
    chunking_strategy = chunking_strategy or kb_cfg.chunking_strategy
    chunk_size = chunk_size if chunk_size is not None else kb_cfg.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else kb_cfg.chunk_overlap

    arxiv_path = Path(arxiv_file)
    if not arxiv_path.exists():
        print(f"Error: ArXiv file not found: {arxiv_path}")
        sys.exit(1)

    print("=" * 60)
    print("Building CHAT index from ArXiv papers  [incremental mode]")
    print("=" * 60)

    # Load data
    print(f"\nLoading papers from: {arxiv_path}")
    with open(arxiv_path, encoding="utf-8") as f:
        papers = json.load(f)
    print(f"Loaded {len(papers)} papers")

    # Determine which aliases to process
    if alias:
        aliases_to_process = [alias]
    elif kb in kb_registry:
        aliases_to_process = kb_registry[kb].aliases
    else:
        aliases_to_process = [_settings.default_alias]

    # Initialize embedding service
    print(f"\nInitializing embedding service: {embedding_model}")
    embedding_service = EmbeddingService(
        embedding_model,
        device="cpu",
        embeddings_url=embeddings_url,
    )

    for current_alias in aliases_to_process:
        qdrant_alias = f"{kb}_{current_alias}"
        print(f"\n--- Processing alias: {qdrant_alias} ---")

        helper = QdrantVectorStore(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=qdrant_alias,
        )

        # If alias doesn't resolve, this is a first build for this alias
        if not helper.collection_exists():
            collection_name = f"{kb}_{_timestamp()}"
            print(f"  Creating new collection: {collection_name}")
            vs = QdrantVectorStore(
                host=qdrant_host,
                port=qdrant_port,
                collection_name=collection_name,
            )
            vs.create_collection(dimension=embedding_service.dimension)
            vs.write_meta(
                payload={
                    "build_config": {
                        "chunking_strategy": chunking_strategy,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "embedding_model": embedding_model,
                    },
                    "kb_name": kb,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                dimension=embedding_service.dimension,
            )
            vs.update_alias(qdrant_alias, collection_name)
            target_store = vs
            build_cfg = {
                "chunking_strategy": chunking_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
        else:
            meta = helper.read_meta()
            if meta and "build_config" in meta:
                build_cfg = meta["build_config"]
                print(f"  Using build config from _meta: {build_cfg}")
            else:
                build_cfg = {
                    "chunking_strategy": chunking_strategy,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                }
                print(f"  No _meta found, using CLI defaults: {build_cfg}")
            target_store = helper

        # Ensure collection exists (idempotent)
        target_store.create_collection(
            dimension=embedding_service.dimension,
            force_recreate=False,
        )

        # Chunk and embed documents using this alias's build config
        effective_chunk_size = build_cfg.get("chunk_size", chunk_size)
        effective_chunk_overlap = build_cfg.get("chunk_overlap", chunk_overlap)
        effective_strategy = build_cfg.get("chunking_strategy", chunking_strategy)

        _STRATEGY_TO_TASK = {
            "fixed_token": "chat",
            "code": "code",
            "section_aware": "section_aware",
        }
        task = _STRATEGY_TO_TASK.get(effective_strategy, "chat")
        chunker = get_chunker(
            task=task,
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_chunk_overlap,
        )

        all_chunks = []
        all_metadatas = []
        all_ids: list[str] = []

        for i, paper in enumerate(papers, 1):
            full_text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
            chunks = chunker.chunk(full_text)

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append(
                    {
                        "task": "chat",
                        "source": "arxiv",
                        "arxiv_id": paper["arxiv_id"],
                        "title": paper["title"],
                        "primary_category": paper["primary_category"],
                        "published": paper["published"],
                    }
                )
                all_ids.append(
                    str(uuid.uuid5(_POINT_ID_NS, f"arxiv:{paper['arxiv_id']}:{chunk_idx}"))
                )

            if i % 20 == 0:
                print(f"  Processed {i}/{len(papers)} papers...")

        print(f"  Total chunks created: {len(all_chunks)}")

        # Generate embeddings in batches
        print("  Generating embeddings...")
        batch_size = 32
        for i in range(0, len(all_chunks), batch_size):
            batch_chunks = all_chunks[i : i + batch_size]
            batch_metadata = all_metadatas[i : i + batch_size]
            batch_ids = all_ids[i : i + batch_size]

            embeddings = embedding_service.embed_documents(batch_chunks)
            target_store.add_documents(
                documents=batch_chunks,
                embeddings=embeddings,
                metadatas=batch_metadata,
                ids=batch_ids,
            )

            print(f"  Added batch {i // batch_size + 1}/{(len(all_chunks) - 1) // batch_size + 1}")

        info = target_store.get_collection_info()
        print(f"  Alias '{qdrant_alias}' -- {info['points_count']} points total")

    print("\n" + "=" * 60)
    print("CHAT index build complete!")
    print("=" * 60)


if __name__ == "__main__":
    import fire

    fire.Fire(build_arxiv)
