"""Build PyTorch docs vector index (atomic-replace strategy).

Loads PyTorch documentation, chunks it, generates embeddings, and stores
in a new timestamped Qdrant collection.  A staging alias is used during
the build, then the production alias is atomically swapped.

Usage::

    python -m experiments.scripts.rag_data.build_pytorch_docs_index \
        --pytorch_docs_file assets/rag_data/pytorch_docs/pytorch_docs.json \
        --kb pytorch_docs --alias champion
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to path to import rag module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from rag.chunking import get_chunker
from rag.embeddings import EmbeddingService
from rag.vector_store import QdrantVectorStore
from shared.config import get_knowledge_bases, get_settings


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_pytorch_docs(
    pytorch_docs_file: str = "assets/rag_data/pytorch_docs/pytorch_docs.json",
    kb: str = "pytorch_docs",
    alias: str | None = None,
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
    embedding_model: str | None = None,
    embeddings_url: str | None = None,
    chunking_strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> None:
    """Build vector index for PyTorch docs (replace mode).

    A fresh timestamped collection is built.  A staging alias is created
    during the build, then the production alias is atomically swapped
    to the new collection.

    Args:
        pytorch_docs_file: Path to PyTorch docs JSON.
        kb: Knowledge base name.
        alias: Alias to target (e.g. champion).
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
    alias = alias or _settings.default_alias

    if kb not in kb_registry:
        available = ", ".join(kb_registry) or "(none)"
        print(f"Error: knowledge base '{kb}' not found. Available: {available}")
        sys.exit(1)

    kb_cfg = kb_registry[kb]
    chunking_strategy = chunking_strategy or kb_cfg.chunking_strategy
    chunk_size = chunk_size if chunk_size is not None else kb_cfg.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else kb_cfg.chunk_overlap

    docs_path = Path(pytorch_docs_file)
    if not docs_path.exists():
        print(f"Error: PyTorch docs file not found: {docs_path}")
        sys.exit(1)

    qdrant_alias = f"{kb}_{alias}"
    staging_alias = f"{qdrant_alias}_staging"
    collection_name = f"{kb}_{_timestamp()}"

    print("=" * 60)
    print("Building CODE index from PyTorch docs  [replace mode]")
    print(f"  target alias: {qdrant_alias}")
    print(f"  staging alias: {staging_alias}")
    print(f"  new collection: {collection_name}")
    print("=" * 60)

    # Load data
    print(f"\nLoading docs from: {docs_path}")
    with open(docs_path, encoding="utf-8") as f:
        docs = json.load(f)
    print(f"Loaded {len(docs)} documentation pages")

    # Initialize services
    print(f"\nInitializing embedding service: {embedding_model}")
    embedding_service = EmbeddingService(
        embedding_model,
        device="cpu",
        embeddings_url=embeddings_url,
    )

    # Read existing build config from current champion (if any)
    helper = QdrantVectorStore(
        host=qdrant_host,
        port=qdrant_port,
        collection_name=qdrant_alias,
    )
    if helper.collection_exists():
        meta = helper.read_meta()
        if meta and "build_config" in meta:
            build_cfg = meta["build_config"]
            print(f"  Reusing build config from current {qdrant_alias}: {build_cfg}")
            chunking_strategy = build_cfg.get("chunking_strategy", chunking_strategy)
            chunk_size = build_cfg.get("chunk_size", chunk_size)
            chunk_overlap = build_cfg.get("chunk_overlap", chunk_overlap)
        else:
            print("  No _meta in current champion; using CLI defaults.")

    # Create new collection
    print(f"\nConnecting to Qdrant at {qdrant_host}:{qdrant_port}")
    vector_store = QdrantVectorStore(
        host=qdrant_host,
        port=qdrant_port,
        collection_name=collection_name,
    )
    vector_store.create_collection(dimension=embedding_service.dimension)

    # Write _meta to the new collection
    vector_store.write_meta(
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

    # Point the staging alias at the new collection
    vector_store.update_alias(staging_alias, collection_name)

    # Chunk and embed documents
    print("\nChunking and embedding documents...")
    _STRATEGY_TO_TASK = {"fixed_token": "chat", "code": "code", "section_aware": "section_aware"}
    task = _STRATEGY_TO_TASK.get(chunking_strategy or "code", "code")
    chunker = get_chunker(task=task, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    all_chunks = []
    all_metadatas = []

    for i, doc in enumerate(docs, 1):
        chunks = chunker.chunk(doc["content"])

        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append(
                {
                    "task": "code",
                    "source": "pytorch_docs",
                    "url": doc["url"],
                    "title": doc["title"],
                    "scraped_at": doc["scraped_at"],
                }
            )

        if i % 10 == 0:
            print(f"  Processed {i}/{len(docs)} pages...")

    print(f"\nTotal chunks created: {len(all_chunks)}")

    # Generate embeddings in batches
    print("Generating embeddings...")
    batch_size = 32
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i : i + batch_size]
        batch_metadata = all_metadatas[i : i + batch_size]

        embeddings = embedding_service.embed_documents(batch_chunks)
        vector_store.add_documents(
            documents=batch_chunks,
            embeddings=embeddings,
            metadatas=batch_metadata,
        )

        print(f"  Added batch {i // batch_size + 1}/{(len(all_chunks) - 1) // batch_size + 1}")

    # ---- Atomic alias swap ------------------------------------------------
    old_target = helper.resolve_alias(qdrant_alias)

    if old_target is None:
        collections = vector_store.client.get_collections().collections
        if any(c.name == qdrant_alias for c in collections):
            print(f"Migrating: deleting legacy collection '{qdrant_alias}'")
            vector_store.delete_collection(qdrant_alias)

    # Point the production alias to the freshly-built collection
    print(f"Swapping alias '{qdrant_alias}' -> '{collection_name}'")
    vector_store.update_alias(qdrant_alias, collection_name)

    # Print summary
    info = vector_store.get_collection_info()
    print("\n" + "=" * 60)
    print("CODE index build complete!")
    print(f"  Alias: {qdrant_alias} -> {collection_name}")
    print(f"  Staging alias: {staging_alias} -> {collection_name}")
    print(f"  Total documents: {info['points_count']}")
    print("=" * 60)


if __name__ == "__main__":
    import fire

    fire.Fire(build_pytorch_docs)
