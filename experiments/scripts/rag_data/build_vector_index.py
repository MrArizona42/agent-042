"""Build vector indices from collected data.

Loads documents, chunks them, generates embeddings, and stores in Qdrant.
Creates separate collections for different knowledge bases.

Update modes
------------
* ``incremental``  – add new documents to every resolved alias (upsert).
                     Used by the daily ArXiv DAG so that old papers are preserved.
* ``replace``      – build into a new timestamped collection, then atomically
                     swap the Qdrant alias.  Used by the weekly PyTorch-docs DAG.

Collection naming
-----------------
* Physical collections: ``{kb}_{timestamp}``  (e.g. ``pytorch_docs_20260314_120000``).
* Aliases: ``{kb}_{role}``  (e.g. ``arxiv_champion``).
* Staging aliases during rebuild: ``{kb}_{role}_staging``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add src to path to import rag module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from rag.chunking import get_chunker
from rag.embeddings import EmbeddingService
from rag.vector_store import QdrantVectorStore
from shared.config import get_knowledge_bases

# Arbitrary but fixed namespace for UUID5-based point IDs.  Must remain
# constant across runs so the same (source, chunk_index) pair always
# produces the same UUID, enabling upsert-based deduplication.
_POINT_ID_NS = uuid.UUID("b8c9d0e1-f2a3-4b5c-6d7e-8f9a0b1c2d3e")


def _timestamp() -> str:
    """Return a UTC timestamp string suitable for collection names."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_arxiv_papers(json_file: Path) -> list[dict]:
    """Load ArXiv papers from JSON file."""
    with open(json_file, encoding="utf-8") as f:
        return json.load(f)


def load_pytorch_docs(json_file: Path) -> list[dict]:
    """Load PyTorch docs from JSON file."""
    with open(json_file, encoding="utf-8") as f:
        return json.load(f)


# =========================================================================
# Incremental strategy (e.g. ArXiv)
# =========================================================================


def build_chat_index(
    arxiv_file: Path,
    qdrant_host: str,
    qdrant_port: int,
    embedding_model: str,
    embeddings_url: str | None = None,
    *,
    kb_name: str = "arxiv",
    alias: str | None = None,
    chunking_strategy: str = "fixed_token",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
):
    """Build / update vector index for ArXiv papers (incremental mode).

    For each resolved alias of the KB, new papers are upserted into the
    existing collection.  If the collection already has a ``_meta`` point
    the build config is read from it (CLI args are ignored for that alias).
    On first build the CLI args are used and ``_meta`` is written.

    If *alias* is specified only that single alias is updated; otherwise
    all aliases registered for the KB are updated.
    """
    print("=" * 60)
    print("Building CHAT index from ArXiv papers  [incremental mode]")
    print("=" * 60)

    # Load data
    print(f"\nLoading papers from: {arxiv_file}")
    papers = load_arxiv_papers(arxiv_file)
    print(f"Loaded {len(papers)} papers")

    # Determine which aliases to process
    kb_registry = get_knowledge_bases()
    if alias:
        aliases_to_process = [alias]
    elif kb_name in kb_registry:
        aliases_to_process = kb_registry[kb_name].aliases
    else:
        aliases_to_process = ["champion"]

    # Initialize embedding service
    print(f"\nInitializing embedding service: {embedding_model}")
    embedding_service = EmbeddingService(
        embedding_model, device="cpu", embeddings_url=embeddings_url,
    )

    for current_alias in aliases_to_process:
        qdrant_alias = f"{kb_name}_{current_alias}"
        print(f"\n--- Processing alias: {qdrant_alias} ---")

        helper = QdrantVectorStore(
            host=qdrant_host, port=qdrant_port, collection_name=qdrant_alias,
        )

        # If alias doesn't resolve, this is a first build for this alias
        if not helper.collection_exists():
            # Create new timestamped collection
            collection_name = f"{kb_name}_{_timestamp()}"
            print(f"  Creating new collection: {collection_name}")
            vs = QdrantVectorStore(
                host=qdrant_host, port=qdrant_port, collection_name=collection_name,
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
                    "kb_name": kb_name,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                dimension=embedding_service.dimension,
            )
            # Point alias at this new collection
            vs.update_alias(qdrant_alias, collection_name)
            target_store = vs
            build_cfg = {
                "chunking_strategy": chunking_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            }
        else:
            # Read _meta from existing collection to get build config
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
            dimension=embedding_service.dimension, force_recreate=False,
        )

        # Chunk and embed documents using this alias's build config
        effective_chunk_size = build_cfg.get("chunk_size", chunk_size)
        effective_chunk_overlap = build_cfg.get("chunk_overlap", chunk_overlap)
        effective_strategy = build_cfg.get("chunking_strategy", chunking_strategy)

        task = "chat" if effective_strategy == "fixed_token" else effective_strategy
        chunker = get_chunker(
            task=task, chunk_size=effective_chunk_size, chunk_overlap=effective_chunk_overlap,
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

            print(
                f"  Added batch "
                f"{i // batch_size + 1}/{(len(all_chunks) - 1) // batch_size + 1}"
            )

        info = target_store.get_collection_info()
        print(f"  Alias '{qdrant_alias}' — {info['points_count']} points total")

    print("\n" + "=" * 60)
    print("CHAT index build complete!")
    print("=" * 60)


# =========================================================================
# Replace strategy (e.g. PyTorch docs)
# =========================================================================


def build_code_index(
    pytorch_docs_file: Path,
    qdrant_host: str,
    qdrant_port: int,
    embedding_model: str,
    embeddings_url: str | None = None,
    *,
    kb_name: str = "pytorch_docs",
    alias: str = "champion",
    chunking_strategy: str = "code",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
):
    """Build vector index for PyTorch docs (replace mode).

    A fresh timestamped collection is built.  A staging alias
    ``{kb}_{alias}_staging`` is created during the build, then the
    production alias ``{kb}_{alias}`` is atomically swapped to the
    new collection.

    If a champion already exists, its ``_meta`` is read and the same
    build config is reused (CLI args are ignored).  On first build the
    CLI args are used and ``_meta`` is written.
    """
    qdrant_alias = f"{kb_name}_{alias}"
    staging_alias = f"{qdrant_alias}_staging"
    collection_name = f"{kb_name}_{_timestamp()}"

    print("=" * 60)
    print("Building CODE index from PyTorch docs  [replace mode]")
    print(f"  target alias: {qdrant_alias}")
    print(f"  staging alias: {staging_alias}")
    print(f"  new collection: {collection_name}")
    print("=" * 60)

    # Load data
    print(f"\nLoading docs from: {pytorch_docs_file}")
    docs = load_pytorch_docs(pytorch_docs_file)
    print(f"Loaded {len(docs)} documentation pages")

    # Initialize services
    print(f"\nInitializing embedding service: {embedding_model}")
    embedding_service = EmbeddingService(
        embedding_model, device="cpu", embeddings_url=embeddings_url,
    )

    # Read existing build config from current champion (if any)
    helper = QdrantVectorStore(
        host=qdrant_host, port=qdrant_port, collection_name=qdrant_alias,
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
        host=qdrant_host, port=qdrant_port, collection_name=collection_name,
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
            "kb_name": kb_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        dimension=embedding_service.dimension,
    )

    # Point the staging alias at the new collection
    vector_store.update_alias(staging_alias, collection_name)

    # Chunk and embed documents
    print("\nChunking and embedding documents...")
    task = "code" if chunking_strategy == "code" else chunking_strategy
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
    # Resolve old target for cleanup
    old_target = helper.resolve_alias(qdrant_alias)

    if old_target is None:
        # First run or migration from a legacy collection.
        collections = vector_store.client.get_collections().collections
        if any(c.name == qdrant_alias for c in collections):
            print(f"Migrating: deleting legacy collection '{qdrant_alias}'")
            vector_store.delete_collection(qdrant_alias)

    # Point the production alias to the freshly-built collection
    print(f"Swapping alias '{qdrant_alias}' → '{collection_name}'")
    vector_store.update_alias(qdrant_alias, collection_name)

    # Print summary
    info = vector_store.get_collection_info()
    print("\n" + "=" * 60)
    print("CODE index build complete!")
    print(f"  Alias: {qdrant_alias} → {collection_name}")
    print(f"  Staging alias: {staging_alias} → {collection_name}")
    print(f"  Total documents: {info['points_count']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Build vector indices for RAG")
    parser.add_argument(
        "--task",
        choices=["chat", "code", "both"],
        default="both",
        help="Which index to build",
    )
    parser.add_argument(
        "--kb",
        default=None,
        help="Knowledge base name (e.g. 'arxiv', 'pytorch_docs')",
    )
    parser.add_argument(
        "--alias",
        default=None,
        help="Alias to target (e.g. 'champion', 'challenger'). "
             "For incremental KBs, omit to update all aliases.",
    )
    parser.add_argument(
        "--chunking-strategy",
        default=None,
        help="Chunking strategy (e.g. 'fixed_token', 'code', 'section_aware')",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size (tokens or characters depending on strategy)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Overlap between chunks",
    )
    parser.add_argument(
        "--arxiv-file",
        type=Path,
        default=Path("../../../assets/rag_data/arxiv/arxiv_papers.json"),
        help="Path to ArXiv papers JSON",
    )
    parser.add_argument(
        "--pytorch-docs-file",
        type=Path,
        default=Path("../../../assets/rag_data/pytorch_docs/pytorch_docs.json"),
        help="Path to PyTorch docs JSON",
    )
    parser.add_argument(
        "--qdrant-host",
        default="localhost",
        help="Qdrant server host",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant server port",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="(deprecated, ignored) Model is now configured on the embeddings service",
    )
    parser.add_argument(
        "--embeddings-url",
        default=None,
        help="URL of the embeddings microservice (defaults to GATEWAY_EMBEDDINGS_URL env var)",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="(deprecated, ignored) Kept for backwards compatibility with cached DAGs",
    )

    args = parser.parse_args()

    # Build requested indices
    if args.task in ["chat", "both"]:
        if not args.arxiv_file.exists():
            print(f"Error: ArXiv file not found: {args.arxiv_file}")
            print(
                "Download ArXiv papers first"
                " using experiments/scripts/prefetch_assets.ipynb (section 8)"
            )
            sys.exit(1)

        extra_kwargs: dict = {}
        if args.kb:
            extra_kwargs["kb_name"] = args.kb
        if args.alias:
            extra_kwargs["alias"] = args.alias
        if args.chunking_strategy:
            extra_kwargs["chunking_strategy"] = args.chunking_strategy
        if args.chunk_size is not None:
            extra_kwargs["chunk_size"] = args.chunk_size
        if args.chunk_overlap is not None:
            extra_kwargs["chunk_overlap"] = args.chunk_overlap

        build_chat_index(
            arxiv_file=args.arxiv_file,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            embedding_model=args.embedding_model,
            embeddings_url=args.embeddings_url,
            **extra_kwargs,
        )

    if args.task in ["code", "both"]:
        if not args.pytorch_docs_file.exists():
            print(f"Error: PyTorch docs file not found: {args.pytorch_docs_file}")
            print(
                "Scrape PyTorch docs first"
                " using experiments/scripts/prefetch_assets.ipynb (section 9)"
            )
            sys.exit(1)

        extra_kwargs = {}
        if args.kb:
            extra_kwargs["kb_name"] = args.kb
        if args.alias:
            extra_kwargs["alias"] = args.alias
        if args.chunking_strategy:
            extra_kwargs["chunking_strategy"] = args.chunking_strategy
        if args.chunk_size is not None:
            extra_kwargs["chunk_size"] = args.chunk_size
        if args.chunk_overlap is not None:
            extra_kwargs["chunk_overlap"] = args.chunk_overlap

        build_code_index(
            pytorch_docs_file=args.pytorch_docs_file,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            embedding_model=args.embedding_model,
            embeddings_url=args.embeddings_url,
            **extra_kwargs,
        )

    print("\n✅ All requested indices built successfully!")


if __name__ == "__main__":
    main()
