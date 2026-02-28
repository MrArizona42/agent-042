"""Build vector indices from collected data.

Loads documents, chunks them, generates embeddings, and stores in Qdrant.
Creates separate collections for different tasks (chat, code).

Update modes
------------
* ``merge``  – add new documents to the existing collection (upsert).
               Used by the daily ArXiv DAG so that old papers are preserved.
* ``replace`` – build into a staging collection, then atomically swap a
               Qdrant alias so live queries see zero downtime.
               Used by the weekly PyTorch-docs DAG.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

# Add src to path to import rag module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from rag.chunking import get_chunker
from rag.embeddings import EmbeddingService
from rag.vector_store import QdrantVectorStore

# Arbitrary but fixed namespace for UUID5-based point IDs.  Must remain
# constant across runs so the same (source, chunk_index) pair always
# produces the same UUID, enabling upsert-based deduplication.
_POINT_ID_NS = uuid.UUID("b8c9d0e1-f2a3-4b5c-6d7e-8f9a0b1c2d3e")


def load_arxiv_papers(json_file: Path) -> list[dict]:
    """Load ArXiv papers from JSON file."""
    with open(json_file, encoding="utf-8") as f:
        return json.load(f)


def load_pytorch_docs(json_file: Path) -> list[dict]:
    """Load PyTorch docs from JSON file."""
    with open(json_file, encoding="utf-8") as f:
        return json.load(f)


def build_chat_index(
    arxiv_file: Path,
    qdrant_host: str,
    qdrant_port: int,
    embedding_model: str,
):
    """Build / update vector index for chat task from ArXiv papers (merge mode).

    New papers are upserted into the existing ``chat_documents`` collection so
    that previously-ingested papers are preserved across daily runs.
    Deterministic UUID-based point IDs prevent duplicate chunks.

    Args:
        arxiv_file: Path to arxiv_papers.json
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        embedding_model: Embedding model name
    """
    print("=" * 60)
    print("Building CHAT index from ArXiv papers  [merge mode]")
    print("=" * 60)

    # Load data
    print(f"\nLoading papers from: {arxiv_file}")
    papers = load_arxiv_papers(arxiv_file)
    print(f"Loaded {len(papers)} papers")

    # Initialize services
    print(f"\nInitializing embedding service: {embedding_model}")
    embedding_service = EmbeddingService(embedding_model, device="cpu")

    print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
    vector_store = QdrantVectorStore(
        host=qdrant_host,
        port=qdrant_port,
        collection_name="chat_documents",
    )

    # Create collection only if it doesn't exist (no force-recreate)
    vector_store.create_collection(
        dimension=embedding_service.dimension,
        force_recreate=False,
    )

    # Chunk and embed documents
    print("\nChunking and embedding documents...")
    chunker = get_chunker(task="chat", chunk_size=512, chunk_overlap=50)

    all_chunks = []
    all_metadatas = []
    all_ids: list[str] = []

    for i, paper in enumerate(papers, 1):
        # Combine title and abstract for better context
        full_text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"

        # Chunk the document
        chunks = chunker.chunk(full_text)

        # Create metadata and deterministic IDs for each chunk
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

    print(f"\nTotal chunks created: {len(all_chunks)}")

    # Generate embeddings in batches
    print("Generating embeddings...")
    batch_size = 32
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i : i + batch_size]
        batch_metadata = all_metadatas[i : i + batch_size]
        batch_ids = all_ids[i : i + batch_size]

        embeddings = embedding_service.embed_documents(batch_chunks)
        vector_store.add_documents(
            documents=batch_chunks,
            embeddings=embeddings,
            metadatas=batch_metadata,
            ids=batch_ids,
        )

        print(f"  Added batch {i // batch_size + 1}/{(len(all_chunks) - 1) // batch_size + 1}")

    # Print summary
    info = vector_store.get_collection_info()
    print("\n" + "=" * 60)
    print("CHAT index build complete!")
    print("  Collection: chat_documents")
    print(f"  Total documents: {info['points_count']}")
    print("=" * 60)


def build_code_index(
    pytorch_docs_file: Path,
    qdrant_host: str,
    qdrant_port: int,
    embedding_model: str,
):
    """Build vector index for code task from PyTorch docs (replace mode).

    A fresh staging collection is built and then the ``code_documents`` alias
    is atomically swapped to point to it.  The previous collection is deleted
    only after the swap succeeds, guaranteeing zero downtime.

    Args:
        pytorch_docs_file: Path to pytorch_docs.json
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        embedding_model: Embedding model name
    """
    alias_name = "code_documents"
    staging_name = f"code_documents_{time.time_ns()}"

    print("=" * 60)
    print("Building CODE index from PyTorch docs  [replace mode]")
    print(f"  staging collection: {staging_name}")
    print("=" * 60)

    # Load data
    print(f"\nLoading docs from: {pytorch_docs_file}")
    docs = load_pytorch_docs(pytorch_docs_file)
    print(f"Loaded {len(docs)} documentation pages")

    # Initialize services
    print(f"\nInitializing embedding service: {embedding_model}")
    embedding_service = EmbeddingService(embedding_model, device="cpu")

    print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}")
    vector_store = QdrantVectorStore(
        host=qdrant_host,
        port=qdrant_port,
        collection_name=staging_name,
    )

    # Create the staging collection (never force-recreate — name is unique)
    vector_store.create_collection(
        dimension=embedding_service.dimension,
        force_recreate=False,
    )

    # Chunk and embed documents
    print("\nChunking and embedding documents...")
    chunker = get_chunker(task="code", chunk_size=800, chunk_overlap=100)

    all_chunks = []
    all_metadatas = []

    for i, doc in enumerate(docs, 1):
        # Chunk the content
        chunks = chunker.chunk(doc["content"])

        # Create metadata for each chunk
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
    old_target = vector_store.resolve_alias(alias_name)

    if old_target is None:
        # First run or migration from a pre-alias collection.
        # If a legacy collection with the same name as the alias exists,
        # remove it so the alias name becomes available.
        # Check for a legacy collection whose name equals the alias name.
        # We must query the real collections list (not collection_exists())
        # because collection_exists() resolves aliases too, and here we
        # specifically need to know if a *collection* is occupying the name.
        collections = vector_store.client.get_collections().collections
        if any(c.name == alias_name for c in collections):
            print(f"Migrating: deleting legacy collection '{alias_name}'")
            vector_store.delete_collection(alias_name)

    # Point the alias to the freshly-built staging collection
    print(f"Swapping alias '{alias_name}' → '{staging_name}'")
    vector_store.update_alias(alias_name, staging_name)

    # Clean up the old collection (if any)
    if old_target:
        print(f"Deleting old collection '{old_target}'")
        vector_store.delete_collection(old_target)

    # Print summary
    info = vector_store.get_collection_info()
    print("\n" + "=" * 60)
    print("CODE index build complete!")
    print(f"  Alias: {alias_name} → {staging_name}")
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
        help="Embedding model to use",
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

        build_chat_index(
            arxiv_file=args.arxiv_file,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            embedding_model=args.embedding_model,
        )

    if args.task in ["code", "both"]:
        if not args.pytorch_docs_file.exists():
            print(f"Error: PyTorch docs file not found: {args.pytorch_docs_file}")
            print(
                "Scrape PyTorch docs first"
                " using experiments/scripts/prefetch_assets.ipynb (section 9)"
            )
            sys.exit(1)

        build_code_index(
            pytorch_docs_file=args.pytorch_docs_file,
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            embedding_model=args.embedding_model,
        )

    print("\n✅ All requested indices built successfully!")


if __name__ == "__main__":
    main()
