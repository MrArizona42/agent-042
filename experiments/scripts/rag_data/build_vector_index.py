"""Build vector indices from collected data.

Loads documents, chunks them, generates embeddings, and stores in Qdrant.
Creates separate collections for different tasks (chat, code).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add src to path to import rag module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from rag.chunking import get_chunker
from rag.embeddings import EmbeddingService
from rag.vector_store import QdrantVectorStore


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
    force_recreate: bool = False,
):
    """Build vector index for chat task from ArXiv papers.

    Args:
        arxiv_file: Path to arxiv_papers.json
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        embedding_model: Embedding model name
        force_recreate: If True, delete existing collection
    """
    print("=" * 60)
    print("Building CHAT index from ArXiv papers")
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

    # Create collection
    vector_store.create_collection(
        dimension=embedding_service.dimension,
        force_recreate=force_recreate,
    )

    # Chunk and embed documents
    print("\nChunking and embedding documents...")
    chunker = get_chunker(task="chat", chunk_size=512, chunk_overlap=50)

    all_chunks = []
    all_metadatas = []

    for i, paper in enumerate(papers, 1):
        # Combine title and abstract for better context
        full_text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"

        # Chunk the document
        chunks = chunker.chunk(full_text)

        # Create metadata for each chunk
        for chunk in chunks:
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

        if i % 20 == 0:
            print(f"  Processed {i}/{len(papers)} papers...")

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
    force_recreate: bool = False,
):
    """Build vector index for code task from PyTorch docs.

    Args:
        pytorch_docs_file: Path to pytorch_docs.json
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        embedding_model: Embedding model name
        force_recreate: If True, delete existing collection
    """
    print("=" * 60)
    print("Building CODE index from PyTorch docs")
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
        collection_name="code_documents",
    )

    # Create collection
    vector_store.create_collection(
        dimension=embedding_service.dimension,
        force_recreate=force_recreate,
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

    # Print summary
    info = vector_store.get_collection_info()
    print("\n" + "=" * 60)
    print("CODE index build complete!")
    print("  Collection: code_documents")
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
        help="Force recreate collections (deletes existing data)",
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
            force_recreate=args.force_recreate,
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
            force_recreate=args.force_recreate,
        )

    print("\n✅ All requested indices built successfully!")


if __name__ == "__main__":
    main()
